from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import DiffusionPipeline, AutoencoderKL
from transformers import AutoModel, BitsAndBytesConfig
import torch
import os
import json
import sqlite3
import time
import uuid
import glob
from typing import List
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from safetensors.torch import load_file

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getcwd()

CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DB_FILE = os.path.join(PROJECT_ROOT, "database", "history.db")
HTML_FILE = os.path.join(BASE_DIR, "index.html")
LORA_DIR = os.path.join(PROJECT_ROOT, "models", "LoRA")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# === 数据库管理 ===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try: c.execute("ALTER TABLE history ADD COLUMN lora_info TEXT")
    except: pass
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            prompt TEXT,
            width INTEGER,
            height INTEGER,
            steps INTEGER,
            seed INTEGER,
            filename TEXT,
            timestamp TEXT,
            lora_info TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_history(record):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # 确保 loras 是纯字典列表
    lora_data = record.get('loras', [])
    lora_json = json.dumps(lora_data)
    c.execute('''
        INSERT INTO history (id, prompt, width, height, steps, seed, filename, timestamp, lora_info)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record['id'], record['prompt'], record['width'], record['height'], 
        record['steps'], record['seed'], record['filename'], record['timestamp'],
        lora_json
    ))
    conn.commit()
    conn.close()

def get_history_list(page=1, size=20):
    offset = (page - 1) * size
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY timestamp DESC LIMIT ? OFFSET ?', (size, offset))
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        loras = []
        if "lora_info" in row.keys() and row["lora_info"]:
            try: loras = json.loads(row["lora_info"])
            except: pass
        elif "lora_name" in row.keys() and row["lora_name"]:
            loras = [{"name": row["lora_name"], "scale": row["lora_scale"]}]

        results.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "width": row["width"],
            "height": row["height"],
            "steps": row["steps"],
            "seed": row["seed"],
            "url": f"/outputs/{row['filename']}",
            "timestamp": row["timestamp"],
            "loras": loras
        })
    return results

def delete_history_item(item_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT filename FROM history WHERE id = ?', (item_id,))
    row = c.fetchone()
    if row:
        file_path = os.path.join(OUTPUT_DIR, row[0])
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass
    c.execute('DELETE FROM history WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()

# === 模型加载 ===
try:
    from diffusers.quantizers import PipelineQuantizationConfig
except ImportError:
    PipelineQuantizationConfig = None

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f: return json.load(f)
        except Exception as e: print(f"Error loading config: {e}")
    return {"cache_dir": "./models", "model_id": "Tongyi-MAI/Z-Image-Turbo"}

model_config = load_config()
pipe = None

def get_pipeline():
    global pipe
    if pipe is None:
        print(f"Loading model {model_config['model_id']}...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(" - [1/4] Pre-loading VAE (FP32)...")
            vae = AutoencoderKL.from_pretrained(
                model_config['model_id'], subfolder="vae",
                torch_dtype=torch.float32, cache_dir=model_config['cache_dir'], trust_remote_code=True
            )

            print(" - [2/4] Pre-loading Text Encoder (FP16)...")
            text_encoder = AutoModel.from_pretrained(
                model_config['model_id'], subfolder="text_encoder",
                torch_dtype=torch.float16, cache_dir=model_config['cache_dir'], trust_remote_code=True
            )

            quant_config = None
            if device == "cuda":
                print(f" - [3/4] Configuring 4-bit NF4 Quantization (FP32 Compute)...")
                bnb_config_kwargs = {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.float32,
                }
                if PipelineQuantizationConfig:
                    quant_config = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs=bnb_config_kwargs)
                else:
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(**bnb_config_kwargs)

            print(" - [4/4] Assembling Pipeline...")
            pipe = DiffusionPipeline.from_pretrained(
                model_config['model_id'],
                vae=vae, text_encoder=text_encoder, quantization_config=quant_config,
                torch_dtype=torch.float32, low_cpu_mem_usage=True, 
                cache_dir=model_config['cache_dir'], trust_remote_code=True
            )
            
            if device == "cuda":
                print("Applying Final Optimizations...")
                pipe.vae.to("cuda")
                pipe.text_encoder.to("cuda")
                try: pipe.vae.enable_tiling()
                except: pass
            else:
                pipe.to(device)
            print(f"✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    return pipe

# === FastAPI App ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting...")
    init_db()
    try: get_pipeline()
    except Exception as e: print(f"Startup load failed: {e}")
    yield
    print("🛑 Server shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class LoraConfig(BaseModel):
    name: str
    scale: float

class GenerateRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    steps: int = 8 
    guidance_scale: float = 1.0 
    seed: int = -1
    loras: List[LoraConfig] = []

# === 路由 ===
@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def read_ui():
    if not os.path.exists(HTML_FILE): return HTMLResponse(content="Error: index.html not found", status_code=404)
    with open(HTML_FILE, "r", encoding="utf-8") as f: return f.read()

@app.get("/history")
async def read_history(page: int = 1, size: int = 20): return get_history_list(page, size)

@app.delete("/history/{item_id}")
async def delete_history(item_id: str):
    delete_history_item(item_id)
    return {"status": "success"}

@app.get("/loras")
async def list_loras():
    loras = []
    if os.path.exists(LORA_DIR):
        files = glob.glob(os.path.join(LORA_DIR, "*.safetensors"))
        for f in files: loras.append(os.path.basename(f))
    return loras

@app.post("/generate")
def generate_image(req: GenerateRequest):
    if req.height % 16 != 0 or req.width % 16 != 0:
        raise HTTPException(status_code=400, detail="Dimensions must be divisible by 16")

    try:
        pipeline = get_pipeline()
        torch.cuda.empty_cache()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # === 核心：稳健的多 LoRA 加载逻辑 ===
        active_adapters = []
        adapter_weights = []
        
        try:
            # 1. 彻底卸载旧权重
            pipeline.unload_lora_weights()
            
            for idx, lora_config in enumerate(req.loras):
                if not lora_config.name: continue
                lora_path = os.path.join(LORA_DIR, lora_config.name)
                if not os.path.exists(lora_path): continue

                adapter_name = f"lora_{idx}"
                print(f"🔌 Loading LoRA [{idx}]: {lora_config.name} (Scale: {lora_config.scale})")
                
                try:
                    # 【核心修正】：统一使用 load_file -> 过滤 -> 转FP32 -> 注入
                    # 这一步解决了 V100 上的 NaN (黑图) 问题和 TextEncoder 不匹配问题
                    state_dict = load_file(lora_path)
                    
                    # 1. 过滤：剔除 Text Encoder 权重
                    # 2. 转换：强制转为 FP32 (解决 NaN 核心)
                    unet_keys = {}
                    for k, v in state_dict.items():
                        if "text" not in k and "te" not in k:
                            unet_keys[k] = v.to(dtype=torch.float32) # <--- 关键！
                    
                    if len(unet_keys) > 0:
                        pipeline.load_lora_weights(unet_keys, adapter_name=adapter_name)
                        active_adapters.append(adapter_name)
                        adapter_weights.append(lora_config.scale)
                        print(f"   ✅ Loaded {len(unet_keys)} keys (FP32 Converted).")
                    else:
                        print("   ⚠️ No valid U-Net keys found.")
                        
                except Exception as e:
                    print(f"   ❌ Failed to load {lora_config.name}: {e}")

            # 激活 Adapter
            if active_adapters:
                pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)

        except Exception as e:
            print(f"❌ Critical LoRA Error: {e}")

        seed = req.seed
        if seed == -1: seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        generator = torch.Generator(device).manual_seed(seed)
        
        print(f"Generating: '{req.prompt}' | {req.width}x{req.height}")
        
        with torch.inference_mode():
            image = pipeline(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
            ).images[0]
        
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath, format="PNG")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 转换 Pydantic 对象为字典
        loras_dict_list = [l.dict() for l in req.loras]

        record = {
            "id": unique_id,
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "seed": seed,
            "filename": filename,
            "timestamp": timestamp,
            "loras": loras_dict_list
        }
        save_to_history(record)
        record["url"] = f"/outputs/{filename}"
        return record
        
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        torch.cuda.empty_cache()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    uvicorn.run(app, host="0.0.0.0", port=8000)