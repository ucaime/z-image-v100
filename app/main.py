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
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
# 【新增】引入 safetensors 用于手动读取和过滤权重
from safetensors.torch import load_file

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getcwd()

CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DB_FILE = os.path.join(PROJECT_ROOT, "database", "history.db")
HTML_FILE = os.path.join(BASE_DIR, "index.html")
# 【新增】LoRA 目录
LORA_DIR = os.path.join(PROJECT_ROOT, "models", "LoRA")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# === 数据库管理 ===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # 尝试新增字段，兼容旧数据库
    try:
        c.execute("ALTER TABLE history ADD COLUMN lora_name TEXT")
        c.execute("ALTER TABLE history ADD COLUMN lora_scale REAL")
    except:
        pass
        
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
            lora_name TEXT,
            lora_scale REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_to_history(record):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO history (id, prompt, width, height, steps, seed, filename, timestamp, lora_name, lora_scale)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record['id'], record['prompt'], record['width'], record['height'], 
        record['steps'], record['seed'], record['filename'], record['timestamp'],
        record.get('lora_name'), record.get('lora_scale')
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
        results.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "width": row["width"],
            "height": row["height"],
            "steps": row["steps"],
            "seed": row["seed"],
            "url": f"/outputs/{row['filename']}",
            "timestamp": row["timestamp"],
            "lora_name": row["lora_name"] if "lora_name" in row.keys() else None,
            "lora_scale": row["lora_scale"] if "lora_scale" in row.keys() else 1.0
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

# === 模型加载逻辑 ===
try:
    from diffusers.quantizers import PipelineQuantizationConfig
except ImportError:
    PipelineQuantizationConfig = None

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    return {
        "cache_dir": "./models",
        "model_id": "Tongyi-MAI/Z-Image-Turbo"
    }

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
                    quant_config = PipelineQuantizationConfig(
                        quant_backend="bitsandbytes_4bit", quant_kwargs=bnb_config_kwargs
                    )
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
                try:
                    pipe.vae.enable_tiling()
                except:
                    pass
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
    try:
        get_pipeline()
    except Exception as e:
        print(f"Startup load failed: {e}")
    yield
    print("🛑 Server shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    steps: int = 8 
    guidance_scale: float = 1.0 
    seed: int = -1
    # 【新增】LoRA 参数
    lora_name: str = ""
    lora_scale: float = 1.0

# === 路由定义 ===
@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def read_ui():
    if not os.path.exists(HTML_FILE): return HTMLResponse(content="Error: index.html not found", status_code=404)
    with open(HTML_FILE, "r", encoding="utf-8") as f: return f.read()

@app.get("/history")
async def read_history(page: int = 1, size: int = 20):
    return get_history_list(page, size)

@app.delete("/history/{item_id}")
async def delete_history(item_id: str):
    delete_history_item(item_id)
    return {"status": "success"}

@app.get("/loras")
async def list_loras():
    """获取所有可用 LoRA"""
    loras = []
    if os.path.exists(LORA_DIR):
        files = glob.glob(os.path.join(LORA_DIR, "*.safetensors"))
        for f in files:
            loras.append(os.path.basename(f))
    return loras

@app.post("/generate")
def generate_image(req: GenerateRequest):
    if req.height % 16 != 0 or req.width % 16 != 0:
        raise HTTPException(status_code=400, detail="宽和高必须是 16 的倍数")

    try:
        pipeline = get_pipeline()
        torch.cuda.empty_cache()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # === 核心：LoRA 加载逻辑 ===
        active_lora = None
        adapter_name = "default"
        
        # 1. 卸载旧权重，保证纯净
        try:
            pipeline.unload_lora_weights()
        except:
            pass

        # 2. 加载新 LoRA
        if req.lora_name:
            lora_path = os.path.join(LORA_DIR, req.lora_name)
            if os.path.exists(lora_path):
                print(f"🔌 Loading LoRA: {req.lora_name}")
                try:
                    # 读取原始权重文件
                    state_dict = load_file(lora_path)
                    
                    # 【关键步骤】过滤权重：只保留 U-Net 相关的，剔除 Text Encoder
                    # SDXL LoRA 的 Text Encoder 键通常包含 "text_encoder" 或 "te0", "te1"
                    # 而 U-Net 键通常包含 "unet"
                    unet_keys = {}
                    for k, v in state_dict.items():
                        # 过滤掉 explicitly 属于 text encoder 的 key
                        if "text_encoder" not in k and "lora_te" not in k:
                            unet_keys[k] = v
                    
                    if len(unet_keys) > 0:
                        # 加载过滤后的权重
                        pipeline.load_lora_weights(unet_keys, adapter_name=adapter_name)
                        
                        # 设置权重强度 (使用 set_adapters 而不是 cross_attention_kwargs)
                        pipeline.set_adapters([adapter_name], adapter_weights=[req.lora_scale])
                        active_lora = req.lora_name
                        print(f"   ✅ LoRA loaded successfully (U-Net only, {len(unet_keys)} keys). Scale: {req.lora_scale}")
                    else:
                        print("   ⚠️ No U-Net weights found in LoRA file.")
                        
                except Exception as e:
                    print(f"   ❌ LoRA Load Failed: {e}")
                    print("   ⚠️ Proceeding with base model only.")
                    active_lora = None
            else:
                print(f"⚠️ LoRA file not found: {req.lora_name}")

        seed = req.seed
        if seed == -1: seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        generator = torch.Generator(device).manual_seed(seed)
        
        print(f"Generating: '{req.prompt}' | {req.width}x{req.height} | LoRA: {active_lora}")
        
        with torch.inference_mode():
            image = pipeline(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
                # 注意：这里不再传递 cross_attention_kwargs，因为已经用了 set_adapters
            ).images[0]
        
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath, format="PNG")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        record = {
            "id": unique_id,
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "seed": seed,
            "filename": filename,
            "timestamp": timestamp,
            "lora_name": active_lora,
            "lora_scale": req.lora_scale
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