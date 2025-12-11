# Z-Image Turbo Workstation (V100 Optimized)

这是一个专为 NVIDIA V100 (16GB) 显卡优化的 Z-Image Turbo 本地生成服务。
采用 4-bit NF4 量化存储 + FP32 计算的混合精度策略，解决了 V100 上的黑图、花屏和显存溢出问题。

## 功能特性
- **极低显存占用**：约 8GB VRAM（4-bit 量化）。
- **高质量输出**：FP32 计算精度，无噪点。
- **WebUI 界面**：支持提示词配置、参数调整、历史记录查看。
- **历史管理**：内置 SQLite 数据库，自动保存生成记录。

## 目录结构
- `app/`: 源代码 (main.py, index.html)
- `models/`: 模型缓存
- `outputs/`: 生成的图片文件
- `history.db`: 历史记录数据库
- `config.json`: 配置文件

## 快速开始

### 1. 准备环境
确保安装了 CUDA 12.1 和 Conda。

```bash
conda create -n zimage python=3.10 -y
conda activate zimage
pip install -r requirements.txt
