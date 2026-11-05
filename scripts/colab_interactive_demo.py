#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab 交互式 Python 演示脚本
1. 本地交互式：可在 VS Code / Cursor / PyCharm 的 Interactive Window 直接按 Shift+Enter 执行单元格 (# %%)；
2. Colab CLI 远程执行：
    colab run --gpu T4 scripts/colab_interactive_demo.py
    或者：
    colab exec -s <session_name> -f scripts/colab_interactive_demo.py
"""

# %% [markdown]
# # 1. 运行时与硬件检测

# %%
import os
import sys
import platform
import psutil
import torch

print("=" * 60)
print(f"🖥️ 操作系统: {platform.system()} {platform.release()}")
print(f"🐍 Python 版本: {platform.python_version()}")
print(f"⚙️ CPU 核心数: {os.cpu_count()}")
print(f"💾 系统内存: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
print(f"🔥 PyTorch 版本: {torch.__version__}")
print(f"⚡ CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🎮 GPU 设备: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram_bytes = getattr(props, "total_memory", None)
    if vram_bytes is None:
        vram_bytes = props.total_mem
    print(f"🧠 显存大小: {vram_bytes / (1024**3):.2f} GB")
else:
    print("ℹ️ 未检测到 GPU，当前运行在 CPU 模式下。")
print("=" * 60)


# %% [markdown]
# # 2. 算力压测与速度对比 (CPU vs GPU)

# %%
import time

def benchmark_matmul(size=4000):
    print(f"\n[基准测试] 正在计算 {size}x{size} 单精度浮点稠密矩阵乘法...")
    
    # CPU 测试
    cpu_x = torch.randn(size, size, dtype=torch.float32)
    cpu_y = torch.randn(size, size, dtype=torch.float32)
    
    t0 = time.perf_counter()
    cpu_z = torch.matmul(cpu_x, cpu_y)
    t_cpu = time.perf_counter() - t0
    print(f"⏱️ CPU 计算耗时: {t_cpu:.4f} 秒")
    
    # GPU 测试
    if torch.cuda.is_available():
        gpu_x = torch.randn(size, size, device="cuda", dtype=torch.float32)
        gpu_y = torch.randn(size, size, device="cuda", dtype=torch.float32)
        
        # GPU 预热
        _ = torch.matmul(gpu_x[:200, :200], gpu_y[:200, :200])
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        gpu_z = torch.matmul(gpu_x, gpu_y)
        torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        
        speedup = t_cpu / t_gpu
        print(f"⚡ GPU 计算耗时: {t_gpu:.4f} 秒")
        print(f"🚀 GPU 加速比: {speedup:.2f}x 倍")
    else:
        print("⚠️ 未开启 GPU 加速器，跳过 GPU 耗时对比")

benchmark_matmul(size=4000)


# %% [markdown]
# # 3. 极简深度学习训练循环演示 (Toy Neural Network)

# %%
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, in_features=32, hidden=64, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )
        
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 模拟数据
x_train = torch.randn(2000, 32, device=device)
y_train = torch.randn(2000, 1, device=device)

print(f"\n[训练演示] 模型已部署至: {device}")
for epoch in range(1, 6):
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}/5 - Loss: {loss.item():.5f}")

print("\n🎉 交互式 Python 演示完成！")
