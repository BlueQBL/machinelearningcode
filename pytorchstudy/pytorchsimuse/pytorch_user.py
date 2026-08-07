'''
测试PyTorch是否GPU安装成功
并且体验GPU计算速度与CPU计算速度的差异
'''

import torch
import time

# 检查 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 检查 GPU 是否可用
gpu_available = torch.cuda.is_available()
print("GPU 可用:", gpu_available)

# 设备选择
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda") if gpu_available else None

# 创建测试数据
size = 10000
a_cpu = torch.rand(size, size, device=device_cpu)
b_cpu = torch.rand(size, size, device=device_cpu)

# 在 CPU 上计算
start_time = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
end_time = time.time()
print("CPU 计算时间: {:.4f} 秒".format(end_time - start_time))

# 在 GPU 上计算（如果可用）
if gpu_available:
    a_gpu = a_cpu.to(device_gpu)
    b_gpu = b_cpu.to(device_gpu)

    start_time = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # 确保 GPU 计算完成
    end_time = time.time()
    print("GPU 计算时间: {:.4f} 秒".format(end_time - start_time))
else:
    print("未检测到 GPU，跳过 GPU 计算测试")
