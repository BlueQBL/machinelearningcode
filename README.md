# machinelearningcode
> 个人PyTorch学习仓库，仓库内全部代码均经过本地调试验证，可以直接运行，用于记录深度学习学习Demo与实验片段。

## 📚 项目介绍
本仓库用于存放个人学习PyTorch过程中的练习代码，涵盖张量基础操作、自动求导、网络构建、模型训练、数据处理、小案例实验等内容。
代码全部在本地环境跑通，保证可复现，方便自己复盘回顾，也可供有需要的同学参考学习。

## 📁 目录结构
> 仓库会持续新增多个模块文件夹，不同目录存放不同主题的练习代码
> machinelearningcode
├── pytorchstudy # PyTorch 学习总目录
│ ├── pytorchsimuse # PyTorch 基础使用示例
│ ├── module_demo # 网络模块、层组件练习
│ ├── train_demo # 模型训练循环相关示例
│ └── ... # 后续新增各类练习子文件夹
├── other_study # 其他机器学习相关练习（按需新增）
└── README.md
> 
## ✅ 本地运行环境
- Python >= 3.9
- PyTorch（CPU/CUDA版本按本机硬件环境选择安装）
- 依赖：numpy、matplotlib等

### 环境安装
```bash
# 前往pytorch官网复制对应本机环境的安装命令
# pip3 install torch torchvision torchaudio

pip install numpy matplotlib
```
使用方式

1.克隆仓库到本地

git clone https://github.com/BlueQBL/machinelearningcode.git
cd machinelearningcode


2.进入对应主题文件夹，运行脚本
# 示例：进入基础示例目录
cd pytorchstudy/pytorchsimuse

# 直接执行py脚本
python demo.py

代码内容
Tensor 基础：张量创建、索引切片、运算、维度变换
Autograd 自动求导机制示例
网络搭建：nn.Module、Sequential两种写法
数据集加载、训练循环、损失函数、优化器完整流程
各类小 Demo，聚焦理解 PyTorch 底层核心逻辑
后续持续补充更多学习实验代码
💡 备注
代码定位为学习演示，优先保证本地可运行，不做工程化封装
代码内附带个人理解注释，便于复习
仓库会不断新增不同主题文件夹，迭代学习案例
欢迎自取参考学习
