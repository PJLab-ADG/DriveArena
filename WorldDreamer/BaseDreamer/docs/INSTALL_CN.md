# 安装指南

所有代码均在以下环境中测试：

Linux (已在 Ubuntu 22.04 上测试)

CUDA 11.3 或更高版本

```bash
# 安装虚拟环境
conda create -n dreamer python=3.9
conda activate dreamer
```

从源码安装 `nuplan-devkit`
```bash
cd WorldDreamer/BaseDreamer/third_party/nuplan-devkit
pip install -r requirements.txt
pip install -e .
```
安装 `Pytorch==1.10.2` 和 `torchvision==0.11.3`
```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
安装其他第三方包的源码，使用 `cd ${FOLDER}; pip install -e .`
```
# 安装第三方包
third_party/
├── bevfusion -> 基于 db75150
├── diffusers -> 基于 v0.17.1 (afcca3916)
└── xformers -> 对 0.0.19 进行了微小修改，以支持 pytorch1.10.2
```

安装 WorldDreamer/BaseDreamer 的依赖项
```bash
cd ..
pip install -r requirements.txt
```

