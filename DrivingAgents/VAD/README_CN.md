<div align="center">   
  
# 使用 [VAD](https://github.com/hustvl/VAD) 作为驾驶代理

</div>


## 概览
- [更新日志](#更新日志)
- [安装](#安装)
- [入门指南](#入门指南)


## 更新日志
[2024-11-12] 我们简化了 [VAD](https://github.com/hustvl/VAD)  的输入，移除了 gt 标签输入，并使用 fastapi 从 [World Dreamer](../../WorldDreamer/) 接收输入，并将输出传递给 [TrafficManager](../../TrafficManager/)。

## 安装

**a. 环境：创建一个 conda 虚拟环境并激活它。**
```shell
conda create -n vad python=3.8 -y
conda activate vad
```

**b. Torch：按照[官方指南](https://pytorch.org/)安装 PyTorch 和 torchvision。**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```


**c. GCC：确保 conda 环境中 gcc>=5。**
```shell
# If gcc is not installed:
# conda install -c omgarcia gcc-6 # gcc-6.2

export PATH=YOUR_GCC_PATH/bin:$PATH
# Eg: export PATH=/mnt/gcc-5.4/bin:$PATH
```

**d. CUDA：在安装 MMCV 系列之前，你需要设置 CUDA_HOME（用于在 GPU 上编译一些算子）。**
```shell
export CUDA_HOME=YOUR_CUDA_PATH/
# Eg: export CUDA_HOME=/mnt/cuda-11.1/
```

**e. 安装 mmcv。**
```shell
pip install mmcv-full==1.4.0
# If it's not working, try:
# pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**f. 安装 mmdet and mmseg。**
```
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**g. 安装 numba。**
```shell
conda install numba==0.48.0
```

**h. 安装 timm。**
```shell
pip install timm
```

**i. 从源代码安装 mmdet3d。**
```shell
cd ~
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -v -e .
```

**j. 安装 nuscenes-devkit。**
```shell
pip install nuscenes-devkit==1.1.9
```

**k. 从源代码安装 CAMixerSR。**
```shell
cd VAD
git clone https://github.com/icandle/CAMixerSR.git
# 1. Comment out the  "_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]" in "CAMixerSR/codes/basicsr/archs/__init__.py" 
# 2. Comment out the  "_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]" in "CAMixerSR/codes/basicsr/models/__init__.py" 
```

## 入门指南

### 预训练权重
从这里下载VAD的预训练权重 [here](https://drive.google.com/file/d/1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS/view?usp=sharing)。同时，需要下载 CAMixerSR的权重。下载完成后，把它们放到 `ckpts/` 文件夹中。
```shell
cd VAD
mkdir ckpts
wget 'https://github.com/icandle/CAMixerSR/blob/main/pretrained_models/LightSR/CAMixerSRx4_DF.pth'
```

### 使用 FastAPI 运行服务

```shell
cd VAD
python demo/vad_fast_api.py
```