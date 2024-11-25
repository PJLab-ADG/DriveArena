<div align="center">   
  
# Using [VAD](https://github.com/hustvl/VAD) as driving agent.

</div>


## Overview
- [Changelog](#changelog)
- [Installation](#installation)
- [Getting Started](#getting-started)


## Changelog
[2024-11-12] We simplified the [VAD](https://github.com/hustvl/VAD) input by removing the gt tag input and using fastapi to receive input from [World Dreamer](../../WorldDreamer/) and pass the output to [TrafficManager](../../TrafficManager/).

## Installation

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n vad python=3.8 -y
conda activate vad
```

**b. Torch: Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```


**c. GCC: Make sure gcc>=5 in conda env.**
```shell
# If gcc is not installed:
# conda install -c omgarcia gcc-6 # gcc-6.2

export PATH=YOUR_GCC_PATH/bin:$PATH
# Eg: export PATH=/mnt/gcc-5.4/bin:$PATH
```

**d. CUDA: Before installing MMCV family, you need to set up the CUDA_HOME (for compiling some operators on the gpu).**
```shell
export CUDA_HOME=YOUR_CUDA_PATH/
# Eg: export CUDA_HOME=/mnt/cuda-11.1/
```

**e. Install mmcv.**
```shell
pip install mmcv-full==1.4.0
# If it's not working, try:
# pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**f. Install mmdet and mmseg.**
```
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**g. Install numba.**
```shell
conda install numba==0.48.0
```

**h. Install timm.**
```shell
pip install timm
```

**i. Install mmdet3d.**
```shell
cd ~
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -v -e .
```

**j. Install nuscenes-devkit.**
```shell
pip install nuscenes-devkit==1.1.9
```

**k. Install CAMixerSR from source code.**
```shell
cd VAD
git clone https://github.com/icandle/CAMixerSR.git
# 1. Comment out the  "_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]" in "CAMixerSR/codes/basicsr/archs/__init__.py" 
# 2. Comment out the  "_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]" in "CAMixerSR/codes/basicsr/models/__init__.py" 
```

## Getting Started

### Pretrained Weight
Download the VAD model [here](https://drive.google.com/file/d/1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS/view?usp=sharing) and the CAMixerSR model. Put them into the `ckpts/` folder
```shell
cd VAD
mkdir ckpts
wget 'https://github.com/icandle/CAMixerSR/blob/main/pretrained_models/LightSR/CAMixerSRx4_DF.pth'
```

### Running service with FastAPI

```shell
cd VAD
python demo/vad_fast_api.py
```