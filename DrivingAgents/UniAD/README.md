<div align="center">   
  
# Using [UniAD](https://github.com/OpenDriveLab/UniAD) as driving agent.

</div>


<!-- <h3 align="center">
  <a href="https://arxiv.org/abs/2212.10156">arXiv</a> |
  <a href="https://www.youtube.com/watch?v=cyrxJJ_nnaQ">Video</a> |
  <a href="https://opendrivelab.com/e2ead/UniAD_plenary_talk_slides.pdf">Slides</a>
</h3> -->



## Overview
- [Changelog](#changelog)
- [Installation](#installation)
- [Getting Started](#getting-started)


## Changelog
[2024-09-04] We simplified the [UniAD](https://github.com/OpenDriveLab/UniAD) input by removing the gt tag input and using fastapi to receive input from [World Dreamer](../../WorldDreamer/) and pass the output to [TrafficManager](../../TrafficManager/).

## Installation

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n uniad python=3.8 -y
conda activate uniad
```

**b. Torch: Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install cudatoolkit=11.1.1 -c conda-forge
# We use cuda-11.1 by default
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

**f. Install requirements.**
```shell
# Install Dependencies
cd UniAD
pip install -r requirements.txt
```


**g. Install mmdet3d from source code.**
```shell
cd ~
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -v -e .
```

**h. Install CAMixerSR from source code.**
```shell
cd UniAD
git clone https://github.com/icandle/CAMixerSR.git
# 1. Comment out the  "_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]" in "CAMixerSR/codes/basicsr/archs/__init__.py" 
# 2. Comment out the  "_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]" in "CAMixerSR/codes/basicsr/models/__init__.py" 


```

## Getting Started

### Pretrained Weight
```shell
cd UniAD
mkdir ckpts
wget 'https://github.com/icandle/CAMixerSR/blob/main/pretrained_models/LightSR/CAMixerSRx4_DF.pth'
wget 'https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth'
```

### Running service with FastAPI

```shell
cd UniAD
python demo/uniad_fast_api.py
```

