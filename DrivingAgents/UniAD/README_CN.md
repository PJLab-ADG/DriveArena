<div align="center">   
  
# 使用 [UniAD](https://github.com/OpenDriveLab/UniAD) 作为驾驶代理

</div>


<!-- <h3 align="center">
  <a href="https://arxiv.org/abs/2212.10156">arXiv</a> |
  <a href="https://www.youtube.com/watch?v=cyrxJJ_nnaQ">视频</a> |
  <a href="https://opendrivelab.com/e2ead/UniAD_plenary_talk_slides.pdf">幻灯片</a>
</h3> -->



## 概览
- [更新日志](#更新日志)
- [安装](#安装)
- [入门指南](#入门指南)


## 更新日志
[2024-09-04] 我们简化了 [UniAD](https://github.com/OpenDriveLab/UniAD) 的输入，移除了 gt 标签输入，并使用 fastapi 从 [World Dreamer](../../WorldDreamer/) 接收输入，并将输出传递给 [TrafficManager](../../TrafficManager/)。

## 安装

**a. 环境：创建一个 conda 虚拟环境并激活它。**
```shell
conda create -n uniad python=3.8 -y
conda activate uniad
```

**b. Torch：按照[官方指南](https://pytorch.org/)安装 PyTorch 和 torchvision。**
```shell
conda install cudatoolkit=11.1.1 -c conda-forge
# 我们默认使用 cuda-11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# 推荐 torch>=1.9
```


**c. GCC：确保 conda 环境中 gcc>=5。**
```shell
# 如果未安装 gcc：
# conda install -c omgarcia gcc-6 # gcc-6.2

export PATH=YOUR_GCC_PATH/bin:$PATH
# 例如：export PATH=/mnt/gcc-5.4/bin:$PATH
```

**d. CUDA：在安装 MMCV 系列之前，你需要设置 CUDA_HOME（用于在 GPU 上编译一些算子）。**
```shell
export CUDA_HOME=YOUR_CUDA_PATH/
# 例如：export CUDA_HOME=/mnt/cuda-11.1/
```

**e. 安装 mmcv。**
```shell
pip install mmcv-full==1.4.0
# 如果不起作用，请尝试：
# pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**f. 安装依赖。**
```shell
# 安装依赖
cd UniAD
pip install -r requirements.txt
```


**g. 从源代码安装 mmdet3d。**
```shell
cd ~
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -v -e .
```

**h. 从源代码安装 CAMixerSR。**
```shell
cd UniAD
git clone https://github.com/icandle/CAMixerSR.git
# 1. Comment out the  "_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]" in "CAMixerSR/codes/basicsr/archs/__init__.py" 
# 2. Comment out the  "_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]" in "CAMixerSR/codes/basicsr/models/__init__.py" 
```

## 入门指南

### 预训练权重
```shell
cd UniAD
mkdir ckpts
wget 'https://github.com/icandle/CAMixerSR/raw/refs/heads/main/pretrained_models/LightSR/CAMixerSRx4_DF.pth'
wget 'https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth'
```

### 使用 FastAPI 运行服务

```shell
cd UniAD
python demo/uniad_fast_api.py
```

