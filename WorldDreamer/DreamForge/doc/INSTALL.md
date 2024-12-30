# Installation

All the codes are tested in the following environment:

Linux (tested on Ubuntu 22.04)

CUDA 11.8

```bash
# Install a virtual environment.
conda create -n dreamforge python=3.9
conda activate dreamforge
```

Install `torch==2.0.0+cu118`, `torchvision==0.15.1+cu118`, and `torchaudio==2.0.1+cu118`

Install the source code for other , with `cd ${FOLDER}; pip install -e .`

```
# Install third-party packages
third_party/
├── mmdetection3d
└── diffusers -> based on v0.17.1 (afcca3916)
```

1. mmdetection3d
```bash
pip install mmcv-full==1.6.0
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0
cd mmdetection3d
pip install -v -e .
```

2. xformer
```bash
pip install xformers==0.0.19
```


Install the dependencies
```bash
cd ..
pip install -r requirements.txt
```