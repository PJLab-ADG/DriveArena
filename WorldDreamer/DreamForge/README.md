# DreamForge
[Motion-Aware Autoregressive Video Generation for Multi-View Driving Scenes](https://arxiv.org/abs/2409.04003)

**Note**: Some sections of the code are currently being prepared and updated. Please stay tuned.

## Overview
- [Installation](doc/INSTALL.md)
- [Getting Started](doc/GETTING_STARTED.md)
- [Model Zoo](#model-zoo)

## Installation
Please refer to [INSTALL.md](doc/INSTALL.md) for the installation.

## Getting Started
Please refer to [GETTING_STARTED.md](doc/GETTING_STARTED.md) to learn more usage about this project.

## Model Zoo

| Model       | Resolution  | Epoch   | Description |
| ----------- | ----------- | --------|-------------|
| [dreamforge-s](https://huggingface.co/Jianbiao/dreamforge-s/tree/main) | 224x400 | 100 | Single-frame version, trained on nuScenes |
| [dreamforge-t](https://huggingface.co/Jianbiao/dreamforge-t/tree/main) | 224x400 | 100 | Autoregressive video generation, trained on nuScenes| 

## Acknowledgment
We utilized the following repos during development:
- [MagicDrive](https://github.com/cure-lab/MagicDrive)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [diffusers](https://github.com/huggingface/diffusers)
