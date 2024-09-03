# World Dreamer

A Controllable Autoregressive Generative Network

**Note**: Some sections of the code are currently being prepared and updated. Please stay tuned.

## Overview
- [Changelog](#changelog)
- [Installation](docs/INSTALL.md) | [环境安装](docs/INSTALL_CN.md)
- [Getting Started](docs/GETTING_STARTED.md) | [使用说明](docs/GETTING_STARTED_CN.md)
- [Model Zoo](#model-zoo)


## Changelog
[2014-09-15] `WorldDreamer` v1.0 is released. We now support single-frame autoregressive generator on `nuScenes` and `nuPlan` datasets.


## Installation
Please refer to [[INSTALL.md](docs/INSTALL.md) | [环境安装](docs/INSTALL_CN.md)] for the installation.

## Getting Started
Please refer to [[GETTING_STARTED.md](docs/GETTING_STARTED.md) | [使用说明](docs/GETTING_STARTED_CN.md)] to learn more usage about this project.

## Model Zoo

| Model       | Iteration   | Description |
| ----------- | ----------- |---------------|
| [SDv1.5_mv_single_ref_nus](https://huggingface.co/jokester-yxm/DriveArena/tree/main/SDv1.5_mv_single_ref_nus) | 200k | single-frame auto-regressive, trained on nuScenes |
| [SDv1.5_mv_single_ref_nus_nup]() (comming soon) | 200k | single-frame auto-regressive, trained on nuScenes + nuPlan | 
