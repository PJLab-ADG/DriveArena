# Traffic Manager

A Traffic Simulator for Generating Realistic Traffic Flow on Global Street Maps

**Note**: Some sections of the code are currently being prepared and updated. Please stay tuned.

## Overview
- [Changelog](#changelog)
- [Installation](docs/INSTALL.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Model Zoo](#model-zoo)


## Changelog
[2014-09-15] `WorldDreamer` v1.0 is released. We now support single-frame autoregressive generator on `nuScenes` and `nuPlan` datasets.


## Quickstart
Please refer to [quickstart.md](docs/quickstart.md) or the chinese version [quickstart_CN.md](docs/quickstart_CN.md) to start Traffic Manager.

## Getting Started
Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

## Model Zoo

| Model       | Iteration   | Description |
| ----------- | ----------- |---------------|
| [SDv1.5_mv_single_ref_nus](https://huggingface.co/jokester-yxm/DriveArena/tree/main/SDv1.5_mv_single_ref_nus) | 200k | single-frame auto-regressive, trained on nuScenes |
| [SDv1.5_mv_single_ref_nus_nup]() (comming soon) | 200k | single-frame auto-regressive, trained on nuScenes + nuPlan | 
