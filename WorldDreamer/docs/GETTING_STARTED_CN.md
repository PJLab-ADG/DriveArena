# WorldDreamer - 入门指南

除非另有说明，以下所有代码均在 `WorldDreamer` 文件夹中运行。

## 概述
- [数据集准备](#数据集准备)
- [训练与测试](#训练与测试)
- [模型库](#模型库)

## 数据集准备

目前我们提供了 [nuScenes 数据集](#nuScenes-数据集) 和 [nuPlan 数据集](#nuPlan-数据集) 的数据加载器。

### nuScenes 数据集

- 请下载官方的 [nuScenes 数据集](https://www.nuscenes.org/download) 并将文件组织如下。
```
${DATASET_ROOT}/nuscenes/
├── maps
├── samples
├── sweeps
└── v1.0-trainval
```

- 通过运行以下命令安装 `nuscenes-devkit`：
```shell
pip install nuscenes-devkit==1.1.11
```
- 运行以下命令生成 **(关键帧/样本)** 的 ann_file，这可能需要几个小时：

```shell
python -m tools.create_data nuscenes \
--root-path /path/to/nuscenes --out-dir ./data/nuscenes_mmdet3d-t-keyframes/ \
--extra-tag nuscenes --only_info
```
- 运行以下命令生成 (12hz/扫描) 的 ann_file，这可能需要更长的时间。我们使用它们来训练模型。
	- 首先，按照 ASAP 的说明生成 nuScenes 的插值注释。
    
    注意：需要修改 ASAP 中的以下代码：
	- 在 sAP3D/nusc_annotation_generator.py 中，请注释掉 line357，并将 line101 修改为 val_scene_ids = splits['val'] + splits['train']。
	- 修改 scripts/ann_generator.sh 中的数据集路径为你自定义的数据集路径。
然后，你可以在 ASAP 根目录下运行以下命令：
