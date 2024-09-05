# WorldDreamer - 入门指南

除非另有说明，以下所有代码均在 `WorldDreamer` 文件夹中运行。

## 概述
- [数据集准备](#数据集准备)
- [预训练权重](#预训练权重)
- [训练与测试](#训练与测试)

## 数据集准备
（使用仿真器时可以跳过这一步骤）

目前我们提供了 [nuScenes 数据集](#nuscenes-数据集) 和 [nuPlan 数据集](#nuplan-数据集) 的数据加载器。

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
- 运行以下命令生成`ann_file` **(12hz/sweep)**，这可能需要更长的时间。我们使用它们来训练模型。
	- 首先，按照 [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md) 的说明生成 nuScenes 的插值注释。
    
        **注意**：需要修改 ASAP 中的以下代码：
        - 在 `sAP3D/nusc_annotation_generator.py` 中，请注释掉 [line357](https://github.com/JeffWang987/ASAP/blob/52316629f2a87ef2ef5bbc634d33e9544b5e39a7/sAP3D/nusc_annotation_generator.py#L357)，并将 [line101](https://github.com/JeffWang987/ASAP/blob/52316629f2a87ef2ef5bbc634d33e9544b5e39a7/sAP3D/nusc_annotation_generator.py#L101) 修改为 `val_scene_ids = splits['val'] + splits['train']`。
        - 修改 `scripts/ann_generator.sh` 中的数据集路径为你自定义的数据集路径。
    
        然后，你可以在 ASAP 根目录下运行以下命令：

        ```
        bash scripts/ann_generator.sh 12 --ann_strategy 'interp' 
        ```

        （可选）为sweep生成advanced annotations。（我们没有观察到interp和advanced之间的主要区别。你可以参考  [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md) 的实现。此步骤可以跳过。）
        
        
        将生成的文件夹重命名为 `interp_12Hz_trainval` 并将其移动到你的 nuScenes 数据集根目录中。

    - 使用以下命令生成 12hz 的 ann_file。
        ```
        python tools/create_data.py nuscenes \
        --root-path /path/to/nuscenes \
        --out-dir ./data/nuscenes_mmdet3d-12Hz \
        --extra-tag nuscenes_interp_12Hz \
        --max-sweeps -1 \
        --version interp_12Hz_trainval
        ```

    - 为了获取包含时间、天气、街道风格、道路结构和外观等元素的详细场景描述，我们提供了使用 GPT-4V 的代码来优化图像描述。在使用之前，请修改 `.pkl` 文件的路径和其他信息，例如 `ChatGPT API` 密钥。
        ```
        python tools/description.py
        ```

    - （可选但推荐）我们建议生成 `.h5` 格式的 BEV 地图缓存文件，以加快数据加载过程。
        ```bash
        # 为val生成地图缓存
        python tools/prepare_map_aux.py +process=val +subfix=12Hz_interp

        # 为train生成地图缓存
        python tools/prepare_map_aux.py +process=train +subfix=12Hz_interp
        ```
        生成缓存文件后，将它们移动到 `./data/nuscenes_map_aux_12Hz_interp`

    
    - 最终的数据结构应如下所示：
        ```
        ${ROOT}/data/
            ├── ...
            ├── nuscenes_mmdet3d-keyframes
            │       ├── nuscenes_infos_train.pkl
            │       └── nuscenes_infos_val.pkl
            ├── nuscenes_mmdet3d-12Hz
            |       ├── nuscenes_interp_12Hz_infos_train.pkl
            |       └── nuscenes_interp_12Hz_infos_val.pkl
            └── nuscenes_map_aux_12Hz_interp  # 来自插值
                    ├── train_200x200_12Hz_interp.h5
                    └── val_200x200_12Hz_interp.h5
        ```

### nuPlan 数据集

- 为确保训练数据的均匀分布，我们从 NuPlan 数据集中选择了 64 个log。此选择包括在拉斯维加斯记录的 21 个日志、在匹兹堡记录的 21 个日志、在波士顿记录的 11 个日志以及在新加坡记录的 11 个日志。这些日志的名称列在 [nuplan.yaml](../tools/data_converter/nuplan.yaml) 中的 `dreamer_train` 和 `dreamer_val` 分类下。请下载官方的 [nuPlan 数据集](https://www.nuscenes.org/nuplan#download) 并将文件组织如下：
    ```
    ${DATASET_ROOT}/nuplan-v1.1/
    ├── sensor_blobs
            ├── ...
            └── ...
    └── splits
            └── trainval
                ├── ...
                └── ...
    ```

- `nuplan-devkit` 需要从源码安装

    ```bash
    cd third_party/nuplan-devkit
    pip install -r requirements.txt
    pip install -e .
    ```

- 为train/val做准备，运行以下命令生成 `ann_file`。
    ```bash 
    python tools/create_data.py nuplan --root-path /path/to/nuplan/dataset/ --version dreamer-trainval --out-dir data/nuplan --split-yaml tools/data_converter/nuplan.yaml
    ```

- 为了获取包含时间、天气、街道风格、道路结构和外观等元素的详细场景描述，我们提供了使用 GPT-4V 的代码来优化图像描述。在使用之前，请修改 `.pkl` 文件的路径和其他信息，例如 `ChatGPT API` 密钥。
    ``` bash
    python tools/description.py
    ```

- （可选但推荐）我们建议生成 BEV 地图的 `.h5` 格式缓存文件，以加快数据加载过程。
    ``` bash
    # 为val生成地图缓存
    python tools/prepare_map_aux_nuplan.py +process=val +subfix=nuplan_map_aux

    # 为train生成地图缓存
    python tools/prepare_map_aux_nuplan.py +process=train +subfix=nuplan_map_aux
    ```
    生成缓存文件后，将它们移动到 `./data/nuplan`

- 最终的数据结构应如下所示：
    ```
    ${ROOT}/data/
    ├── ...
    └── nuplan
            ├── ...
            ├── nuplan_infos_train.pkl
            ├── nuplan_infos_val.pkl
            ├── nuplan_infos_train_with_note.pkl
            ├── nuplan_infos_val_with_note.pkl
            ├── train_200x200_12Hz_interp.h5
            └── val_200x200_12Hz_interp.h5
    ```

## 预训练权重

我们使用了 [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)([备用链接](https://huggingface.co/pt-sk/stable-diffusion-1.5))和 [CLIP-ViT](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) 的预训练权重。

请将它们放在 `${ROOT}/pretrained/` 目录下，如下所示：

```
${ROOT}/pretrained/
        ├── stable-diffusion-v1-5/
        └── CLIP-ViT-B-32-laion2B-s34B-b79K/
```
我们 WorldDreamer 模型的 **预训练权重** 可以在[这里](https://huggingface.co/jokester-yxm/DriveArena/tree/main)下载。更多checkpoints的信息，请查看 [Model Zoo](../README.md/#model-zoo)。

你可以将它们组织成以下形式：
```
${ROOT}/dreamer_pretrained/
        ├── SDv1.5_mv_single_ref_nus
                ├── hydra
                └── weight-S200000
        └── other weights ...
```

## 训练与测试

### 训练

训练单帧自回归版本：
```bash
scripts/dist_train.sh 8 runner=8gpus
```

### 测试

使用预训练权重进行测试：
```bash
python tools/test.py resume_from_checkpoint=./dreamer_pretrained/SDv1.5_mv_single_ref_nus/weight-S200000
```
使用自己的权重进行测试：
```bash
python tools/test.py resume_from_checkpoint=path/to/your/weight
```
在从 OpenStreetMap 裁剪的演示数据上进行测试：
```bash
python tools/test.py runner.validation_index=demo resume_from_checkpoint=path/to/your/weight
```

## Todo

- [ ] 检查 tensorboard 代码
- [x] 检查地图可视化代码
