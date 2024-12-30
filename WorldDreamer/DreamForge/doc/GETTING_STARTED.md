# Getting Started

The following codes are all run in the `DreamForge` folder unless otherwise specified.

## Overview
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Weights](#pretrained-weights)
- [Training & Testing](#training--testing)

The following codes are all run in the `DreamForge` folder unless otherwise specified.

## Dataset Preparation

Currently we provide the dataloader of [nuScenes dataset](#nuscenes-dataset).

### nuScenes Dataset


- Please download the official [nuScenes dataset](https://www.nuscenes.org/download) and organized the files as follows.

```
${DATASET_ROOT}/nuscenes/
├── maps
├── samples
├── sweeps
└── v1.0-trainval
```
- Install the nuscenes-devkit by running the following command:
```shell
pip install nuscenes-devkit==1.1.11
```
- Generate the `ann_file` **(with keyframes / samples)** by running the following command, it may take several hours:
```shell
python -m tools.create_data nuscenes \
--root-path /path/to/nuscenes --out-dir ./data/nuscenes_mmdet3d-t-keyframes/ \
--extra-tag nuscenes --only_info
```
- Generate the `ann_file` **(with 12hz / sweeps)** by running the following command, it may take longer time. We use them to train the model.
    
    - Firstly, follow [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md) to generate interp annotations for nuScenes. 

        **Note**: The following codes in ASAP need to be modified:
        
        - In `sAP3D/nusc_annotation_generator.py`, please comment [line357](https://github.com/JeffWang987/ASAP/blob/52316629f2a87ef2ef5bbc634d33e9544b5e39a7/sAP3D/nusc_annotation_generator.py#L357), and modify [line101](https://github.com/JeffWang987/ASAP/blob/52316629f2a87ef2ef5bbc634d33e9544b5e39a7/sAP3D/nusc_annotation_generator.py#L101) to `val_scene_ids = splits['val'] + splits['train']`.
        
        - Modify the dataset path in `scripts/ann_generator.sh` to your custom dataset path.
    
        Then, you can run the following command in ASAP root:
        ```
        bash scripts/ann_generator.sh 12 --ann_strategy 'interp' 
        ```
        (Optional) Generate advanced annotations for sweeps. (We do not observe major difference between interp and advanced. You can refer to the implementation of [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md). This step can be skipped.)

        Rename the generated folder to `interp_12Hz_trainval` and move it into your nuScenes dataset root.
        
    - Use the following command to generate ann_file with 12hz.
        ```
        python tools/create_data.py nuscenes \
        --root-path /path/to/nuscenes \
        --out-dir ./data/nuscenes_mmdet3d-12Hz \
        --extra-tag nuscenes_interp_12Hz \
        --max-sweeps -1 \
        --version interp_12Hz_trainval
        ```

- To obtain detailed scene descriptions that include elements like time, weather, street style, road structure, and appearance, we provide code to refine the image captions using GPT-4V. Before using, please modify the path to the `.pkl` file and other information such as the `ChatGPT API key`.
    ```
    python tools/description.py
    ```

- (Optional but recommended) We recommend generating cache files in `.h5` format of the BEV map to speed up the data loading process.
    ```
    # generate map cache for val
    python tools/prepare_map_aux.py +process=val +subfix=12Hz_interp

    # generate map cache for train
    python tools/prepare_map_aux.py +process=train +subfix=12Hz_interp
    ```
    After generating the cache files, move them to `./data/nuscenes_map_aux_12Hz_interp`

- Download the `nuscenes_interp_12Hz_infos_track2_eval.pkl` and `nuscenes_interp_12Hz_infos_track2_eval_long.pkl` from [w-coda2024/track2](https://coda-dataset.github.io/w-coda2024/track2/).

- The final data structure should look like this:
    ```
    ${ROOT}/data/
    ├── ...
    ├── nuscenes_mmdet3d-keyframes
    │       ├── nuscenes_infos_train.pkl
    │       └── nuscenes_infos_val.pkl
    ├── nuscenes_mmdet3d-12Hz
    |       ├── nuscenes_interp_12Hz_infos_train.pkl
    |       └── nuscenes_interp_12Hz_infos_val.pkl
    ├── nuscenes_mmdet3d-12Hz_description
    |       ├── nuscenes_interp_12Hz_updated_description_train.pkl
    |       └── nuscenes_interp_12Hz_updated_description_val.pkl
    ├── nuscenes_map_aux_12Hz_interp  # from interp
    |       ├── train_200x200_12Hz_interp.h5
    |       └── val_200x200_12Hz_interp.h5
    └── workshop
            ├── nuscenes_interp_12Hz_infos_track2_eval.pkl
            └── nuscenes_interp_12Hz_infos_track2_eval_long.pkl
    ```



## Pretrained Weights
We used the pre-trained weights of 
[stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) ([backup_link](https://huggingface.co/pt-sk/stable-diffusion-1.5)).

We assume you put them at `${ROOT}/pretrained/` as follows:

```
${ROOT}/pretrained/
        └── stable-diffusion-v1-5/
```
More information about the **Pre-trained weights**, please refer to [Model Zoo](../README.md).

You can organize them into this form:
```
${ROOT}/pretrained/
        ├── dreamforge-s
        ├── dreamforge-t
        └── stable-diffusion-v1-5/
```

## Training & Testing
### Train 
Stage-1: Train the single-frame version:
```bash
scripts/dist_train.sh 8 runner=8gpus +exp=dreamforge_s
```
Stage-2: Change the variant `pretrained_dreamforge` in [SDv1.5mv_rawbox_t.yaml](../configs/model/SDv1.5mv_rawbox_t.yaml) to the path of the weights for the single-frame version. Then, train the video autoregressive version:
```bash
scripts/dist_train.sh 8 runner=8gpus_t +exp=dreamforge_t
```
### Test

Test with the pre-trained weight.
1. Perform inference on the single-frame version:
```bash
python tools/test_single.py resume_from_checkpoint=./pretrained/dreamforge-s
```
2. Perform inference on the video autogressive version:
```bash
python tools/test.py --model_single ./pretrained/dreamforge-s --model ./pretrained/dreamforge-t --overlap_condition
```
3. Generate 16-frame videos or long videos (19s):
```bash
python tools/test_submit.py --input ./data/workshop/nuscenes_interp_12Hz_infos_track2_eval_long.pkl
```