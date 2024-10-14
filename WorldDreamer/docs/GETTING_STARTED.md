# WorldDreamer - Getting Started

The following codes are all run in the `WorldDreamer` folder unless otherwise specified.

## Overview
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Weights](#pretrained-weights)
- [Training & Testing](#training--testing)

## Dataset Preparation 
(You can skip this step if you only want to use the simulation.)

Currently we provide the dataloader of [nuScenes dataset](#nuscenes-dataset) and [nuPlan dataset](#nuplan-dataset).

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
    └── nuscenes_map_aux_12Hz_interp  # from interp
            ├── train_200x200_12Hz_interp.h5
            └── val_200x200_12Hz_interp.h5
    ```
    > 🌻 You can download the `.pkl` files from [huggingface](https://huggingface.co/datasets/jokester-yxm/DriveArena_data).

### nuPlan Dataset

- To ensure a likely even distribution of the training data, we selected 64 logs from the NuPlan dataset. This selection includes 21 logs recorded in Las Vegas, 21 logs recorded in Pittsburgh, 11 logs recorded in Boston, and 11 logs recorded in Singapore. The names of the selected logs are listed under the `dreamer_train` and `dreamer_val` categories in [nuplan.yaml](../tools/data_converter/nuplan.yaml). Please download the official [nuPlan dataset](https://www.nuscenes.org/nuplan#download) and organized the files as follows:

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

- The nuplan-devkit need to be installed from source. 
```bash
cd third_party/nuplan-devkit
pip install -r requirements.txt
pip install -e .
```

- To prepare for training/validation, generate the `ann_file` by running the following command.
```bash 
python tools/create_data.py nuplan --root-path /path/to/nuplan/dataset/ --version dreamer-trainval --out-dir data/nuplan --split-yaml tools/data_converter/nuplan.yaml
```

- Refine the scene descriptions with the following command.

``` bash
python tools/description.py
```

- (Optional but recommended) We recommend generating cache files in `.h5` format of the bev map to speed up the data loading process.
    ``` bash
    # generate map cache for val
    python tools/prepare_map_aux_nuplan.py +process=val +subfix=nuplan_map_aux

    # generate map cache for train
    python tools/prepare_map_aux_nuplan.py +process=train +subfix=nuplan_map_aux
    ```
    After generating the cache files, move them to `./data/nuplan`


- The final data structure should look like this:
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
## Pretrained Weights
We used the pre-trained weights of 
[stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) ([backup_link](https://huggingface.co/pt-sk/stable-diffusion-1.5)) and
[CLIP-ViT](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K).

We assume you put them at `${ROOT}/pretrained/` as follows:

```
${ROOT}/pretrained/
        ├── stable-diffusion-v1-5/
        └── CLIP-ViT-B-32-laion2B-s34B-b79K/
```
**Pre-trained weights** of our WorldDreamer can be downloaded [here](https://huggingface.co/jokester-yxm/DriveArena/tree/main). More information about the ckeckpoints, please refer to [Model Zoo](../README.md/#model-zoo).

You can organize them into this form:
```
${ROOT}/dreamer_pretrained/
        ├── SDv1.5_mv_single_ref_nus
                ├── hydra
                └── weight-S200000
        └── other weights ...
```
## Training & Testing
### Train 

Train the single-frame autoregressive version:
```bash
scripts/dist_train.sh 8 runner=8gpus
```
### Test
Test with the pre-trained weight:
```bash
python tools/test.py resume_from_checkpoint=./dreamer_pretrained/SDv1.5_mv_single_ref_nus/weight-S200000
```
Test with your own weight:
```bash
python tools/test.py resume_from_checkpoint=path/to/your/weight
```
Test on the demo data, which is crop from the OpenStreetMap:
```bash
python tools/test.py runner.validation_index=demo resume_from_checkpoint=path/to/your/weight
```
## Todo
- [ ] check tensorboard code
- [x] check map visualization code