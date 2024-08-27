# Getting Started

## Dataset Preparation

Currently we provide the dataloader of nuScenes dataset.

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
```
pip install nuscenes-devkit==1.1.11
```
- Generate the `ann_file` **(with keyframes / samples)** by running the following command, it may take several hours:
```
python -m tools.create_data nuscenes \
--root-path /path/to/nuscenes --out-dir ./data/nuscenes_mmdet3d-t-keyframes/ \
--extra-tag nuscenes --only_info
```
- Generate the `ann_file` **(with 12hz / sweeps)** by running the following command, it may take longer time. We use them to train the model.
    
    - Firstly, follow [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md) to generate interp annotations for nuScenes. 

        **Note**: The following codes need to be modified:
        
        - In `sAP3D/nusc_annotation_generator.py`, please comment [line357](https://github.com/JeffWang987/ASAP/blob/52316629f2a87ef2ef5bbc634d33e9544b5e39a7/sAP3D/nusc_annotation_generator.py#L357), and modify [line101](https://github.com/JeffWang987/ASAP/blob/52316629f2a87ef2ef5bbc634d33e9544b5e39a7/sAP3D/nusc_annotation_generator.py#L101) to `val_scene_ids = splits['val'] + splits['train']`.
        
        - Modify the dataset path in `scripts/ann_generator.sh` to your custom dataset path.
    
        Then, you can run the following command in ASAP root:
        ```
        bash scripts/ann_generator.sh 12 --ann_strategy 'interp' 
        ```
        (Optional) Generate advanced annotations for sweeps. (We do not observe major difference between interp and advanced. You can refer to the implementation of [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md). This step can be skipped.)

        Rename the generated folder to `interp_12Hz_trainval` and move it into your nuScenes dataset root.
        
    - Use the following command to generate ann_file.
        ```
        python tools/create_data.py nuscenes \
        --root-path /path/to/nuscenes \
        --out-dir ./data/nuscenes_mmdet3d-12Hz \
        --extra-tag nuscenes_interp_12Hz \
        --max-sweeps -1 \
        --version interp_12Hz_trainval
        ```
- (Optional but recommended) We recommend generating cache files in h5 format of the bev map to speed up the data loading process.
    ```
    # generate map cache for val
    python tools/prepare_map_aux.py +process=val +subfix=12Hz_interp

    # generate map cache for train
    python tools/prepare_map_aux.py +process=train +subfix=12Hz_interp
    ```
    After generating the cache files, move them to `./data/nuscenes_map_aux_12Hz_interp`
- Modify descriptions.

- The final data structure should look like this:
    ```
    ${ROOT}/data/
    ├── ...
    ├── nuscenes_mmdet3d-keyframes
    │	    ├── nuscenes_infos_train.pkl
    │	    └── nuscenes_infos_val.pkl
    ├── nuscenes_mmdet3d-12Hz
    |       ├── nuscenes_interp_12Hz_infos_train.pkl
    |       └── nuscenes_interp_12Hz_infos_val.pkl
    └── nuscenes_map_aux_12Hz_interp  # from interp
            ├── train_200x200_12Hz_interp.h5
            └── val_200x200_12Hz_interp.h5
    ```


### nuPlan Dataset



## Training & Testing
### Pretrained Weights
We used the pre-trained weights of 
[stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and
[CLIP-ViT](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K).

We assume you put them at ${ROOT}/pretrained/ as follows:

```
${ROOT}/pretrained/
        ├── stable-diffusion-v1-5/
        └── CLIP-ViT-B-32-laion2B-s34B-b79K/
            
```
**Pre-trained weights** of our dreamer can be downloaded [here]().
```
${ROOT}/dreamer-pretrained/
        └── SDv1.5_mv_single_ref_nus
                ├── hydra
                └── weight_S200000
            
```
### Train 

Train the single-frame autoregressive version:
```
scripts/dist_train.sh 8 runner=8gpuss
```
### Test
Test with the pre-trained weight:
```
python tools/test.py resume_from_checkpoint=./dreamer-pretrained/SDv1.5_mv_single_ref_nus/weight_S200000
```
Test with your own weight:
```
python tools/test.py resume_from_checkpoint=./dreamer-log/path/to/your/weight
```

## Todo
- [ ] check tensorboard code
- [ ] check map visualization code
