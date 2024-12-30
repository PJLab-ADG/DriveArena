#############################
# nuPlan
#############################

# create ann_info
python tools/create_data.py nuplan --root-path /nas/shared/public/ADLab/datasets/Nuplan/ --version dreamer-trainval --out-dir data/nuplan --split-yaml tools/data_converter/nuplan.yaml

# obtain detailed scene description
python tools/description.py

# generate map cache for val
python tools/prepare_map_aux_nuplan.py +process=val +subfix=nuplan

# generate map cache for train
python tools/prepare_map_aux_nuplan.py +process=train +subfix=nuplan

# then move cache files to `../data/nuplan`