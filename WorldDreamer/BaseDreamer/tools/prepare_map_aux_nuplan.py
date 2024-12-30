import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import h5py
import torch
import numpy as np
from tqdm import tqdm

from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.loading_utils import one_hot_encode, one_hot_decode
from accelerate.utils import set_seed

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from dataset import *


KEYS2SAVE = {
    "gt_masks_bev": np.uint8,
}


def collate_as_it_is(sample):
    return sample


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, real_dataset):
        super().__init__()
        self.real_dataset = real_dataset

    def __len__(self):
        return self.real_dataset.__len__()

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.real_dataset.test_mode:
            return self.real_dataset.prepare_test_data(idx)
        # data can be None
        data = self.real_dataset.prepare_train_data(idx)
        if data is None:
            input_dict = self.real_dataset.get_data_info(idx)
            return input_dict['token']
        return data


def save_key_in_h5(h5: h5py.File, key_dtype, token, data):
    for key in key_dtype.keys():
        if key not in data:
            logging.info(f"There is no {key} in {token}")
            continue
        if key not in h5:
            grp = h5.create_group(key)
        else:
            grp = h5[key]
        if token in grp:
            continue
        if key_dtype[key] == np.uint8:
            encoded = one_hot_encode(data[key].astype(np.uint8))
            grp.create_dataset(token, data=encoded)
            decoded = one_hot_decode(encoded, data[key].shape[0])
            assert (data[key] == decoded).all()
        else:
            grp.create_dataset(token, data=data[key], dtype=key_dtype[key])


@hydra.main(version_base=None, config_path="../configs/dataset", config_name="Nuplan_map_cache_interp")
def main(cfg: DictConfig):
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    set_seed(cfg.seed)

    # add subfix
    if not hasattr(cfg, "subfix"):
        cfg.subfix = "_tmp"

    # amend cfg for data
    cfg.dataset.train_pipeline[-1]["keys"].append("gt_masks_bev")
    cfg.dataset.train_pipeline[-1].meta_lis_keys.append("token")
    cfg.dataset.train_pipeline[7].safe = False
    cfg.dataset.test_pipeline[-1]["keys"].append("gt_masks_bev")
    cfg.dataset.test_pipeline[-1].meta_lis_keys.append("token")
    cfg.dataset.test_pipeline[6].safe = False

    # make sure we have all items. HACK: force add these params, even not set
    with open_dict(cfg):
        cfg.dataset.data.train.filter_empty_gt = False
        cfg.dataset.data.val.filter_empty_gt = False
        cfg.dataset.data.test.filter_empty_gt = False

    if "process" not in cfg:
        print("Please specify data split to process: +process=train or val")
        return

    # datasets
    if cfg.process == "train":
        dataset = build_dataset(
            OmegaConf.to_container(cfg.dataset.data.train, resolve=True)
        )
    elif cfg.process == "val":
        dataset = build_dataset(
            OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
        )
    else:
        print("nothing to do")
        return

    # this wrapper ensure that we do not drop any None item.
    # HACK: we shuffle here to get more accurate time estimation.
    dataset = DatasetWrapper(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=4, num_workers=10,
        prefetch_factor=2, collate_fn=collate_as_it_is,
    )

    # NOTE: if with h5py <= 3.4, it will track timestamp, which make file hash
    # different. ref: https://github.com/h5py/h5py/pull/1958
    with h5py.File(f"{cfg.process}_{cfg.subfix}.h5", "w") as h5:
        for batch in tqdm(loader):
            for data in batch:
                if isinstance(data, str):
                    logging.warn(f'Error real token: {data}')
                    continue
                token = data["metas"].data["token"]
                token = token.replace("/", ";")
                save_key_in_h5(h5, KEYS2SAVE, token, data)


if __name__ == "__main__":
    main()
