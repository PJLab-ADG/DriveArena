import random
from typing import Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader, Subset

from .sampler import StatefulDistributedSampler
from .nuscenes_map_dataset import NuScenesMapDataset, collate_fn
from .nuscenes_map_dataset_t import NuScenesMapDatasetT, collate_fn_t
from functools import partial


# Deterministic dataloader
def get_seed_worker(seed):
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def prepare_dataloader(
    dataset,
    batch_size=None,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[ProcessGroup] = None,
    bucket_config=None,
    num_bucket_build_workers=1,
    prefetch_factor=None,
    **kwargs,
):
    _kwargs = kwargs.copy()
    if isinstance(dataset, (NuScenesMapDataset, NuScenesMapDatasetT, Subset)):
        process_group = process_group or _get_default_group()
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            # seed=seed,
        )
        collate_praram = dataset.dataset.img_collate_param if isinstance(dataset, Subset) else dataset.img_collate_param
        collate_func = collate_fn
        if isinstance(dataset, Subset):
            if isinstance(dataset.dataset, NuScenesMapDatasetT):
                collate_func = collate_fn_t
        else:
            if isinstance(dataset, NuScenesMapDatasetT):
                collate_func = collate_fn_t
        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                drop_last=drop_last,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=partial(collate_func, **collate_praram),
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            sampler,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

