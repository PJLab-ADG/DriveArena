# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Mapping, Tuple

import torch


@dataclass
class DeviceLimit:
    name: str = "default"  # pattern to match from `torch.cuda.get_device_name()`
    source: str = ""
    sm: Tuple[int, int] = (0, 0)
    # bytes/s
    gmem_bandwidth: float = math.inf
    # dtype -> TFlop/s
    gemm_tflops: Mapping[torch.dtype, float] = field(default_factory=dict)


# For f32, we assume we can use tf32
DEVICE_LIMITS: Tuple[DeviceLimit, ...] = (
    DeviceLimit(
        "A100",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",  # noqa: E501
        sm=(8, 0),
        gmem_bandwidth=1.5 * (1024**4),
        gemm_tflops={
            torch.float64: 19.5,
            torch.float32: 156,
            torch.float16: 312,
            torch.bfloat16: 312,
            torch.int8: 624,
        },
    ),
    DeviceLimit(
        "A30",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf",
        sm=(8, 0),
        gmem_bandwidth=933 * (1024**3),
        gemm_tflops={
            torch.float64: 10.3,
            torch.float32: 82,
            torch.float16: 165,
            torch.bfloat16: 165,
            torch.int8: 330,
        },
    ),
    DeviceLimit(
        "T4",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf",
        sm=(7, 5),
        gmem_bandwidth=300 * (1024**3),
        gemm_tflops={
            torch.float32: 8.1,
            torch.float16: 65,
            torch.int8: 130,
        },
    ),
    # Assuming SXM2
    DeviceLimit(
        "V100",
        "https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf",
        sm=(7, 0),
        gmem_bandwidth=900 * (1024**3),
        gemm_tflops={
            torch.float64: 7.8,
            torch.float32: 15.7,
            torch.float16: 125,
        },
    ),
    DeviceLimit(
        "P100",
        "https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-datasheet.pdf",
        sm=(6, 0),
        gmem_bandwidth=732 * (1024**3),
        gemm_tflops={
            torch.float64: 5.3,
            torch.float32: 10.6,
            torch.float16: 21.2,
        },
    ),
)


def get_device_limits(device) -> DeviceLimit:
    """Currently only implemented for GPUs"""
    if device is not None and device.type == "cuda":
        device_sm = torch.cuda.get_device_capability(device)
        device_name = torch.cuda.get_device_name(device)
        for lim in DEVICE_LIMITS:
            if lim.sm == device_sm:
                if lim.name in device_name:
                    return lim
    return DeviceLimit()
