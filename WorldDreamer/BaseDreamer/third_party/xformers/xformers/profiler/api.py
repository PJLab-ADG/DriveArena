# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Sequence, Tuple

import torch.nn as nn

from .profiler import MemSnapshotsProfiler, NsightProfiler, PyTorchProfiler, _Profiler
from .slow_ops_profiler import DetectSlowOpsProfiler

DEFAULT_SCHEDULE = (
    (MemSnapshotsProfiler, 0, 2),
    (DetectSlowOpsProfiler, 2, 4),
    (NsightProfiler, 4, 6),
    (PyTorchProfiler, 6, 20),
)


def profile(
    output_dir: str,
    module: Optional[nn.Module] = None,
    schedule: Sequence[Tuple[Any, int, int]] = DEFAULT_SCHEDULE,
):
    """
    A pre-configured profiler that will run on the first ~20 steps of the training
    It will provide multiple traces that can be exploited later.
    Use it in a context manager around your training loop, and call `xformers.profiler.step`
    before starting the next iteration.

    :Examples:

    .. code-block:: python

        import torch
        import timm.models
        import xformers.profiler

        dtype = torch.bfloat16
        device = "cuda"
        model = timm.models.vit_large_patch16_224().to(device).to(dtype)
        inp = torch.zeros([64, 3, 224, 224], device=device, dtype=dtype)
        optim = torch.optim.Adam(model.parameters())

        with xformers.profiler.profile(
            output_dir="profile_data",
            module=model,
            schedule=[
                (MemSnapshotsProfiler, 0, 2),
                (DetectSlowOpsProfiler, 2, 4),
                (NsightProfiler, 4, 6),
                (PyTorchProfiler, 6, 20),
            ]
        ):
            for i in range(20):
                model(inp).sum().backward()
                optim.step()
                optim.zero_grad()
                xformers.profiler.step()

        # alternatively, use the profiler without context and with ``.start()`` / `.stop()`
        # calls.

        xprofiler = xformers.profiler.profile(...)
        xprofiler.start()

        for i in range(20):
            model(inp).sum().backward()
            optim.step()
            optim.zero_grad()
            xprofiler.step()

        xprofiler.stop()
    """
    return _Profiler(output_dir=output_dir, schedule=schedule, module=module)


def step() -> None:
    """See `xformers.profiler.profile`"""
    # Silently return if no profiler is enabled
    if _Profiler._CURRENT_PROFILER is None:
        return
    _Profiler._CURRENT_PROFILER.step()
