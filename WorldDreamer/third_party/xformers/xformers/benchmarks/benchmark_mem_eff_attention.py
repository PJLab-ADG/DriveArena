# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import random
from functools import partial

import torch
from torch.utils import benchmark
from utils import benchmark_main_helper

import xformers.ops
import xformers.ops.fmha as fmha

torch.backends.cuda.matmul.allow_tf32 = False


def create_attn_bias(
    bias_type,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    device,
    dtype,
    bias_requires_grad: bool = False,
):
    NoneType = type(None)
    if bias_type is NoneType:
        return None
    if bias_type is torch.Tensor:
        attn_bias = (
            torch.randn((batch_size * num_heads, 1, kv_len), device=device, dtype=dtype)
            * 3
        )
        return attn_bias.expand(batch_size * num_heads, q_len, kv_len)
    if bias_type is xformers.ops.LowerTriangularMask:
        return bias_type()
    assert False, f"Unsupported bias type: {bias_type}"


def ref_attention_bmk(q, k, v, attn_bias=None, p=0.0):
    if isinstance(attn_bias, xformers.ops.AttentionMask):
        attn_bias = (
            attn_bias.materialize((q.shape[0], 1, q.shape[1], k.shape[1]))
            .to(q)
            .squeeze()
        )
    q = q * (1.0 / q.shape[-1] ** 0.5)
    if attn_bias is None:
        attn = q @ k.transpose(-2, -1)
    else:
        # equivalent to (q @ k.transpose(-2, -1) + m).softmax(-1) @ v
        # but faster, and is what is used in PyTorch now
        attn = torch.baddbmm(attn_bias, q, k.transpose(-2, -1))
    attn = attn.softmax(-1)
    if p > 0:
        attn = torch.nn.functional.dropout(attn, p=p)
    return attn @ v


def ref_attention(q, k, v, attn_bias, p=0.0):
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    out = ref_attention_bmk(T(q), T(k), T(v), attn_bias, p)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


min_run_time = 0.5
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = [
    # ViT
    (384, 197, 1, 88),
    (384, 197, 1, 80),
    (384, 197, 1, 64),
    (1024, 197, 1, 88),
    (1024, 197, 1, 80),
    (1024, 197, 1, 64),
    # ViT-Huge
    (32 * 16, 197, 1, 80),
    (32, 197, 16, 80),
    (32, 197, 16, 64),
    (32, 197, 16, 128),
    # ViT-Giant
    (16 * 16, 197, 1, 88),
    (16, 197, 16, 88),
    (16, 197, 16, 64),
    (16, 197, 16, 128),
    # FB models
    (1024, 82, 8, 64),
    (150, 256, 16, 64),
    (64, 256, 12, 64),
    # Stable diffusion (https://github.com/huggingface/diffusers/pull/532)
    (1, 4096, 16, 40),  # 512x512
    (1, 16384, 16, 40),  # 1024x1024
    (1, 4096, 16, 80),
    (1, 16384, 16, 80),
    # + bs4
    (4, 4096, 16, 40),
    (4, 16384, 16, 40),
    (4, 4096, 16, 80),
    (4, 16384, 16, 80),
    # ParlAI model
    (256, 4096, 16, 64),
    # Zetta B M H K
    (8, 2048, 20, 128),
    # LLaMa 70b - mp=8/16
    *sorted(list(itertools.product([1, 2], [2048, 4096, 8192], [4, 8], [128]))),
    *sorted(list(itertools.product([16], [128, 512, 1024], [16], [16, 32, 64, 128]))),
]

OPS = [
    (xformers.ops.fmha.cutlass.FwOp, xformers.ops.fmha.cutlass.BwOp),
    (xformers.ops.fmha.flash.FwOp, xformers.ops.fmha.flash.BwOp),
    # TODO: Triton is not stable: it can trigger Illegal Memory Accesses
    # and its performance varies a lot between runs.
    # (xformers.ops.fmha.triton.FwOp, xformers.ops.fmha.triton.BwOp),
]


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        shape=SHAPES,
        num_threads=NUM_THREADS,
        dropout_p=[0.0],
        attn_bias_cfg=[(type(None), False)],
        dtype=[torch.half],
    )
)

# Add more cases with some variations
for c in CASES.copy():
    c = c.copy()
    c.update(
        random.Random(str(c["shape"])).choice(
            [
                {"dropout_p": 0.3},
                {"attn_bias_cfg": (torch.Tensor, False)},
                {"attn_bias_cfg": (torch.Tensor, True)},
                {"attn_bias_cfg": (xformers.ops.LowerTriangularMask, False)},
                {"dtype": torch.bfloat16},
                {"dtype": torch.float},
            ]
        )
    )
    CASES.append(c)


def create_tensors(shape, dtype, requires_grad=False):
    B, M, H, K = shape
    qkv = torch.rand(
        [B, M, 3, H, K], device=device, dtype=dtype, requires_grad=requires_grad
    )
    q, k, v = xformers.ops.unbind(qkv, 2)
    return qkv, q, k, v


def mem_eff_attention_fw(shape, num_threads: int, attn_bias_cfg, dropout_p, dtype):
    B, M, H, K = shape
    _, q, k, v = create_tensors(shape, dtype)
    attn_bias_type, attn_bias_requires_grad = attn_bias_cfg
    if attn_bias_requires_grad:
        return
    bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        num_heads=H,
        q_len=M,
        kv_len=M,
        device=device,
        dtype=dtype,
        bias_requires_grad=attn_bias_requires_grad,
    )
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias, p=dropout_p)

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = (
        f"{dtype_str} {B}-{M}-{H}-{K}, p={dropout_p}, "
        f"BiasT={attn_bias_type.__name__}"
    )

    has_run = False
    for fw_op, bw_op in OPS:
        if not fw_op.supports(inp):
            continue

        yield benchmark.Timer(
            stmt="fn(q, k, v, attn_bias, p)",
            globals={
                "q": q,
                "k": k,
                "v": v,
                "attn_bias": inp.attn_bias,
                "p": dropout_p,
                "fn": partial(
                    xformers.ops.memory_efficient_attention, op=(fw_op, bw_op)
                ),
            },
            label=f"attention (attn_bias={attn_bias_type})",
            description=fw_op.NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )
        has_run = True

    if not has_run:
        return

    yield benchmark.Timer(
        stmt="fn(q, k, v, attn_bias, p)",
        globals={
            "q": q,
            "k": k,
            "v": v,
            "attn_bias": inp.attn_bias,
            "p": dropout_p,
            "fn": ref_attention,
        },
        label=f"attention (attn_bias={attn_bias_type})",
        description="eager",
        sub_label=sub_label,
        num_threads=num_threads,
    )


def mem_eff_attention_bw(shape, num_threads: int, attn_bias_cfg, dropout_p, dtype):
    B, M, H, K = shape
    qkv, q, k, v = create_tensors(shape, dtype, requires_grad=True)

    attn_bias_type, attn_bias_requires_grad = attn_bias_cfg
    bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        num_heads=H,
        q_len=M,
        kv_len=M,
        device=device,
        dtype=dtype,
        bias_requires_grad=attn_bias_requires_grad,
    )
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias, p=dropout_p)

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = (
        f"{dtype_str} {B}-{M}-{H}-{K}, p={dropout_p}, "
        f"BiasT={attn_bias_type.__name__}, BiasGrad={attn_bias_requires_grad}"
    )

    has_run = False
    for fw_op, bw_op in OPS:
        if not fw_op.supports(inp) or not bw_op.supports(inp):
            continue
        has_run = True
        out = xformers.ops.memory_efficient_attention(
            inp.query, inp.key, inp.value, inp.attn_bias, inp.p, op=(fw_op, bw_op)
        )
        grad_benchmark = torch.ones_like(q)

        yield benchmark.Timer(
            stmt="out.backward(grad, retain_graph=True)",
            globals={
                "out": out,
                "grad": grad_benchmark,
            },
            label=f"attention backward (attn_bias={attn_bias_type})",
            description=bw_op.NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )
        del out

    if not has_run:
        return
    yield benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={
            "out": ref_attention(q, k, v, inp.attn_bias, dropout_p),
            "grad": grad_benchmark,
        },
        label=f"attention backward (attn_bias={attn_bias_type})",
        description="vanilla",
        sub_label=sub_label,
        num_threads=num_threads,
    )


benchmark_main_helper(mem_eff_attention_fw, CASES, min_run_time=min_run_time)
benchmark_main_helper(mem_eff_attention_bw, CASES, min_run_time=min_run_time)
