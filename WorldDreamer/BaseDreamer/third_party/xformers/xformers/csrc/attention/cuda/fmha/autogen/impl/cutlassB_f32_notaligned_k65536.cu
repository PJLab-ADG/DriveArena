/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "../../kernel_backward.h"
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f32_notaligned_64x64_k65536_sm50` is for sm50-sm70, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f32_notaligned_64x64_k65536_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f32_notaligned_64x64_k65536_sm75` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
