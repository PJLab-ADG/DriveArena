# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## TBD
### Fixed
### Added

## [0.0.18] - 2023-03-31
### Added
- Added `xformers.ops.index_select_cat` and `xformers.ops.scaled_index_add` - those are experimental functions that only work with a few shapes, and can be used to write efficient stochastic depth in transformer architectures for instance

### Fixed
- fMHA: `memory_efficient_attention` now accepts `torch.Tensor` as attention bias for any seqlen, although there are still requirements on the alignment of the bias tensor (see facebookresearch/xformers#683)

## [0.0.17] - 2023-03-28
### Fixed
- fMHA: Fixed BW pass on Sm86/Sm89 GPUs when `K > 64` (RTX 3090, RTX 4090, A6000, ..) [facebookresearch/xformers#631]

### Added
- fMHA/CUTLASS: Added tensor attn bias support [facebookresearch/xformers#587] - contribution from [@jfc4050](https://github.com/jfc4050)
- fMHA/CUTLASS: Added tensor attn bias grad support [facebookresearch/xformers#587] - contribution from [@jfc4050](https://github.com/jfc4050)
- fMHA/CUTLASS: Added dropout support [facebookresearch/xformers#587] - contribution from [@jfc4050](https://github.com/jfc4050)
- fMHA: Added support for varying sequence lengths [facebookresearch/xformers#500]


## [0.0.16] - 2023-01-31
### Fixed
- Updated triton dependency [facebookresearch/xformers#418]
- Stripe lineinfo from binaries, reducing the binary size [facebookresearch/xformers#549]
- Added support for pip wheels [facebookresearch/xformers#588, facebookresearch/xformers#573, facebookresearch/xformers#534, facebookresearch/xformers#523, ...] big thanks to [@AbdBarho](https://github.com/AbdBarho)!
- Fixed compatibility with Python 3.7 [facebookresearch/xformers#541] - thanks to [@susumuota](https://github.com/susumuota)
- fMHA: Fixed strides for QKV gradients for cutlass attention [facebookresearch/xformers#535]
- fMHA: Stricter inputs validation to avoid CUDA errors for unsupported inputs [facebookresearch/xformers#592]
- fMHA/Flash-Attention: Updated to https://github.com/HazyResearch/flash-attention/commit/a1f49a2b92b6fa022379bbebafed9d7f5e96a675 with multiple changes from [@TriDao](https://github.com/tridao) that make the operator up to 20% faster
- fMHA/Flash-Attention: Fixed backward pass wrapper, where non-contiguous gradients could give the wrong result [facebookresearch/xformers#548]
- fMHA: Separate each operator into forward and backward operators. It's now possible to use any combination of forward+backward (for instance Triton forward and Flash-Attention backward) [facebookresearch/xformers#560]

### Added
- fMHA: Added Triton operator for forward pass from [Flash-Attention](https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py) authored by [@TriDao](https://github.com/tridao), will be automatically used on A100 when compatible
- fMHA: Added [`xformers.ops.memory_efficient_attention_forward`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention_forward), [`xformers.ops.memory_efficient_attention_forward_requires_grad`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention_forward_requires_grad), [`xformers.ops.memory_efficient_attention_backward`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention_backward) for power-users who write custom autograd functions [facebookresearch/xformers#560]
- fMHA: Support for custom scaling for the CUTLASS-based kernel [facebookresearch/xformers#530] - contribution from [@comaniac](https://github.com/comaniac)

## [0.0.15] - Skipped

## [0.0.14] - 2022-11-10
### Fixed
- fMHA/CUTLASS: The current CUDA stream is now used by the kernel [facebookresearch/xformers#491]
- fMHA/CUTLASS: Improve overall performance

### Added
- SwiGLU: Added `xformers.ops.SwiGLU` and its functional counterpart (`xformers.ops.swiglu`) [facebookresearch/xformers#490]
- fMHA: Possible to combine CUTLASS's forward with flash-attention's backward pass [facebookresearch/xformers#469] - improves performance on A100 for K = 128
- fMHA: Add custom `xformers.ops.unbind` operator to avoid a cat in the attention block [facebookresearch/xformers#458]

## [0.0.13] - 2022-09-26
### Added
- fMHA: Added CUTLASS-based kernel for `xformers.ops.memory_efficient_attention`. This kernel is automatically depending on the inputs, and works on any GPU after P100 [facebookresearch/xformers#362]

## [0.0.12] - 2022-08-08
### Fixed
- Removed duplicated biases in the FusedMLP layers [facebookresearch/xformers#317]
- Rotary embeddings respecting input types [facebookresearch/xformers#326]
- Poolformer style instantiating useless projection layers [facebookresearch/xformers#349]
- Fix layer position not being properly tracked, causing extra layernorms for programmatic xformers [facebookresearch/xformers#348]
- Pass use_triton flag to LayerNorm module [facebookresearch/xformers#336]

### Added
- Four blocksparsity layouts from DeepSpeed [facebookresearch/xformers#320]
- Support several initialization options [facebookresearch/xformers#312]
- Conv2DFeedforward feedforward part [facebookresearch/xformers#321]
- VisualAttention [facebookresearch/xformers#329]
- Automatic blocksparse for causal attention [facebookresearch/xformers#334]
- Better hierarchical transformer generation [facebookresearch/xformers#345]
- Fused operations with AOTAutograd/NVFuser, integration into MLP [facebookresearch/xformers#357]
- Refactor LRA code to use Pytorch Lightning [facebookresearch/xformers#343]

## [0.0.11] - 2022-05-30
### Fixed
- Fix some torchscriptability [facebookresearch/xformers#246]
- Fix FourierMix being compatible with AMP [facebookresearch/xformers#258]
- Better asserts on QKV dimensions [facebookresearch/xformers#264]
- Better perfs for FusedMLP and FusedLinearLayer [facebookresearch/xformers#283]
- Deepnorm init missing self-attention [facebookresearch/xformers#284]

### Added
- Simplicial Embeddings [facebookresearch/xformers#259]
- Mem efficient attention, FW pass [facebookresearch/xformers#267]
- MHA benchmark
- MLP benchmark
- Move all triton kernels to triton v2 [facebookresearch/xformers#272]
- Mem efficient attention, BW pass [facebookresearch/xformers#281]
- Metaformer support [facebookresearch/xformers#294]

## [0.0.10] - 2022-03-14
### Fixed
- Expose bias flag for feedforwards, same default as Timm [facebookresearch/xformers#220]
- Update eps value for layernorm, same default as torch [facebookresearch/xformers#221]
- PreNorm bugfix, only one input was normalized [facebookresearch/xformers#233]
- Fix bug where embedding dimensions that did not match model dim would lead to a crash [facebookresearch/xformers#244]

### Added
- Add DeepNet (DeepNorm) residual path and init [facebookresearch/xformers#227]

## [0.0.9] - 2022-02-09
### Added
- Compositional Attention [facebookresearch/xformers#41]
- Experimental Ragged attention [facebookresearch/xformers#189]
- Mixture of Experts [facebookresearch/xformers#181]
- BlockSparseTensor [facebookresearch/xformers#202]
- Nd-tensor support for triton softmax [facebookresearch/xformers#210]

### Fixed
- Bugfix Favor, single feature map [facebookresearch/xformers#183]
- Sanity check blocksparse settings [facebookresearch/xformers#207]
- Fixed some picklability [facebookresearch/xformers#204]

## [0.0.8] - 2022-01-07
### Fixed
- Much faster fused dropout [facebookresearch/xformers#164]
- Fused dropout repeatability [facebookresearch/xformers#173]

### Added
- Embedding weight tying option [facebookresearch/xformers#172]

## [0.0.7] - 2021-11-30
### Fixed
- Dropout setting not properly passed in many attentions [facebookresearch/xformers#123]

## [0.0.6] - 2021-11-24
### Fixed
- Fix self attention optimization not being triggered, broken residual path [facebookresearch/xformers#119]
- Improve speed by not using contiguous Tensors when not needed [facebookresearch/xformers#119]

### Added
- Attention mask wrapper [facebookresearch/xformers#113]
- ViT comparison benchmark [facebookresearch/xformers#117]

## [0.0.4] - 2021-11-16
### Fixed
- Homogenizing the masks, additive or bool [facebookresearch/xformers#79][facebookresearch/xformers#85][facebookresearch/xformers#86]
- Fix causality flag not being respected [facebookresearch/xformers#103]
- Enabling FusedLayerNorm by default in the factory if Triton is available
- Fixing Favor with fp16
- Fixing Favor trainability

### Added
- Fused dropout/bias/activation layer [facebookresearch/xformers#58]
- Fused layernorm used by default in the factory [facebookresearch/xformers#92]


## [0.0.3] - 2021-11-01
### Fixed
- Nystrom causal attention [facebookresearch/xformers#75]


## [0.0.2] - 2021-11-01
### Fixed
- More robust blocksparse [facebookresearch/xformers#24]

### Added
- Rotary embeddings [facebookresearch/xformers#32]
- More flexible layernorm [facebookresearch/xformers#50]
