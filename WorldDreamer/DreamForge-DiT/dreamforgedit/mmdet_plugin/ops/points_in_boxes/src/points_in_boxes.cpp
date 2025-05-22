// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                        at::Tensor pts_indices_tensor);

int points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                        at::Tensor box_idx_of_points_tensor);

int points_in_boxes_batch(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                          at::Tensor box_idx_of_points_tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("points_in_boxes_gpu", &points_in_boxes_gpu,
        "points_in_boxes_gpu forward (CUDA)");
  m.def("points_in_boxes_batch", &points_in_boxes_batch,
        "points_in_boxes_batch forward (CUDA)");
  m.def("points_in_boxes_cpu", &points_in_boxes_cpu,
        "points_in_boxes_cpu forward (CPU)");
}
