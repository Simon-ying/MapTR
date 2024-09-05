// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
 
// CUDA function declarations
void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, float* out);
 
 
/*
  Function: pillar pooling (forward, cuda)
  Args:
    depth            : input depth, FloatTensor[n, d, h, w]
    feat             : input features, FloatTensor[n, h, w, c]
    out              : output features, FloatTensor[b, c, h_out, w_out]
    ranks_depth      : depth index of points, IntTensor[n_points]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
*/
void bev_pool_v2_forward(
  const at::Tensor _depth,  // (B, N, D, fH, fW)
  const at::Tensor _feat,   // (B, N, fH, fW, C)
  at::Tensor _out,  // (B, D_Z, D_Y, D_X, C)
  const at::Tensor _ranks_depth,    // (N_points, ),
  const at::Tensor _ranks_feat,     // (N_points, ),
  const at::Tensor _ranks_bev,      // (N_points, ),
  const at::Tensor _interval_lengths,   // (N_pillar, )
  const at::Tensor _interval_starts     // (N_pillar, )
) {
  int c = _feat.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_depth));
  const float* depth = _depth.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_depth = _ranks_depth.data_ptr<int>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();
 
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
 
  float* out = _out.data_ptr<float>();
  bev_pool_v2(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}
 