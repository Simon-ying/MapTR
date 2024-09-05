// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
 
#include <stdio.h>
#include <stdlib.h>
 
 
// CUDA内核函数，用于处理3D点云数据的特征聚合
__global__ void bev_pool_v2_kernel(int c, int n_intervals,
                                  const float *__restrict__ depth,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ out) {
  // 计算当前线程的全局索引，确定处理的数据位置
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 计算当前pillar的索引
  int index = idx / c;
  // 计算当前处理的特征通道的索引
  int cur_c = idx % c;
  // 如果当前pillar索引超出范围，则线程直接返回
  if (index >= n_intervals) return;
  // 获取当前pillar的起始点索引
  int interval_start = interval_starts[index];
  // 获取当前pillar包含的点的数量
  int interval_length = interval_lengths[index];
  // 初始化累加器，用于累计特征值
  float psum = 0;
  // 指向当前深度值和特征值的指针
  const float* cur_depth;
  const float* cur_feat;
  // 遍历当前pillar的所有点
  for(int i = 0; i < interval_length; i++) {
    // 获取当前点的深度值
    cur_depth = depth + ranks_depth[interval_start + i];
    // 获取当前通道对应的特征值
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    // 累加当前点的特征值与深度值的乘积
    psum += *cur_feat * *cur_depth;
  }
  // 获取当前pillar在BEV（鸟瞰图）网格中的索引
  const int* cur_rank = ranks_bev + interval_start;
  // 定位输出数组中的相应位置
  float* cur_out = out + *cur_rank * c + cur_c;
  // 将累加的特征值写入输出数组
  *cur_out = psum;
}
 
 
// 定义 bev_pool_v2 函数，用于并行处理3D点云数据
void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat, 
                 const int* ranks_depth, const int* ranks_feat, const int* ranks_bev, 
                 const int* interval_starts, const int* interval_lengths, float* out) {
  // 调用CUDA内核函数bev_pool_v2_kernel
  // 使用n_intervals * c计算所需的线程总数，然后除以256确定需要多少个线程块
  // 每个线程块使用256个线程
  bev_pool_v2_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}
 