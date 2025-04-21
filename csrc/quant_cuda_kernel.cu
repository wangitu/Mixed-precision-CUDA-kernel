#include <cstdio>
#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>


// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif


constexpr int BLOCKWIDTH = 128;
constexpr int BLOCKHEIGHT3 = 48;
constexpr int BLOCKHEIGHT4 = 64;


__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}
  
__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}


__global__ void VecQuant3MatMulKernelCherryQBatchedG64(
  const float* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const float* __restrict__ cherry_weight,
  const int* __restrict__   cherry_indices,
  const float* __restrict__ scaling_factors,
  float* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size
);

__global__ void VecQuant3MatMulKernelCherryQBatched(
  const float* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const float* __restrict__ cherry_weight,
  const int* __restrict__   cherry_indices,
  const float* __restrict__ scaling_factors,
  float* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size, int group_size
);

__global__ void VecQuant3MatMulKernelCherryQBatchedG64Half(
  const half* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const half* __restrict__ cherry_weight,
  const int16_t* __restrict__   cherry_indices,
  const half* __restrict__ scaling_factors,
  half* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size
);

__global__ void VecQuant3MatMulKernelCherryQBatchedHalf(
  const half* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const half* __restrict__ cherry_weight,
  const int16_t* __restrict__   cherry_indices,
  const half* __restrict__ scaling_factors,
  half* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size, int group_size
);

__global__ void VecQuant4MatMulKernelCherryQBatchedHalf(
  const half* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const half* __restrict__ cherry_weight,
  const int16_t* __restrict__   cherry_indices,
  const half* __restrict__ scaling_factors,
  half* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size, int group_size
);

__global__ void Dequantize3NormalKernel(
  const uint8_t* __restrict__   qweight,
  const float* __restrict__ scaling_factors,
  float* __restrict__       out,
  int height, int width, int cherry_size, int group_size
);

__global__ void Dequantize4NormalKernel(
  const uint8_t* __restrict__   qweight,
  const float* __restrict__ scaling_factors,
  float* __restrict__       out,
  int height, int width, int cherry_size, int group_size
);


void vecquant3matmul_cherryq_cuda(
  torch::Tensor vec,
  torch::Tensor qweight,
  torch::Tensor cherry_weight,
  torch::Tensor cherry_indices,
  torch::Tensor scaling_factors,
  torch::Tensor mul,
  int group_size, bool fp16
) {
  int height = qweight.size(0);
  int width = qweight.size(1);
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int cherry_size = cherry_indices.size(0);

  // 指派额外的线程块进行 cherry_weight 的浮点数乘法
  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3 + (cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  if (fp16 && group_size == 64) {
    VecQuant3MatMulKernelCherryQBatchedG64Half<<<blocks, threads>>>(
      reinterpret_cast<half*>(vec.data_ptr<at::Half>()),
      qweight.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(cherry_weight.data_ptr<at::Half>()),
      cherry_indices.data_ptr<int16_t>(),
      reinterpret_cast<half*>(scaling_factors.data_ptr<at::Half>()),
      reinterpret_cast<half*>(mul.data_ptr<at::Half>()),
      width, batch, vec_height, cherry_size
    );
  } else if (fp16 && group_size % 128 == 0) {
    VecQuant3MatMulKernelCherryQBatchedHalf<<<blocks, threads>>>(
      reinterpret_cast<half*>(vec.data_ptr<at::Half>()),
      qweight.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(cherry_weight.data_ptr<at::Half>()),
      cherry_indices.data_ptr<int16_t>(),
      reinterpret_cast<half*>(scaling_factors.data_ptr<at::Half>()),
      reinterpret_cast<half*>(mul.data_ptr<at::Half>()),
      width, batch, vec_height, cherry_size, group_size
    );
  } else if (!fp16 && group_size == 64){
    VecQuant3MatMulKernelCherryQBatchedG64<<<blocks, threads>>>(
      vec.data_ptr<float>(),
      qweight.data_ptr<uint8_t>(),
      cherry_weight.data_ptr<float>(),
      cherry_indices.data_ptr<int>(),
      scaling_factors.data_ptr<float>(),
      mul.data_ptr<float>(),
      width, batch, vec_height, cherry_size
    );
  } else if (!fp16 && group_size % 128 == 0) {
    VecQuant3MatMulKernelCherryQBatched<<<blocks, threads>>>(
      vec.data_ptr<float>(),
      qweight.data_ptr<uint8_t>(),
      cherry_weight.data_ptr<float>(),
      cherry_indices.data_ptr<int>(),
      scaling_factors.data_ptr<float>(),
      mul.data_ptr<float>(),
      width, batch, vec_height, cherry_size, group_size
    );
  }

  // cudaDeviceSynchronize();
  // // 检查同步是否出错
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //     printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  //     exit(1);
  // }
}

// 3bit vec matmul (group_size = 64)
__global__ void VecQuant3MatMulKernelCherryQBatchedG64(
  const float* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const float* __restrict__ cherry_weight,
  const int* __restrict__   cherry_indices,
  const float* __restrict__ scaling_factors,
  float* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size // group_size = 64
) {
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  if (col >= width) return;

  // 处理 vec 中与当前线程块对应的 128 个 float 激活值
  __shared__ float blockvec[BLOCKWIDTH];
  // 当前线程块的每个线程处理一列参数 (128 个 int3)，由于 group_size = 64，故每列参数对应两个 scaling_factor
  float factor_up, factor_down;

  // 每个线程束中只能含有同一线程块中的线程，因此以下分支指令不会发生线程束分化
  if (blockIdx.x >= gridDim.x - (cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH) {
    // cherry 参数乘法
    float res;
    // k: 当前线程处理的 cherry 参数在 cherry_indices 的第 k 列
    int row = ((cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH - (gridDim.x - blockIdx.x));
    int k = row * BLOCKWIDTH + threadIdx.x;
    
    #pragma unroll 16 // 循环展开，减少分支指令开销
    for (int b = 0; b < batch; b++) {
      res = 0;
      blockvec[threadIdx.x] = (k < cherry_size) ? vec[b * vec_height + cherry_indices[k]] : 0;
      __syncthreads();
      
      // i: 当前线程处理的 cherry 参数在 cherry_weight 的第 i 行
      int i = row * BLOCKWIDTH;
      int j = 0;
      while (i < cherry_size && j < BLOCKWIDTH) {
        res += cherry_weight[i * width + col] * blockvec[j];
        i++; j++;
      }

      atomicAdd(&mul[b * width + col], res);
    }
    return;
  }

  // 处理从第 BLOCKHEIGHT3 * blockIdx.x 行开始向下的 48 行 int8 参数 (= 128 个 int3 参数)
  int row = BLOCKHEIGHT3 * blockIdx.x;
  int off = BLOCKWIDTH * blockIdx.x;

  // 1. 加载 scaling_factors
  // blockIdx.x * (BLOCKWIDTH / group_size) = blockIdx.x * 2: 当前处理的一列参数从第几个 group 开始
  // (vec_height - cherry_size + group_size - 1) / group_size = (vec_height - cherry_size + 63) / 64): 除去 cherry 参数后一共能形成几个 group
  {
    int group_idx = blockIdx.x * 2;
    factor_up = (group_idx < (vec_height - cherry_size + 63) / 64) ? scaling_factors[group_idx * width + col] : 0;
    factor_down = (group_idx + 1 < (vec_height - cherry_size + 63) / 64) ? scaling_factors[(group_idx + 1) * width + col] : 0;
  }

  // i: 当前处理的参数在 qweight 中的索引；j: 与当前处理的参数相乘的激活值在 blockvec 中的索引
  int i, j;
  float res, r;
  uint8_t q1, q2, tmp;

  // 2. 加载当前需要相乘的激活值
  // (a) 二分查找 k 在 cherry_indices 中的 upper_bound
  // (b) 如果 k 就在 cherry_indices 中，则将其移动至最后并设为0，否则将其向前移动
  // (c) 注意线程束分化的问题
  int k = blockIdx.x * BLOCKWIDTH + threadIdx.x;
  int low = 0;
  {
    int high = cherry_size;
    while (low < high) {
      int mid = (low + high) >> 1;
      int cmp = (cherry_indices[mid] - mid <= k);
      low = cmp ? mid + 1 : low;
      high = cmp ? high : mid;
    }
  }
  
  #pragma unroll 16 // 循环展开，减少分支指令开销
  for (int b = 0; b < batch; b++) {
    i = width * row + col;
    res = r = 0;

    {
      int idx = b * vec_height + k + low;
      blockvec[threadIdx.x] = (idx < batch * vec_height) ? vec[idx] : 0;
    }
    __syncthreads();

    // cherry 参数不用 dequantize & mul
    const int loop_bound = vec_height - off - cherry_size;

    for (j = 0; j < min(loop_bound, BLOCKWIDTH); j += 8) {
      if (j == BLOCKWIDTH / 2) {
        r = res; res = 0;
      }

      // cherry 参数不用 dequantize & mul
      q1 = __ldg(&qweight[i]);
      res += ((q1 & 0x7) - 3.5f) * blockvec[j + 0];
      res += (((q1 >> 3) & 0x7) - 3.5f) * blockvec[j + 1];

      i += width;
      q2 = __ldg(&qweight[i]);
      tmp = (q1 >> 6) | ((q2 << 2) & 0x4);
      res += (tmp - 3.5f) * blockvec[j + 2];
      res += (((q2 >> 1) & 0x7) - 3.5f) * blockvec[j + 3];
      res += (((q2 >> 4) & 0x7) - 3.5f) * blockvec[j + 4];

      i += width;
      q1 = __ldg(&qweight[i]);
      tmp = (q2 >> 7) | ((q1 << 1) & 0x6);
      res += (tmp - 3.5f) * blockvec[j + 5];
      res += (((q1 >> 2) & 0x7) - 3.5f) * blockvec[j + 6];
      res += (((q1 >> 5) & 0x7) - 3.5f) * blockvec[j + 7];
      i += width;
    }

    // 如果访问到第二个 group 需要交换两组的累加和以便与正确的 scaling_factor 相乘
    if (j > BLOCKWIDTH / 2) {
      float t = res; res = r; r = t;
    }
    atomicAdd(&mul[b * width + col], factor_up * res + factor_down * r);
  }
}

__global__ void VecQuant3MatMulKernelCherryQBatchedG64Half(
  const half* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const half* __restrict__ cherry_weight,
  const int16_t* __restrict__   cherry_indices,
  const half* __restrict__ scaling_factors,
  half* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size
) {
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  if (col >= width) return;

  // 处理 vec 中与当前线程块对应的 128 个 float 激活值
  __shared__ float blockvec[BLOCKWIDTH];
  // 当前线程块的每个线程处理一列参数 (128 个 int3)，由于 group_size >= 128，故每列参数对应一个 scaling_factor
  float factor_up, factor_down;

  // 每个线程束中只能含有同一线程块中的线程，因此以下分支指令不会发生线程束分化
  if (blockIdx.x >= gridDim.x - (cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH) {
    // cherry 参数乘法
    float res;
    // k: 当前线程处理的 cherry 参数在 cherry_indices 的第 k 列
    int row = ((cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH - (gridDim.x - blockIdx.x));
    int k = row * BLOCKWIDTH + threadIdx.x;

    #pragma unroll 16 // 循环展开，减少分支指令开销
    for (int b = 0; b < batch; b++) {
      res = 0;
      blockvec[threadIdx.x] = (k < cherry_size) ? __half2float(vec[b * vec_height + cherry_indices[k]]) : 0;
      __syncthreads();
      
      // i: 当前线程处理的 cherry 参数在 cherry_weight 的第 i 行
      int i = row * BLOCKWIDTH;
      int j = 0;
      while (i < cherry_size && j < BLOCKWIDTH) {
        res += __half2float(cherry_weight[i * width + col]) * blockvec[j];
        i++; j++;
      }

      atomicAdd(&mul[b * width + col], __float2half(res));
    }
    return;
  }

  // 处理从第 BLOCKHEIGHT3 * blockIdx.x 行开始向下的 48 行 int8 参数 (= 128 个 int3 参数)
  int row = BLOCKHEIGHT3 * blockIdx.x;
  int off = BLOCKWIDTH * blockIdx.x;

  // 1. 加载 scaling_factors
  // blockIdx.x / (group_size / BLOCKWIDTH): 当前处理的一列参数是第几个 group
  // (vec_height - cherry_size + group_size - 1) / group_size: 除去 cherry 参数后一共能形成几个 group
  {
    int group_idx = blockIdx.x * 2;
    factor_up = (group_idx < (vec_height - cherry_size + 63) / 64) ? __half2float(scaling_factors[group_idx * width + col]) : 0;
    factor_down = (group_idx + 1 < (vec_height - cherry_size + 63) / 64) ? __half2float(scaling_factors[(group_idx + 1) * width + col]) : 0;
  }

  // i: 当前处理的参数在 qweight 中的索引；j: 与当前处理的参数相乘的激活值在 blockvec 中的索引
  int i, j;
  float res, r;
  uint8_t q1, q2, tmp;

  // 2. 加载当前需要相乘的激活值
  // (a) 二分查找 k 在 cherry_indices 中的 upper_bound
  // (b) 如果 k 就在 cherry_indices 中，则将其移动至最后并设为0，否则将其向前移动
  // (c) 注意线程束分化的问题
  int k = blockIdx.x * BLOCKWIDTH + threadIdx.x;
  int low = 0;
  {
    int high = cherry_size;
    while (low < high) {
      int mid = (low + high) >> 1;
      int cmp = (cherry_indices[mid] - mid <= k);
      low = cmp ? mid + 1 : low;
      high = cmp ? high : mid;
    }
  } 
  
  #pragma unroll 16 // 循环展开，减少分支指令开销
  for (int b = 0; b < batch; b++) {
    i = width * row + col;
    res = r = 0;

    {
      int idx = b * vec_height + k + low;
      blockvec[threadIdx.x] = (idx < batch * vec_height) ? __half2float(vec[idx]) : 0;
    }
    __syncthreads();

    // cherry 参数不用 dequantize & mul
    const int loop_bound = vec_height - off - cherry_size;

    for (j = 0; j < min(loop_bound, BLOCKWIDTH); j += 8) {
      if (j == BLOCKWIDTH / 2) {
        r = res; res = 0;
      }
      q1 = __ldg(&qweight[i]);
      res += ((q1 & 0x7) - 3.5f) * blockvec[j];
      res += (((q1 >> 3) & 0x7) - 3.5f) * blockvec[j + 1];

      i += width;
      q2 = __ldg(&qweight[i]);
      tmp = (q1 >> 6) | ((q2 << 2) & 0x4);
      res += (tmp - 3.5f) * blockvec[j + 2];
      res += (((q2 >> 1) & 0x7) - 3.5f) * blockvec[j + 3];
      res += (((q2 >> 4) & 0x7) - 3.5f) * blockvec[j + 4];

      i += width;
      q1 = __ldg(&qweight[i]);
      tmp = (q2 >> 7) | ((q1 << 1) & 0x6);
      res += (tmp - 3.5f) * blockvec[j + 5];
      res += (((q1 >> 2) & 0x7) - 3.5f) * blockvec[j + 6];
      res += (((q1 >> 5) & 0x7) - 3.5f) * blockvec[j + 7];
      i += width;
    }

    // 如果访问到第二个 group 需要交换两组的累加和以便与正确的 scaling_factor 相乘
    if (j > BLOCKWIDTH / 2) {
      float t = res; res = r; r = t;
    }
    atomicAdd(&mul[b * width + col], __float2half(factor_up * res + factor_down * r));
  }
}

// 3bit vec matmul (group_size multiple of 128)
__global__ void VecQuant3MatMulKernelCherryQBatched(
  // __restrict__ 消除指针别名
  const float* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const float* __restrict__ cherry_weight,
  const int* __restrict__   cherry_indices,
  const float* __restrict__ scaling_factors,
  float* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size, int group_size
) {
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  if (col >= width) return;

  // 处理 vec 中与当前线程块对应的 128 个 float 激活值
  __shared__ float blockvec[BLOCKWIDTH];
  // 当前线程块的每个线程处理一列参数 (128 个 int3)，由于 group_size >= 128，故每列参数对应一个 scaling_factor
  float factor;

  // 每个线程束中只能含有同一线程块中的线程，因此以下分支指令不会发生线程束分化
  if (blockIdx.x >= gridDim.x - (cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH) {
    // cherry 参数乘法
    float res;
    // k: 当前线程处理的 cherry 参数在 cherry_indices 的第 k 列
    int row = ((cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH - (gridDim.x - blockIdx.x));
    int k = row * BLOCKWIDTH + threadIdx.x;

    #pragma unroll 16 // 循环展开，减少分支指令开销
    for (int b = 0; b < batch; b++) {
      res = 0;
      blockvec[threadIdx.x] = (k < cherry_size) ? vec[b * vec_height + cherry_indices[k]] : 0;
      __syncthreads();
      
      // i: 当前线程处理的 cherry 参数在 cherry_weight 的第 i 行
      int i = row * BLOCKWIDTH;
      int j = 0;
      while (i < cherry_size && j < BLOCKWIDTH) {
        res += cherry_weight[i * width + col] * blockvec[j];
        i++; j++;
      }

      atomicAdd(&mul[b * width + col], res);
    }
    return;
  }

  // 处理从第 BLOCKHEIGHT3 * blockIdx.x 行开始向下的 48 行 int8 参数 (= 128 个 int3 参数)
  int row = BLOCKHEIGHT3 * blockIdx.x;
  int off = BLOCKWIDTH * blockIdx.x;

  // 1. 加载 scaling_factors
  // blockIdx.x / (group_size / BLOCKWIDTH): 当前处理的一列参数是第几个 group
  // (vec_height - cherry_size + group_size - 1) / group_size: 除去 cherry 参数后一共能形成几个 group
  {
    int group_idx = blockIdx.x / (group_size / BLOCKWIDTH);
    factor = (group_idx < (vec_height - cherry_size + group_size - 1) / group_size) ? scaling_factors[group_idx * width + col] : 0;
  }

  // i: 当前处理的参数在 qweight 中的索引；j: 与当前处理的参数相乘的激活值在 blockvec 中的索引
  int i, j;
  float res;
  uint8_t q1, q2, tmp;

  // 2. 加载当前需要相乘的激活值
  // (a) 二分查找 k 在 cherry_indices 中的 upper_bound
  // (b) 如果 k 就在 cherry_indices 中，则将其移动至最后并设为0，否则将其向前移动
  // (c) 注意线程束分化的问题
  int k = blockIdx.x * BLOCKWIDTH + threadIdx.x;
  int low = 0;
  {
    int high = cherry_size;
    while (low < high) {
      int mid = (low + high) >> 1;
      int cmp = (cherry_indices[mid] - mid <= k);
      low = cmp ? mid + 1 : low;
      high = cmp ? high : mid;
    }
  } 

  #pragma unroll 16 // 循环展开，减少分支指令开销
  for (int b = 0; b < batch; b++) {
    i = width * row + col;
    j = 0, res = 0;

    {
      int idx = b * vec_height + k + low;
      blockvec[threadIdx.x] = (idx < batch * vec_height) ? vec[idx] : 0;
    }
    __syncthreads();

    // cherry 参数不用 dequantize & mul
    const int loop_bound = vec_height - off - cherry_size;

    for (; j < min(loop_bound, BLOCKWIDTH); j += 8) {
      q1 = __ldg(&qweight[i]);
      res += ((q1 & 0x7) - 3.5f) * blockvec[j + 0];
      res += (((q1 >> 3) & 0x7) - 3.5f) * blockvec[j + 1];

      i += width;
      q2 = __ldg(&qweight[i]);
      tmp = (q1 >> 6) | ((q2 << 2) & 0x4);
      res += (tmp - 3.5f) * blockvec[j + 2];
      res += (((q2 >> 1) & 0x7) - 3.5f) * blockvec[j + 3];
      res += (((q2 >> 4) & 0x7) - 3.5f) * blockvec[j + 4];

      i += width;
      q1 = __ldg(&qweight[i]);
      tmp = (q2 >> 7) | ((q1 << 1) & 0x6);
      res += (tmp - 3.5f) * blockvec[j + 5];
      res += (((q1 >> 2) & 0x7) - 3.5f) * blockvec[j + 6];
      res += (((q1 >> 5) & 0x7) - 3.5f) * blockvec[j + 7];
      i += width;
    }
    
    atomicAdd(&mul[b * width + col], factor * res);
  }
}

__global__ void VecQuant3MatMulKernelCherryQBatchedHalf(
  const half* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const half* __restrict__ cherry_weight,
  const int16_t* __restrict__   cherry_indices,
  const half* __restrict__ scaling_factors,
  half* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size, int group_size
) {
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  if (col >= width) return;

  // 处理 vec 中与当前线程块对应的 128 个 float 激活值
  __shared__ float blockvec[BLOCKWIDTH];
  // 当前线程块的每个线程处理一列参数 (128 个 int3)，由于 group_size >= 128，故每列参数对应一个 scaling_factor
  float factor;

  // 每个线程束中只能含有同一线程块中的线程，因此以下分支指令不会发生线程束分化
  if (blockIdx.x >= gridDim.x - (cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH) {
    // cherry 参数乘法
    float res;
    // k: 当前线程处理的 cherry 参数在 cherry_indices 的第 k 列
    int row = ((cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH - (gridDim.x - blockIdx.x));
    int k = row * BLOCKWIDTH + threadIdx.x;

    #pragma unroll 16 // 循环展开，减少分支指令开销
    for (int b = 0; b < batch; b++) {
      res = 0;
      blockvec[threadIdx.x] = (k < cherry_size) ? __half2float(vec[b * vec_height + cherry_indices[k]]) : 0;
      __syncthreads();
      
      // i: 当前线程处理的 cherry 参数在 cherry_weight 的第 i 行
      int i = row * BLOCKWIDTH;
      int j = 0;
      while (i < cherry_size && j < BLOCKWIDTH) {
        res += __half2float(cherry_weight[i * width + col]) * blockvec[j];
        i++; j++;
      }

      atomicAdd(&mul[b * width + col], __float2half(res));
    }
    return;
  }

  // 处理从第 BLOCKHEIGHT3 * blockIdx.x 行开始向下的 48 行 int8 参数 (= 128 个 int3 参数)
  int row = BLOCKHEIGHT3 * blockIdx.x;
  int off = BLOCKWIDTH * blockIdx.x;

  // 1. 加载 scaling_factors
  // blockIdx.x / (group_size / BLOCKWIDTH): 当前处理的一列参数是第几个 group
  // (vec_height - cherry_size + group_size - 1) / group_size: 除去 cherry 参数后一共能形成几个 group
  {
    int group_idx = blockIdx.x / (group_size / BLOCKWIDTH);
    factor = (group_idx < (vec_height - cherry_size + group_size - 1) / group_size) ? __half2float(scaling_factors[group_idx * width + col]) : 0;
  }

  // i: 当前处理的参数在 qweight 中的索引；j: 与当前处理的参数相乘的激活值在 blockvec 中的索引
  int i, j;
  float res;
  uint8_t q1, q2, tmp;

  // 2. 加载当前需要相乘的激活值
  // (a) 二分查找 k 在 cherry_indices 中的 upper_bound
  // (b) 如果 k 就在 cherry_indices 中，则将其移动至最后并设为0，否则将其向前移动
  // (c) 注意线程束分化的问题
  int k = blockIdx.x * BLOCKWIDTH + threadIdx.x;
  int low = 0;
  {
    int high = cherry_size;
    while (low < high) {
      int mid = (low + high) >> 1;
      int cmp = (cherry_indices[mid] - mid <= k);
      low = cmp ? mid + 1 : low;
      high = cmp ? high : mid;
    }
  } 

  #pragma unroll 16 // 循环展开，减少分支指令开销
  for (int b = 0; b < batch; b++) {
    i = width * row + col;
    j = 0, res = 0;

    {
      int idx = b * vec_height + k + low;
      blockvec[threadIdx.x] = (idx < batch * vec_height) ? __half2float(vec[idx]) : 0;
    }
    __syncthreads();

    // cherry 参数不用 dequantize & mul
    const int loop_bound = vec_height - off - cherry_size;

    for (; j < min(loop_bound, BLOCKWIDTH); j += 8) {
      q1 = __ldg(&qweight[i]);
      res += ((q1 & 0x7) - 3.5f) * blockvec[j];
      res += (((q1 >> 3) & 0x7) - 3.5f) * blockvec[j + 1];

      i += width;
      q2 = __ldg(&qweight[i]);
      tmp = (q1 >> 6) | ((q2 << 2) & 0x4);
      res += (tmp - 3.5f) * blockvec[j + 2];
      res += (((q2 >> 1) & 0x7) - 3.5f) * blockvec[j + 3];
      res += (((q2 >> 4) & 0x7) - 3.5f) * blockvec[j + 4];

      i += width;
      q1 = __ldg(&qweight[i]);
      tmp = (q2 >> 7) | ((q1 << 1) & 0x6);
      res += (tmp - 3.5f) * blockvec[j + 5];
      res += (((q1 >> 2) & 0x7) - 3.5f) * blockvec[j + 6];
      res += (((q1 >> 5) & 0x7) - 3.5f) * blockvec[j + 7];
      i += width;
    }
    
    atomicAdd(&mul[b * width + col], __float2half(factor * res));
  }
}


void vecquant4matmul_cherryq_cuda(
  torch::Tensor vec,
  torch::Tensor qweight,
  torch::Tensor cherry_weight,
  torch::Tensor cherry_indices,
  torch::Tensor scaling_factors,
  torch::Tensor mul,
  int group_size
) {
  if (group_size < 128) AT_ERROR("`group_size` must be multiple of 128");
  
  int height = qweight.size(0);
  int width = qweight.size(1);
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int cherry_size = cherry_indices.size(0);

  // 指派额外的线程块进行 cherry_weight 的浮点数乘法
  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4 + (cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelCherryQBatchedHalf<<<blocks, threads>>>(
    reinterpret_cast<half*>(vec.data_ptr<at::Half>()),
    qweight.data_ptr<uint8_t>(),
    reinterpret_cast<half*>(cherry_weight.data_ptr<at::Half>()),
    cherry_indices.data_ptr<int16_t>(),
    reinterpret_cast<half*>(scaling_factors.data_ptr<at::Half>()),
    reinterpret_cast<half*>(mul.data_ptr<at::Half>()),
    width, batch, vec_height, cherry_size, group_size
  );

  // cudaDeviceSynchronize();
  // // 检查同步是否出错
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //     printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  //     exit(1);
  // }
}

__global__ void VecQuant4MatMulKernelCherryQBatchedHalf(
  // __restrict__ 消除指针别名
  const half* __restrict__ vec,
  const uint8_t* __restrict__   qweight,
  const half* __restrict__ cherry_weight,
  const int16_t* __restrict__   cherry_indices,
  const half* __restrict__ scaling_factors,
  half* __restrict__       mul,
  int width, int batch, int vec_height, int cherry_size, int group_size
) {
  // 与 VecQuant3MatMulKernelCherryQBatchedHalf 相同
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  if (col >= width) return;

  // 处理 vec 中与当前线程块对应的 128 个 float 激活值
  __shared__ float blockvec[BLOCKWIDTH];
  // 当前线程块的每个线程处理一列参数 (128 个 int4)，由于 group_size >= 128，故每列参数对应一个 scaling_factor
  float factor;

  // 每个线程束中只能含有同一线程块中的线程，因此以下分支指令不会发生线程束分化
  if (blockIdx.x >= gridDim.x - (cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH) {
    // cherry 参数乘法
    float res;
    // k: 当前线程处理的 cherry 参数在 cherry_indices 的第 k 列
    int row = ((cherry_size + BLOCKWIDTH - 1) / BLOCKWIDTH - (gridDim.x - blockIdx.x));
    int k = row * BLOCKWIDTH + threadIdx.x;

    #pragma unroll 16 // 循环展开，减少分支指令开销
    for (int b = 0; b < batch; b++) {
      res = 0;
      blockvec[threadIdx.x] = (k < cherry_size) ? __half2float(vec[b * vec_height + cherry_indices[k]]) : 0;
      __syncthreads();
      
      // i: 当前线程处理的 cherry 参数在 cherry_weight 的第 i 行
      int i = row * BLOCKWIDTH;
      int j = 0;
      while (i < cherry_size && j < BLOCKWIDTH) {
        res += __half2float(cherry_weight[i * width + col]) * blockvec[j];
        i++; j++;
      }

      atomicAdd(&mul[b * width + col], __float2half(res));
    }
    return;
  }

  // 处理从第 BLOCKHEIGHT4 * blockIdx.x 行开始向下的 64 行 int8 参数 (= 128 个 int4 参数)
  int row = BLOCKHEIGHT4 * blockIdx.x;
  int off = BLOCKWIDTH * blockIdx.x;
  
  // 1. 加载 scaling_factors
  // blockIdx.x / (group_size / BLOCKWIDTH): 当前处理的一列参数是第几个 group
  // (vec_height - cherry_size + group_size - 1) / group_size: 除去 cherry 参数后一共能形成几个 group
  {
    int group_idx = blockIdx.x / (group_size / BLOCKWIDTH);
    factor = (group_idx < (vec_height - cherry_size + group_size - 1) / group_size) ? __half2float(scaling_factors[group_idx * width + col]) : 0;
  }

  // i: 当前处理的参数在 qweight 中的索引；j: 与当前处理的参数相乘的激活值在 blockvec 中的索引
  int i, j;
  float res;
  uint8_t q;

  // 2. 加载当前需要相乘的激活值
  // (a) 二分查找 k 在 cherry_indices 中的 upper_bound
  // (b) 如果 k 就在 cherry_indices 中，则将其移动至最后并设为0，否则将其向前移动
  // (c) 注意线程束分化的问题
  int k = blockIdx.x * BLOCKWIDTH + threadIdx.x;
  int low = 0;
  {
    int high = cherry_size;
    while (low < high) {
      int mid = (low + high) >> 1;
      int cmp = (cherry_indices[mid] - mid <= k);
      low = cmp ? mid + 1 : low;
      high = cmp ? high : mid;
    }
  } 

  #pragma unroll 16 // 循环展开，减少分支指令开销
  for (int b = 0; b < batch; b++) {
    i = width * row + col;
    j = 0, res = 0;

    {
      int idx = b * vec_height + k + low;
      blockvec[threadIdx.x] = (idx < batch * vec_height) ? __half2float(vec[idx]) : 0;
    }
    __syncthreads();

    // cherry 参数不用 dequantize & mul
    const int loop_bound = vec_height - off - cherry_size;

    for (; j < min(loop_bound, BLOCKWIDTH); j += 8) {
      q = __ldg(&qweight[i]);
      res += ((q & 0xf) - 7.5f) * blockvec[j];
      res += (((q >> 4) & 0xf) - 7.5f) * blockvec[j + 1];

      i += width;
      q = __ldg(&qweight[i]);
      res += ((q & 0xf) - 7.5f) * blockvec[j + 2];
      res += (((q >> 4) & 0xf) - 7.5f) * blockvec[j + 3];

      i += width;
      q = __ldg(&qweight[i]);
      res += ((q & 0xf) - 7.5f) * blockvec[j + 4];
      res += (((q >> 4) & 0xf) - 7.5f) * blockvec[j + 5];

      i += width;
      q = __ldg(&qweight[i]);
      res += ((q & 0xf) - 7.5f) * blockvec[j + 6];
      res += (((q >> 4) & 0xf) - 7.5f) * blockvec[j + 7];
      i += width;
    }
    
    atomicAdd(&mul[b * width + col], __float2half(factor * res));
  }
}


void Dequantize3Normal(
  torch::Tensor qweight,
  torch::Tensor scaling_factors,
  torch::Tensor out,
  int height, int cherry_size, int group_size
) {
  int width = qweight.size(1);

  dim3 blocks(
    (qweight.size(0) + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  Dequantize3NormalKernel<<<blocks, threads>>>(
    qweight.data_ptr<uint8_t>(),
    scaling_factors.data_ptr<float>(),
    out.data_ptr<float>(),
    height, width, cherry_size, group_size
  );

  // cudaDeviceSynchronize();
  // // 检查同步是否出错
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //     printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  //     exit(1);
  // }
}

// 3bit dequantize normal params (group_size multiple of 128)
__global__ void Dequantize3NormalKernel(
  // __restrict__ 消除指针别名
  const uint8_t* __restrict__   qweight,
  const float* __restrict__ scaling_factors,
  float* __restrict__       out,
  int height, int width, int cherry_size, int group_size
) {
  // 处理从第 row 行开始向下的 48 行 int8 参数 (= 128 个 int3 参数)
  int row = BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int off = BLOCKWIDTH * blockIdx.x;
  
  if (col >= width) return;

  // 当前线程块的每个线程处理一列参数 (128 个 int3)，由于 group_size >= 128，故每列参数对应一个 scaling_factor
  float factor;

  // 1. 加载 scaling_factors
  // blockIdx.x / (group_size / BLOCKWIDTH): 当前处理的一列参数是第几个 group
  // (vec_height - cherry_size + group_size - 1) / group_size: 除去 cherry 参数后一共能形成几个 group
  {
    int group_idx = blockIdx.x / (group_size / BLOCKWIDTH);
    factor = (group_idx < (height - cherry_size + group_size - 1) / group_size) ? scaling_factors[group_idx * width + col] : 0;
  }

  // i: 当前处理的参数在 qweight 中的索引
  int i = width * row + col;
  uint8_t q1, q2, tmp;

  // cherry 参数不用 dequantize
  const int loop_bound = height - off - cherry_size;

  #pragma unroll 16 // 循环展开，减少分支指令开销
  for (int j = 0; j < min(loop_bound, BLOCKWIDTH); j += 8) {
    q1 = __ldg(&qweight[i]);
    atomicExch(&out[(off + j) * width + col], ((q1 & 0x7) - 3.5f) * factor);
    atomicExch(&out[(off + j + 1) * width + col], (((q1 >> 3) & 0x7) - 3.5f) * factor);

    i += width;
    q2 = __ldg(&qweight[i]);
    tmp = (q1 >> 6) | ((q2 << 2) & 0x4);
    atomicExch(&out[(off + j + 2) * width + col], (tmp - 3.5f) * factor);
    atomicExch(&out[(off + j + 3) * width + col], (((q2 >> 1) & 0x7) - 3.5f) * factor);
    atomicExch(&out[(off + j + 4) * width + col], (((q2 >> 4) & 0x7) - 3.5f) * factor);

    i += width;
    q1 = __ldg(&qweight[i]);
    tmp = (q2 >> 7) | ((q1 << 1) & 0x6);
    atomicExch(&out[(off + j + 5) * width + col], (tmp - 3.5f) * factor);
    atomicExch(&out[(off + j + 6) * width + col], (((q1 >> 2) & 0x7) - 3.5f) * factor);
    atomicExch(&out[(off + j + 7) * width + col], (((q1 >> 5) & 0x7) - 3.5f) * factor);
    i += width;
  }
}


void Dequantize4Normal(
  torch::Tensor qweight,
  torch::Tensor scaling_factors,
  torch::Tensor out,
  int height, int cherry_size, int group_size
) {
  int width = qweight.size(1);

  dim3 blocks(
    (qweight.size(0) + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  Dequantize4NormalKernel<<<blocks, threads>>>(
    qweight.data_ptr<uint8_t>(),
    scaling_factors.data_ptr<float>(),
    out.data_ptr<float>(),
    height, width, cherry_size, group_size
  );

  // cudaDeviceSynchronize();
  // // 检查同步是否出错
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //     printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  //     exit(1);
  // }
}

// 3bit dequantize normal params (group_size multiple of 128)
__global__ void Dequantize4NormalKernel(
  // __restrict__ 消除指针别名
  const uint8_t* __restrict__   qweight,
  const float* __restrict__ scaling_factors,
  float* __restrict__       out,
  int height, int width, int cherry_size, int group_size
) {
  // 处理从第 row 行开始向下的 48 行 int8 参数 (= 128 个 int3 参数)
  int row = BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int off = BLOCKWIDTH * blockIdx.x;
  
  if (col >= width) return;

  // 当前线程块的每个线程处理一列参数 (128 个 int3)，由于 group_size >= 128，故每列参数对应一个 scaling_factor
  float factor;

  // 1. 加载 scaling_factors
  // blockIdx.x / (group_size / BLOCKWIDTH): 当前处理的一列参数是第几个 group
  // (vec_height - cherry_size + group_size - 1) / group_size: 除去 cherry 参数后一共能形成几个 group
  {
    int group_idx = blockIdx.x / (group_size / BLOCKWIDTH);
    factor = (group_idx < (height - cherry_size + group_size - 1) / group_size) ? scaling_factors[group_idx * width + col] : 0;
  }

  // i: 当前处理的参数在 qweight 中的索引
  int i = width * row + col;
  uint8_t q;

  // cherry 参数不用 dequantize
  const int loop_bound = height - off - cherry_size;

  #pragma unroll 16 // 循环展开，减少分支指令开销
  for (int j = 0; j < min(loop_bound, BLOCKWIDTH); j += 8) {
    q = __ldg(&qweight[i]);
    atomicExch(&out[(off + j) * width + col], ((q & 0xf) - 7.5f) * factor);
    atomicExch(&out[(off + j + 1) * width + col], (((q >> 4) & 0xf) - 7.5f) * factor);

    i += width;
    q = __ldg(&qweight[i]);
    atomicExch(&out[(off + j + 2) * width + col], ((q & 0xf) - 7.5f) * factor);
    atomicExch(&out[(off + j + 3) * width + col], (((q >> 4) & 0xf) - 7.5f) * factor);

    i += width;
    q = __ldg(&qweight[i]);
    atomicExch(&out[(off + j + 4) * width + col], ((q & 0xf) - 7.5f) * factor);
    atomicExch(&out[(off + j + 5) * width + col], (((q >> 4) & 0xf) - 7.5f) * factor);

    i += width;
    q = __ldg(&qweight[i]);
    atomicExch(&out[(off + j + 6) * width + col], ((q & 0xf) - 7.5f) * factor);
    atomicExch(&out[(off + j + 7) * width + col], (((q >> 4) & 0xf) - 7.5f) * factor);
    i += width;
  }
}
