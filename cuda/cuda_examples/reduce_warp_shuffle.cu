/**
 * 1D reduce (sum) using warp shuffle — no naive "single thread sums all".
 *
 * Pattern: grid-stride partial per thread -> warp shuffle -> shared for cross-warp ->
 * one partial per block. Second kernel: <=1024 partials -> same block reduction -> scalar.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ float block_reduce_sum(float partial, float* smem) {
    const int lane = threadIdx.x % kWarpSize;
    const int wid = threadIdx.x / kWarpSize;
    const int nw = blockDim.x / kWarpSize;

    partial = warp_reduce_sum(partial);
    if (lane == 0)
        smem[wid] = partial;
    __syncthreads();

    float block_sum = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.f;
    if (wid == 0)
        block_sum = warp_reduce_sum(block_sum);
    return block_sum;
}

__global__ void reduce_sum_kernel(const float* __restrict__ in, float* __restrict__ out_partial,
                                  int n) {
    extern __shared__ float smem[];
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        sum += in[i];

    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0)
        out_partial[blockIdx.x] = sum;
}

/** nblocks <= blockDim.x (e.g. 1024); each thread holds one partial or 0 */
__global__ void reduce_sum_final(const float* __restrict__ partial, float* __restrict__ out,
                                 int nblocks) {
    extern __shared__ float smem[];
    float sum = (threadIdx.x < nblocks) ? partial[threadIdx.x] : 0.f;
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0)
        *out = sum;
}

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    const int n = 1 << 22;
    const int threads = 256;
    const int blocks = std::min(1024, std::max(1, (n + threads - 1) / threads));

    float *d_in, *d_partial, *d_out;
    check(cudaMalloc(&d_in, n * sizeof(float)), "malloc in");
    check(cudaMalloc(&d_partial, blocks * sizeof(float)), "malloc partial");
    check(cudaMalloc(&d_out, sizeof(float)), "malloc out");

    std::vector<float> h(n, 1.0f);
    check(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice), "memcpy");

    size_t shmem = (threads / kWarpSize) * sizeof(float);
    reduce_sum_kernel<<<blocks, threads, shmem>>>(d_in, d_partial, n);
    check(cudaGetLastError(), "kernel1");

    const int final_threads = 1024;
    reduce_sum_final<<<1, final_threads, (final_threads / kWarpSize) * sizeof(float)>>>(
        d_partial, d_out, blocks);
    check(cudaGetLastError(), "kernel2");

    float result = 0.f;
    check(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost), "dtoh");

    float ref = static_cast<float>(n);
    printf("reduce sum: got %.6f expect %.6f err=%g\n", result, ref, fabs(result - ref));

    cudaFree(d_in);
    cudaFree(d_partial);
    cudaFree(d_out);
    return fabs(result - ref) < 1e-3f * ref ? 0 : 1;
}
