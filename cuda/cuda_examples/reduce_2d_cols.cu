/**
 * 2D reduce: (rows, cols) -> (1, cols)  sum along dim 0 (each column one scalar).
 * Example: (1_000_000, 128) row-major -> out[c] = sum_r in[r*cols + c]
 *
 * One CUDA block per column: threads first accumulate strided sums, then
 * warp-shuffle + shared reduction inside the block (no naive single-thread column sum).
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

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

/**
 * blockIdx.x == column index c
 * Each thread sums a subset of rows in that column (stride = blockDim.x).
 */
__global__ void reduce_cols_kernel(const float* __restrict__ in, float* __restrict__ out,
                                   int rows, int cols) {
    extern __shared__ float smem[];
    const int c = blockIdx.x;
    if (c >= cols)
        return;

    float sum = 0.f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x)
        sum += in[static_cast<size_t>(r) * cols + c];

    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0)
        out[c] = sum;
}

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    const int rows = 1'000'000;
    const int cols = 128;
    const int threads = 256; // tunable; must be multiple of 32
    const int shmem = (threads / kWarpSize) * sizeof(float);

    std::vector<float> h(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            h[static_cast<size_t>(r) * cols + c] = 1.0f; // sum per col = rows

    float *d_in, *d_out;
    check(cudaMalloc(&d_in, h.size() * sizeof(float)), "malloc in");
    check(cudaMalloc(&d_out, cols * sizeof(float)), "malloc out");
    check(cudaMemcpy(d_in, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy");

    reduce_cols_kernel<<<cols, threads, shmem>>>(d_in, d_out, rows, cols);
    check(cudaGetLastError(), "kernel");
    check(cudaDeviceSynchronize(), "sync");

    std::vector<float> got(cols);
    check(cudaMemcpy(got.data(), d_out, cols * sizeof(float), cudaMemcpyDeviceToHost), "dtoh");

    const float ref = static_cast<float>(rows);
    float max_err = 0.f;
    for (int c = 0; c < cols; ++c)
        max_err = fmax(max_err, fabs(got[c] - ref));
    printf("reduce_2d cols: max_err=%g (expect each col %f)\n", max_err, ref);

    cudaFree(d_in);
    cudaFree(d_out);
    return max_err < 1e-2f * ref ? 0 : 1;
}
