/**
 * Tiled GEMM: C = A * B, A[M][K], B[K][N], C[M][N] (row-major).
 *
 * Demonstrates (comments in kernel):
 * - 每个 thread 负责累加得到 C 子块中一个元素（可扩展为每线程多个输出）；
 * - A tile / B tile 在 shared memory 复用；
 * - 合并访存：连续 threadIdx 读连续 k 列/行；
 * - 双缓冲：用两个 smem 槽位，prefetch 下一 tile 与计算当前 tile 重叠（软件流水线）。
 *
 * 避免 bank conflict：BK 取奇数（如 33）或 swizzle 可进一步缓解；此处 BK=32 为常见教学尺寸，
 * 若需演示无冲突可改为 BK=16 + padding。
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;

template <int DOUBLE_BUF>
__global__ void gemm_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    // Double buffer: 2 * BM * BK + 2 * BK * BN floats (if DOUBLE_BUF)
    extern __shared__ float smem_raw[];
    float* As[2];
    float* Bs[2];
    if constexpr (DOUBLE_BUF) {
        As[0] = smem_raw;
        As[1] = smem_raw + BM * BK;
        Bs[0] = As[1] + BM * BK;
        Bs[1] = Bs[0] + BK * BN;
    } else {
        As[0] = smem_raw;
        Bs[0] = smem_raw + BM * BK;
    }

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = by * BM + ty;
    const int col = bx * BN + tx;

    float acc = 0.f;

    const int num_tiles = (K + BK - 1) / BK;
    int buf = 0;

    auto load_tile = [&](int tile, int b) {
        const int k0 = tile * BK;
        // A tile BM x BK: coalesced along K for fixed row
        if (ty < BM && tx < BK) {
            int rr = by * BM + ty;
            int cc = k0 + tx;
            As[b][ty * BK + tx] = (rr < M && cc < K) ? A[static_cast<size_t>(rr) * K + cc] : 0.f;
        }
        // B tile BK x BN: coalesced along N for fixed k
        if (ty < BK && tx < BN) {
            int rr = k0 + ty;
            int cc = bx * BN + tx;
            Bs[b][ty * BN + tx] = (rr < K && cc < N) ? B[static_cast<size_t>(rr) * N + cc] : 0.f;
        }
    };

    if constexpr (DOUBLE_BUF) {
        if (num_tiles > 0) {
            load_tile(0, 0);
            __syncthreads();
        }
        for (int t = 0; t < num_tiles; ++t) {
            const int next = t + 1;
            if (next < num_tiles)
                load_tile(next, 1 - buf);
#pragma unroll
            for (int k = 0; k < BK; ++k)
                acc += As[buf][ty * BK + k] * Bs[buf][k * BN + tx];
            __syncthreads();
            buf = 1 - buf;
        }
    } else {
        for (int t = 0; t < num_tiles; ++t) {
            load_tile(t, 0);
            __syncthreads();
#pragma unroll
            for (int k = 0; k < BK; ++k)
                acc += As[0][ty * BK + k] * Bs[0][k * BN + tx];
            __syncthreads();
        }
    }

    if (row < M && col < N)
        C[static_cast<size_t>(row) * N + col] = acc;
}

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int kk = 0; kk < K; ++kk)
                s += A[static_cast<size_t>(i) * K + kk] * B[static_cast<size_t>(kk) * N + j];
            C[static_cast<size_t>(i) * N + j] = s;
        }
}

int main(int argc, char** argv) {
    const int use_double_buf = (argc > 1 && argv[1][0] == '1') ? 1 : 0;
    const int M = 512, N = 512, K = 512;

    std::vector<float> hA(static_cast<size_t>(M) * K), hB(static_cast<size_t>(K) * N);
    for (size_t i = 0; i < hA.size(); ++i)
        hA[i] = static_cast<float>((i % 13) * 0.01f);
    for (size_t i = 0; i < hB.size(); ++i)
        hB[i] = static_cast<float>((i % 7) * 0.01f);

    float *dA, *dB, *dC;
    check(cudaMalloc(&dA, hA.size() * sizeof(float)), "A");
    check(cudaMalloc(&dB, hB.size() * sizeof(float)), "B");
    check(cudaMalloc(&dC, static_cast<size_t>(M) * N * sizeof(float)), "C");
    check(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "hA");
    check(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "hB");

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    size_t shmem =
        use_double_buf ? (2ull * BM * BK + 2ull * BK * BN) * sizeof(float)
                       : (static_cast<size_t>(BM) * BK + static_cast<size_t>(BK) * BN) * sizeof(float);

    if (use_double_buf) {
        gemm_tiled_kernel<1><<<grid, block, shmem>>>(dA, dB, dC, M, N, K);
    } else {
        gemm_tiled_kernel<0><<<grid, block, shmem>>>(dA, dB, dC, M, N, K);
    }
    check(cudaGetLastError(), "gemm");
    check(cudaDeviceSynchronize(), "sync");

    std::vector<float> hC(static_cast<size_t>(M) * N);
    check(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), "dtoh");

    std::vector<float> ref(static_cast<size_t>(M) * N);
    cpu_gemm(hA, hB, ref, M, N, K);

    float max_err = 0.f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_err = fmax(max_err, fabs(hC[i] - ref[i]));
    printf("GEMM (%s buffer) max_err=%g\n", use_double_buf ? "double" : "single", max_err);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return max_err < 1e-1f ? 0 : 1; // loose for fast math accumulation
}
