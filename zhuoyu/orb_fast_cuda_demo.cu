// orb_fast_cuda_demo.cu
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>

struct KeyPoint {
    int x, y, score;
};

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

// FAST circle offsets (radius=3), clockwise 16 points
__constant__ int c_dx[16] = {0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
__constant__ int c_dy[16] = {-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};

__global__ void fast_score_kernel(
    const unsigned char* img, int W, int H, int stride,
    int threshold, int* score_map)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 3 || x >= W - 3 || y < 3 || y >= H - 3) return;

    int c = img[y * stride + x];
    int bright[16], dark[16];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int p = img[(y + c_dy[i]) * stride + (x + c_dx[i])];
        bright[i] = (p > c + threshold);
        dark[i]   = (p < c - threshold);
    }

    // FAST-9 contiguous arc test
    bool is_corner = false;
    int best = 0;
    for (int s = 0; s < 16; ++s) {
        int cb = 0, cd = 0, min_diff = 255;
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            int idx = (s + k) & 15;
            cb += bright[idx];
            cd += dark[idx];
            int p = img[(y + c_dy[idx]) * stride + (x + c_dx[idx])];
            min_diff = min(min_diff, abs(p - c));
        }
        if (cb == 9 || cd == 9) {
            is_corner = true;
            best = max(best, min_diff); // simple FAST score proxy
        }
    }

    if (is_corner) score_map[y * W + x] = best;
}

__global__ void nms_kernel(
    const int* score_map, int W, int H, int* nms_map)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) return;

    int s = score_map[y * W + x];
    if (s == 0) return;

    bool keep = true;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int t = score_map[(y + dy) * W + (x + dx)];
            if (t > s) keep = false;
        }
    }
    if (keep) nms_map[y * W + x] = s;
}

__global__ void collect_kernel(
    const int* nms_map, int W, int H,
    KeyPoint* out_kps, int max_kps, int* counter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int s = nms_map[y * W + x];
    if (s > 0) {
        int idx = atomicAdd(counter, 1);
        if (idx < max_kps) out_kps[idx] = {x, y, s};
    }
}

int main() {
    const int W = 640, H = 480, stride = W;
    std::vector<unsigned char> h_img(W * H, 80);

    // 造一些高对比纹理（演示）
    for (int y = 40; y < H - 40; y += 24)
        for (int x = 40; x < W - 40; x += 24)
            h_img[y * W + x] = 250;

    unsigned char* d_img = nullptr;
    int* d_score = nullptr;
    int* d_nms = nullptr;
    int* d_cnt = nullptr;
    KeyPoint* d_kps = nullptr;

    int max_kps = 20000;
    CUDA_CHECK(cudaMalloc(&d_img, W * H));
    CUDA_CHECK(cudaMalloc(&d_score, W * H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nms, W * H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_kps, max_kps * sizeof(KeyPoint)));

    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), W * H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_score, 0, W * H * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nms, 0, W * H * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(int)));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    fast_score_kernel<<<grid, block>>>(d_img, W, H, stride, 20, d_score);
    nms_kernel<<<grid, block>>>(d_score, W, H, d_nms);
    collect_kernel<<<grid, block>>>(d_nms, W, H, d_kps, max_kps, d_cnt);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_cnt = 0;
    CUDA_CHECK(cudaMemcpy(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
    h_cnt = std::min(h_cnt, max_kps);

    std::vector<KeyPoint> h_kps(h_cnt);
    if (h_cnt > 0) CUDA_CHECK(cudaMemcpy(h_kps.data(), d_kps, h_cnt * sizeof(KeyPoint), cudaMemcpyDeviceToHost));

    std::cout << "Detected keypoints: " << h_cnt << "\n";
    for (int i = 0; i < std::min(10, h_cnt); ++i) {
        std::cout << "(" << h_kps[i].x << "," << h_kps[i].y
                  << "), score=" << h_kps[i].score << "\n";
    }

    cudaFree(d_img); cudaFree(d_score); cudaFree(d_nms); cudaFree(d_cnt); cudaFree(d_kps);
    return 0;
}