// orb_fast_cuda_streams_demo.cu
// A more complete CUDA streams demo for ORB FAST-style feature detection.
//
// What this demonstrates:
//   1. One stream per camera: 6 camera images can be processed independently.
//   2. Per-stream pipeline: H2D -> memset -> FAST -> NMS -> collect -> D2H.
//   3. cudaMemcpyAsync + pinned host memory to overlap copy and compute.
//   4. Same-stream ordering preserves dependencies between FAST/NMS/collect.
//
// Build:
//   nvcc -O3 -arch=sm_86 orb_fast_cuda_streams_demo.cu -o orb_fast_streams
//
// Run:
//   ./orb_fast_streams

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

struct KeyPoint {
    int x;
    int y;
    int score;
};

#define CUDA_CHECK(x) do {                                            \
  cudaError_t err = (x);                                             \
  if (err != cudaSuccess) {                                           \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1);                                                     \
  }                                                                   \
} while (0)

// FAST circle offsets (radius=3), clockwise 16 points.
__constant__ int c_dx[16] = {0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
__constant__ int c_dy[16] = {-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};

__global__ void fast_score_kernel(
    const unsigned char* img, int W, int H, int stride,
    int threshold, int* score_map) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 3 || x >= W - 3 || y < 3 || y >= H - 3) return;

    int c = img[y * stride + x];
    int bright[16];
    int dark[16];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int p = img[(y + c_dy[i]) * stride + (x + c_dx[i])];
        bright[i] = (p > c + threshold);
        dark[i]   = (p < c - threshold);
    }

    bool is_corner = false;
    int best = 0;
    for (int s = 0; s < 16; ++s) {
        int cb = 0;
        int cd = 0;
        int min_diff = 255;
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
            best = max(best, min_diff);
        }
    }

    if (is_corner) score_map[y * W + x] = best;
}

__global__ void nms_kernel(const int* score_map, int W, int H, int* nms_map) {
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
    KeyPoint* out_kps, int max_kps, int* counter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int s = nms_map[y * W + x];
    if (s > 0) {
        int idx = atomicAdd(counter, 1);
        if (idx < max_kps) out_kps[idx] = {x, y, s};
    }
}

struct HostPinnedBuffers {
    unsigned char* images = nullptr;  // [num_frames, num_cams, H, W]
    int* counts = nullptr;            // [num_frames, num_cams]
    KeyPoint* keypoints = nullptr;    // [num_frames, num_cams, max_kps]

    void allocate(int num_frames, int num_cams, int image_bytes, int max_kps) {
        CUDA_CHECK(cudaMallocHost(&images, (size_t)num_frames * num_cams * image_bytes));
        CUDA_CHECK(cudaMallocHost(&counts, (size_t)num_frames * num_cams * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&keypoints,
                                  (size_t)num_frames * num_cams * max_kps * sizeof(KeyPoint)));
    }

    void release() {
        if (images) cudaFreeHost(images);
        if (counts) cudaFreeHost(counts);
        if (keypoints) cudaFreeHost(keypoints);
    }
};

struct CameraStreamContext {
    cudaStream_t stream = nullptr;
    unsigned char* d_img = nullptr;
    int* d_score = nullptr;
    int* d_nms = nullptr;
    int* d_count = nullptr;
    KeyPoint* d_keypoints = nullptr;

    void allocate(int image_bytes, int num_pixels, int max_kps) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(&d_img, image_bytes));
        CUDA_CHECK(cudaMalloc(&d_score, (size_t)num_pixels * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nms, (size_t)num_pixels * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_keypoints, (size_t)max_kps * sizeof(KeyPoint)));
    }

    void release() {
        if (d_img) cudaFree(d_img);
        if (d_score) cudaFree(d_score);
        if (d_nms) cudaFree(d_nms);
        if (d_count) cudaFree(d_count);
        if (d_keypoints) cudaFree(d_keypoints);
        if (stream) cudaStreamDestroy(stream);
    }
};

void fill_synthetic_image(unsigned char* img, int W, int H, int frame_id, int cam_id) {
    std::fill(img, img + W * H, static_cast<unsigned char>(70 + cam_id * 5));

    // Deterministic high-contrast points. Offset each frame/camera to mimic motion.
    int ox = (frame_id * 7 + cam_id * 13) % 24;
    int oy = (frame_id * 5 + cam_id * 11) % 24;
    for (int y = 40 + oy; y < H - 40; y += 24) {
        for (int x = 40 + ox; x < W - 40; x += 24) {
            img[y * W + x] = 250;
            if (x + 1 < W) img[y * W + x + 1] = 20;
            if (y + 1 < H) img[(y + 1) * W + x] = 20;
        }
    }
}

int main() {
    constexpr int W = 640;
    constexpr int H = 480;
    constexpr int stride = W;
    constexpr int num_cams = 6;
    constexpr int num_frames = 8;  // 8帧合成图像
    constexpr int max_kps = 20000; // 输出数组容量上限，不是检测阈值
    constexpr int threshold = 20;


    const int image_bytes = W * H;
    const int num_pixels = W * H;

    HostPinnedBuffers host;
    host.allocate(num_frames, num_cams, image_bytes, max_kps);

    for (int f = 0; f < num_frames; ++f) {
        for (int c = 0; c < num_cams; ++c) {
            unsigned char* img = host.images + ((size_t)f * num_cams + c) * image_bytes;
            fill_synthetic_image(img, W, H, f, c);
            host.counts[f * num_cams + c] = 0;
        }
    }

    std::vector<CameraStreamContext> cams(num_cams);
    for (auto& cam : cams) {
        cam.allocate(image_bytes, num_pixels, max_kps);
    }

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    for (int f = 0; f < num_frames; ++f) {
        for (int c = 0; c < num_cams; ++c) {
            CameraStreamContext& ctx = cams[c];
            cudaStream_t s = ctx.stream;

            unsigned char* h_img = host.images + ((size_t)f * num_cams + c) * image_bytes;
            int* h_count = host.counts + f * num_cams + c;
            KeyPoint* h_kps = host.keypoints + ((size_t)f * num_cams + c) * max_kps;

            // Same stream preserves dependency order for this camera:
            // H2D -> memset -> FAST -> NMS -> collect -> D2H.
            CUDA_CHECK(cudaMemcpyAsync(ctx.d_img, h_img, image_bytes,
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemsetAsync(ctx.d_score, 0, (size_t)num_pixels * sizeof(int), s));
            CUDA_CHECK(cudaMemsetAsync(ctx.d_nms, 0, (size_t)num_pixels * sizeof(int), s));
            CUDA_CHECK(cudaMemsetAsync(ctx.d_count, 0, sizeof(int), s));

            fast_score_kernel<<<grid, block, 0, s>>>(
                ctx.d_img, W, H, stride, threshold, ctx.d_score);
            nms_kernel<<<grid, block, 0, s>>>(ctx.d_score, W, H, ctx.d_nms);
            collect_kernel<<<grid, block, 0, s>>>(
                ctx.d_nms, W, H, ctx.d_keypoints, max_kps, ctx.d_count);

            CUDA_CHECK(cudaMemcpyAsync(h_count, ctx.d_count, sizeof(int),
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaMemcpyAsync(h_kps, ctx.d_keypoints,
                                       (size_t)max_kps * sizeof(KeyPoint),
                                       cudaMemcpyDeviceToHost, s));
        }
    }

    for (auto& cam : cams) {
        CUDA_CHECK(cudaStreamSynchronize(cam.stream));
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));

    std::cout << "=== ORB FAST CUDA Streams Demo ===\n";
    std::cout << "Frames=" << num_frames << " Cameras=" << num_cams
              << " Image=" << W << "x" << H << "\n";
    std::cout << "One stream per camera, each stream runs H2D->FAST->NMS->collect->D2H\n";
    std::cout << "Total time: " << total_ms << " ms\n\n";

    for (int f = 0; f < std::min(num_frames, 2); ++f) {
        for (int c = 0; c < num_cams; ++c) {
            int cnt = std::min(host.counts[f * num_cams + c], max_kps);
            KeyPoint* kps = host.keypoints + ((size_t)f * num_cams + c) * max_kps;
            std::cout << "frame " << f << " cam " << c
                      << " keypoints=" << cnt;
            if (cnt > 0) {
                std::cout << " first=(" << kps[0].x << "," << kps[0].y
                          << "), score=" << kps[0].score;
            }
            std::cout << "\n";
        }
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    for (auto& cam : cams) {
        cam.release();
    }
    host.release();

    return 0;
}
