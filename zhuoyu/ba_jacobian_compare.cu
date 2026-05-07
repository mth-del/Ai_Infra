// bev_feature_lifting.cu
// 场景：自动驾驶多相机 BEV 感知（LSS/BEVFusion 风格）
// 核心操作：把 N_CAM 路相机的图像特征 × 深度分布，投影聚合到 BEV 体素空间
// 算术强度远高于 BA Jacobian：每像素需完成 D × C 次乘加，是典型 compute-bound
//
// 编译：nvcc -O3 -arch=sm_86 ba_jacobian_compare.cu -o bev_lifting
// 运行：./bev_lifting

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

#define CUDA_CHECK(x) do {                                            \
  cudaError_t err = (x);                                             \
  if (err != cudaSuccess) {                                           \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1);                                                     \
  }                                                                   \
} while(0)

// ============================================================
// 超参数（对应 BEVFusion/LSS 典型配置）
// ============================================================
static constexpr int N_CAM = 6;      // 相机数量（前/后/左前/左后/右前/右后）
static constexpr int IMG_H = 56;     // backbone 下采样后的特征图高度（原图 448/8）
static constexpr int IMG_W = 152;    // 特征图宽度（原图 1216/8）
static constexpr int D    = 64;      // 深度 bin 数量
static constexpr int C    = 64;      // 特征通道数（简化，完整版为 256）
static constexpr int BEV_X = 128;    // BEV 体素 X 方向数量
static constexpr int BEV_Y = 128;    // BEV 体素 Y 方向数量

// 总像素数
static constexpr int TOTAL_PIX = N_CAM * IMG_H * IMG_W;

// ============================================================
// 数据结构
// ============================================================

// 每个像素对应的相机内外参预计算结果（投影射线方向 + 深度步进在 BEV 中的落点）
// 实际工程中由预处理模块离线算好
struct PixelRay {
    float bev_x_start;  // 深度 bin 0 时在 BEV 中的 x 坐标（已归一化到 [0,BEV_X)）
    float bev_y_start;  // 深度 bin 0 时在 BEV 中的 y 坐标
    float bev_x_step;   // 深度每增加 1 bin，BEV x 的变化量
    float bev_y_step;   // 深度每增加 1 bin，BEV y 的变化量
    int   valid;        // 该像素是否在 BEV 范围内
};

// ============================================================
// 工具函数
// ============================================================
double ms_now() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

// ============================================================
// CPU baseline
// 三层嵌套：像素 × 深度 bin × 特征通道
// ============================================================
void run_cpu_baseline(
    const float*     feat,      // [N_CAM, C, IMG_H, IMG_W]  图像特征
    const float*     depth,     // [N_CAM, D, IMG_H, IMG_W]  深度分布（已 softmax）
    const PixelRay*  rays,      // [N_CAM, IMG_H, IMG_W]     预计算射线
    float*           bev_feat)  // [BEV_Y, BEV_X, C]         输出 BEV 特征
{
    std::fill(bev_feat, bev_feat + BEV_Y * BEV_X * C, 0.0f);

    for (int pix = 0; pix < TOTAL_PIX; ++pix) {
        const PixelRay& ray = rays[pix];
        if (!ray.valid) continue;

        int cam = pix / (IMG_H * IMG_W);
        int hw  = pix % (IMG_H * IMG_W);

        for (int d = 0; d < D; ++d) {
            float bx = ray.bev_x_start + d * ray.bev_x_step;
            float by = ray.bev_y_start + d * ray.bev_y_step;
            int ibx = (int)bx, iby = (int)by;
            if (ibx < 0 || ibx >= BEV_X || iby < 0 || iby >= BEV_Y) continue;

            float w = depth[cam * D * IMG_H * IMG_W + d * IMG_H * IMG_W + hw];

            for (int c = 0; c < C; ++c) {
                float f = feat[cam * C * IMG_H * IMG_W + c * IMG_H * IMG_W + hw];
                bev_feat[(iby * BEV_X + ibx) * C + c] += w * f;
            }
        }
    }
}

// ============================================================
// CUDA kernel（naive 版）
// 每个线程处理一个 (pix, d) 对，对 C 个通道做 atomicAdd
// grid: (TOTAL_PIX, D), block: 1D 按 C 展开
// ============================================================
__global__ void bev_lifting_kernel(
    const float*    feat,
    const float*    depth,
    const PixelRay* rays,
    float*          bev_feat,
    int total_pix)
{
    int pix = blockIdx.x;
    int d   = blockIdx.y;
    int c   = threadIdx.x;  // block 大小 = C

    if (pix >= total_pix || c >= C) return;

    const PixelRay& ray = rays[pix];
    if (!ray.valid) return;

    float bx = ray.bev_x_start + d * ray.bev_x_step;
    float by = ray.bev_y_start + d * ray.bev_y_step;
    int ibx = (int)bx, iby = (int)by;
    if (ibx < 0 || ibx >= BEV_X || iby < 0 || iby >= BEV_Y) return;

    int cam = pix / (IMG_H * IMG_W);
    int hw  = pix % (IMG_H * IMG_W);

    float w = depth[cam * D * IMG_H * IMG_W + d * IMG_H * IMG_W + hw];
    float f = feat [cam * C * IMG_H * IMG_W + c * IMG_H * IMG_W + hw];

    atomicAdd(&bev_feat[(iby * BEV_X + ibx) * C + c], w * f);
}

// ============================================================
// CUDA kernel（shared memory 优化版）
// 每个 block 负责一个像素的所有 D 个深度 bin
// 同一像素的 C 维特征先加载到 shared memory，D 个 bin 复用
// 显著降低 global memory 读取次数：feat 读取次数从 D×C 次降到 C 次
// ============================================================
__global__ void bev_lifting_kernel_smem(
    const float*    feat,
    const float*    depth,
    const PixelRay* rays,
    float*          bev_feat,
    int total_pix)
{
    // block: (C_THREADS) 处理一个像素的所有 D bins
    // gridDim.x = total_pix
    extern __shared__ float s_feat[];  // [C] 当前像素的特征，由所有线程共享

    int pix = blockIdx.x;
    int tid = threadIdx.x;  // 0 .. C-1

    if (pix >= total_pix) return;

    const PixelRay& ray = rays[pix];

    // 协作加载当前像素特征到 shared memory
    int cam = pix / (IMG_H * IMG_W);
    int hw  = pix % (IMG_H * IMG_W);

    if (tid < C) {
        s_feat[tid] = feat[cam * C * IMG_H * IMG_W + tid * IMG_H * IMG_W + hw];
    }
    __syncthreads();

    if (!ray.valid) return;

    // 每个线程负责一个 depth bin（当 C == blockDim.x 时，每线程处理 1 个 bin）
    // 这里用 tid 同时覆盖 C 个通道，每个 bin 由整个 warp 处理
    for (int d = 0; d < D; ++d) {
        float bx = ray.bev_x_start + d * ray.bev_x_step;
        float by = ray.bev_y_start + d * ray.bev_y_step;
        int ibx = (int)bx, iby = (int)by;
        if (ibx < 0 || ibx >= BEV_X || iby < 0 || iby >= BEV_Y) continue;

        float w = depth[cam * D * IMG_H * IMG_W + d * IMG_H * IMG_W + hw];

        if (tid < C) {
            atomicAdd(&bev_feat[(iby * BEV_X + ibx) * C + tid], w * s_feat[tid]);
        }
    }
}

// ============================================================
// main
// ============================================================
int main() {
    // ---- 数据规模说明 ----
    // feat:     [N_CAM=6, C=64, H=56, W=152]   = 6×64×56×152 float ≈ 12.6M 元素
    // depth:    [N_CAM=6, D=64, H=56, W=152]   = 6×64×56×152 float ≈ 12.6M 元素
    // rays:     [N_CAM×H×W = 51072] PixelRay
    // bev_feat: [BEV_Y=128, BEV_X=128, C=64]  = 128×128×64 float ≈ 1M 元素
    // 总计算量: TOTAL_PIX × D × C = 51072 × 64 × 64 ≈ 209M MAC/帧

    std::cout << "=== BEV Feature Lifting Benchmark ===\n";
    std::cout << "Cams=" << N_CAM << " H=" << IMG_H << " W=" << IMG_W
              << " D=" << D << " C=" << C
              << " BEV=" << BEV_X << "x" << BEV_Y << "\n";
    std::cout << "Total pixels: " << TOTAL_PIX
              << "  Total MACs: " << (long long)TOTAL_PIX * D * C / 1000000 << "M\n\n";

    // ---- 初始化数据 ----
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    int feat_size  = N_CAM * C * IMG_H * IMG_W;
    int depth_size = N_CAM * D * IMG_H * IMG_W;
    int bev_size   = BEV_Y * BEV_X * C;

    std::vector<float>     h_feat(feat_size),  h_depth(depth_size);
    std::vector<PixelRay>  h_rays(TOTAL_PIX);
    std::vector<float>     h_bev_cpu(bev_size, 0.f);
    std::vector<float>     h_bev_gpu(bev_size, 0.f);
    std::vector<float>     h_bev_gpu_smem(bev_size, 0.f);

    for (auto& v : h_feat)  v = dist(rng);
    for (auto& v : h_depth) v = dist(rng);

    // 模拟预计算的射线（简化：均匀铺满 BEV，步长模拟深度推进）
    for (int pix = 0; pix < TOTAL_PIX; ++pix) {
        int hw = pix % (IMG_H * IMG_W);
        int ih = hw / IMG_W, iw = hw % IMG_W;
        h_rays[pix].bev_x_start = (float)iw / IMG_W * BEV_X;
        h_rays[pix].bev_y_start = (float)ih / IMG_H * BEV_Y;
        h_rays[pix].bev_x_step  = 0.3f;
        h_rays[pix].bev_y_step  = 0.2f;
        h_rays[pix].valid       = 1;
    }

    // ---- CPU warmup + timing ----
    run_cpu_baseline(h_feat.data(), h_depth.data(), h_rays.data(), h_bev_cpu.data());
    double t0 = ms_now();
    const int repeat = 5;
    for (int i = 0; i < repeat; ++i) {
        std::fill(h_bev_cpu.begin(), h_bev_cpu.end(), 0.f);
        run_cpu_baseline(h_feat.data(), h_depth.data(), h_rays.data(), h_bev_cpu.data());
    }
    double cpu_ms = (ms_now() - t0) / repeat;
    std::cout << "CPU baseline(avg)     : " << cpu_ms << " ms\n";

    // ---- 分配 GPU 内存 ----
    float*    d_feat  = nullptr;
    float*    d_depth = nullptr;
    PixelRay* d_rays  = nullptr;
    float*    d_bev   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_feat,  feat_size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depth, depth_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rays,  TOTAL_PIX  * sizeof(PixelRay)));
    CUDA_CHECK(cudaMalloc(&d_bev,   bev_size   * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_feat,  h_feat.data(),  feat_size  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth, h_depth.data(), depth_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays,  h_rays.data(),  TOTAL_PIX  * sizeof(PixelRay), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // ---- Naive kernel timing ----
    // grid: (TOTAL_PIX, D), block: C 个线程
    dim3 grid_naive(TOTAL_PIX, D);
    dim3 block_naive(C);

    // warmup
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemset(d_bev, 0, bev_size * sizeof(float)));
        bev_lifting_kernel<<<grid_naive, block_naive>>>(
            d_feat, d_depth, d_rays, d_bev, TOTAL_PIX);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaMemset(d_bev, 0, bev_size * sizeof(float)));
        bev_lifting_kernel<<<grid_naive, block_naive>>>(
            d_feat, d_depth, d_rays, d_bev, TOTAL_PIX);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float naive_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ev0, ev1));
    naive_ms /= 20.f;

    CUDA_CHECK(cudaMemcpy(h_bev_gpu.data(), d_bev, bev_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "CUDA naive kernel(avg): " << naive_ms << " ms"
              << "  speedup " << cpu_ms / naive_ms << "x\n";

    // ---- Shared memory kernel timing ----
    // grid: TOTAL_PIX, block: C 个线程，smem: C float
    dim3 grid_smem(TOTAL_PIX);
    dim3 block_smem(C);
    size_t smem_bytes = C * sizeof(float);

    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemset(d_bev, 0, bev_size * sizeof(float)));
        bev_lifting_kernel_smem<<<grid_smem, block_smem, smem_bytes>>>(
            d_feat, d_depth, d_rays, d_bev, TOTAL_PIX);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaMemset(d_bev, 0, bev_size * sizeof(float)));
        bev_lifting_kernel_smem<<<grid_smem, block_smem, smem_bytes>>>(
            d_feat, d_depth, d_rays, d_bev, TOTAL_PIX);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float smem_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&smem_ms, ev0, ev1));
    smem_ms /= 20.f;

    CUDA_CHECK(cudaMemcpy(h_bev_gpu_smem.data(), d_bev, bev_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "CUDA smem kernel(avg) : " << smem_ms << " ms"
              << "  speedup " << cpu_ms / smem_ms << "x\n";

    // ---- 数值一致性验证 ----
    double err_naive = 0.0, err_smem = 0.0, ref_sum = 0.0;
    int check_n = std::min(bev_size, 2000);
    for (int i = 0; i < check_n; ++i) {
        err_naive += std::abs(h_bev_cpu[i] - h_bev_gpu[i]);
        err_smem  += std::abs(h_bev_cpu[i] - h_bev_gpu_smem[i]);
        ref_sum   += std::abs(h_bev_cpu[i]);
    }
    std::cout << "\nNumerics (first " << check_n << " BEV cells):\n";
    std::cout << "  naive vs CPU : abs_err=" << err_naive
              << "  rel=" << err_naive / (ref_sum + 1e-9) << "\n";
    std::cout << "  smem  vs CPU : abs_err=" << err_smem
              << "  rel=" << err_smem  / (ref_sum + 1e-9) << "\n";

    // ---- 算术强度说明 ----
    long long total_mac = (long long)TOTAL_PIX * D * C;
    long long read_bytes_naive = (long long)TOTAL_PIX * D * (C + 1) * sizeof(float);
    long long read_bytes_smem  = (long long)TOTAL_PIX * (C + D) * sizeof(float);
    std::cout << "\nArithmetic intensity analysis:\n";
    std::cout << "  Total MACs        : " << total_mac / 1e6 << "M\n";
    std::cout << "  Naive global reads: " << read_bytes_naive / 1e6 << " MB"
              << "  AI=" << (float)total_mac * 2 / read_bytes_naive << " FLOP/byte\n";
    std::cout << "  Smem global reads : " << read_bytes_smem  / 1e6 << " MB"
              << "  AI=" << (float)total_mac * 2 / read_bytes_smem  << " FLOP/byte\n";

    // ---- 清理 ----
    CUDA_CHECK(cudaFree(d_feat));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_rays));
    CUDA_CHECK(cudaFree(d_bev));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    return 0;
}
