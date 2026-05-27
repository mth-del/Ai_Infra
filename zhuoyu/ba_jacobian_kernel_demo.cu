// ba_jacobian_kernel_demo.cu
// Scenario: Bundle Adjustment Jacobian construction benchmark.
//
// One observation means one 3D landmark observed by one camera frame.
// For every observation, this demo computes:
//   residual[2]      = projected_uv - observed_uv
//   J_pose[2 x 6]    = d residual / d se3_camera
//   J_point[2 x 3]   = d residual / d point_camera
//
// To isolate "Jacobian construction" kernel cost from a huge sparse-matrix
// writeback, the output stores residuals plus two Jacobian checksums. The
// complete 2x6 and 2x3 blocks are computed in registers in both CPU/GPU paths.
//
// Build:
//   nvcc -O3 -arch=sm_86 ba_jacobian_kernel_demo.cu -o ba_jacobian_demo
//
// Run:
//   ./ba_jacobian_demo              # default: 8M observations
//   ./ba_jacobian_demo 8000000

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(x) do {                                            \
  cudaError_t err = (x);                                             \
  if (err != cudaSuccess) {                                           \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1);                                                     \
  }                                                                   \
} while (0)

// （x,y,z）：3D点在当前相机坐标系的位置
// （u,v）:观测到的2D像素点
// （fx,fy,cx,cy）:相机内参
struct Observation {
    float x;
    float y;
    float z;
    float u;
    float v;
    float fx;
    float fy;
    float cx;
    float cy;
};

struct BAJacobianOut {
    float residual[2];
    float pose_sum;    // checksum of row-major 2 x 6 pose Jacobian
    float point_sum;   // checksum of row-major 2 x 3 point Jacobian
};

double ms_now() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

inline void compute_one_cpu(const Observation& obs, BAJacobianOut& out) {
    const float x = obs.x;
    const float y = obs.y;
    const float z = obs.z;
    const float inv_z = 1.0f / z;
    const float inv_z2 = inv_z * inv_z;

    // 核心投影公式
    const float u_hat = obs.fx * x * inv_z + obs.cx;
    const float v_hat = obs.fy * y * inv_z + obs.cy;
    out.residual[0] = u_hat - obs.u;
    out.residual[1] = v_hat - obs.v;

    // 计算投影对3D坐标的导数[2,3]
    const float du_dx = obs.fx * inv_z;
    const float du_dy = 0.0f;
    const float du_dz = -obs.fx * x * inv_z2;
    const float dv_dx = 0.0f;
    const float dv_dy = obs.fy * inv_z;
    const float dv_dz = -obs.fy * y * inv_z2;

    // Left perturbation: dXc / d(delta) = [I, -skew(Xc)].
    const float jp0  = du_dx;
    const float jp1  = du_dy;
    const float jp2  = du_dz;
    const float jp3  = -du_dy * z + du_dz * y;
    const float jp4  =  du_dx * z - du_dz * x;
    const float jp5  = -du_dx * y + du_dy * x;
    const float jp6  = dv_dx;
    const float jp7  = dv_dy;
    const float jp8  = dv_dz;
    const float jp9  = -dv_dy * z + dv_dz * y;
    const float jp10 =  dv_dx * z - dv_dz * x;
    const float jp11 = -dv_dx * y + dv_dy * x;

    out.pose_sum = jp0 + jp1 + jp2 + jp3 + jp4 + jp5
                 + jp6 + jp7 + jp8 + jp9 + jp10 + jp11;
    out.point_sum = du_dx + du_dy + du_dz + dv_dx + dv_dy + dv_dz;
}

void run_cpu_baseline(const Observation* obs, BAJacobianOut* out, int n) {
    for (int i = 0; i < n; ++i) {
        compute_one_cpu(obs[i], out[i]);
    }
}

// 核心策略就是每一个thread处理一个observation
// 没有跨进程数据复用，所以没有使用shared memory
__global__ void ba_jacobian_kernel(const Observation* obs, BAJacobianOut* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Observation o = obs[i];
    BAJacobianOut r;

    const float x = o.x;
    const float y = o.y;
    const float z = o.z;
    const float inv_z = 1.0f / z;
    const float inv_z2 = inv_z * inv_z;

    const float u_hat = o.fx * x * inv_z + o.cx;
    const float v_hat = o.fy * y * inv_z + o.cy;
    r.residual[0] = u_hat - o.u;
    r.residual[1] = v_hat - o.v;

    const float du_dx = o.fx * inv_z;
    const float du_dy = 0.0f;
    const float du_dz = -o.fx * x * inv_z2;
    const float dv_dx = 0.0f;
    const float dv_dy = o.fy * inv_z;
    const float dv_dz = -o.fy * y * inv_z2;

    const float jp0  = du_dx;
    const float jp1  = du_dy;
    const float jp2  = du_dz;
    const float jp3  = -du_dy * z + du_dz * y;
    const float jp4  =  du_dx * z - du_dz * x;
    const float jp5  = -du_dx * y + du_dy * x;
    const float jp6  = dv_dx;
    const float jp7  = dv_dy;
    const float jp8  = dv_dz;
    const float jp9  = -dv_dy * z + dv_dz * y;
    const float jp10 =  dv_dx * z - dv_dz * x;
    const float jp11 = -dv_dx * y + dv_dy * x;

    r.pose_sum = jp0 + jp1 + jp2 + jp3 + jp4 + jp5
               + jp6 + jp7 + jp8 + jp9 + jp10 + jp11;
    r.point_sum = du_dx + du_dy + du_dz + dv_dx + dv_dy + dv_dz;

    out[i] = r;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (ptr_ != nullptr) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reserve(size_t count) {
        if (count <= capacity_) return;
        if (ptr_ != nullptr) CUDA_CHECK(cudaFree(ptr_));
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        capacity_ = count;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

private:
    T* ptr_ = nullptr;
    size_t capacity_ = 0;
};

void init_observations(std::vector<Observation>& obs) {
    // 创建一个固定种子生成器
    std::mt19937 rng(42);
    // 生成3D点
    std::uniform_real_distribution<float> xy_dist(-3.0f, 3.0f);
    std::uniform_real_distribution<float> z_dist(4.0f, 30.0f);
    // 噪声
    std::uniform_real_distribution<float> noise_dist(-0.5f, 0.5f);

    // 相机参数
    constexpr float fx = 720.0f;
    constexpr float fy = 720.0f;
    constexpr float cx = 640.0f;
    constexpr float cy = 360.0f;

    for (auto& o : obs) {
        o.x = xy_dist(rng);
        o.y = xy_dist(rng);
        o.z = z_dist(rng);
        o.fx = fx;
        o.fy = fy;
        o.cx = cx;
        o.cy = cy;
        o.u = fx * o.x / o.z + cx + noise_dist(rng);
        o.v = fy * o.y / o.z + cy + noise_dist(rng);
    }
}

double checksum_first(const std::vector<BAJacobianOut>& out, int n) {
    double sum = 0.0;
    int m = std::min<int>(n, out.size());
    for (int i = 0; i < m; ++i) {
        sum += std::abs(out[i].residual[0]) + std::abs(out[i].residual[1]);
        sum += std::abs(out[i].pose_sum) + std::abs(out[i].point_sum);
    }
    return sum;
}


void GPU_line(int n_obs){

}


int main(int argc, char** argv) {
    int n_obs = 8 * 1000 * 1000;
    if (argc >= 2) {
        n_obs = std::max(1, std::atoi(argv[1]));
    }

    std::cout << "=== BA Jacobian Kernel Benchmark ===\n";
    std::cout << "Observations: " << n_obs << "\n";
    std::cout << "Input bytes : " << (double)n_obs * sizeof(Observation) / 1e6 << " MB\n";
    std::cout << "Output bytes: " << (double)n_obs * sizeof(BAJacobianOut) / 1e6 << " MB\n\n";

    std::vector<Observation> h_obs(n_obs);
    std::vector<BAJacobianOut> h_out_cpu(n_obs);
    init_observations(h_obs);

    constexpr int cpu_repeat = 3;
    /**[1] CPU baseline**/
    run_cpu_baseline(h_obs.data(), h_out_cpu.data(), n_obs);
    double t0 = ms_now();
    for (int i = 0; i < cpu_repeat; ++i) {
        run_cpu_baseline(h_obs.data(), h_out_cpu.data(), n_obs);
    }
    double cpu_ms = (ms_now() - t0) / cpu_repeat;
    std::cout << "CPU serial baseline(avg): " << cpu_ms << " ms\n";

    /**[2] GPU上运行**/
    DeviceBuffer<Observation> d_obs;
    DeviceBuffer<BAJacobianOut> d_out;
    d_obs.reserve(n_obs);
    d_out.reserve(n_obs);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    constexpr int block_size = 256;
    dim3 block(block_size);
    dim3 grid((n_obs + block_size - 1) / block_size);

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_obs.data(), h_obs.data(),
                          n_obs * sizeof(Observation), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev0, ev1));

    for (int i = 0; i < 3; ++i) {
        ba_jacobian_kernel<<<grid, block>>>(d_obs.data(), d_out.data(), n_obs);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int kernel_repeat = 50;
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < kernel_repeat; ++i) {
        ba_jacobian_kernel<<<grid, block>>>(d_obs.data(), d_out.data(), n_obs);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
    kernel_ms /= kernel_repeat;

    constexpr int check_n = 4096;
    std::vector<BAJacobianOut> h_out_gpu(std::min(check_n, n_obs));
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out.data(),
                          h_out_gpu.size() * sizeof(BAJacobianOut),
                          cudaMemcpyDeviceToHost));

    double cpu_sum = checksum_first(h_out_cpu, check_n);
    double gpu_sum = checksum_first(h_out_gpu, check_n);
    double rel_err = std::abs(cpu_sum - gpu_sum) / (std::abs(cpu_sum) + 1e-9);

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_obs.data(), h_obs.data(),
                          n_obs * sizeof(Observation), cudaMemcpyHostToDevice));
    ba_jacobian_kernel<<<grid, block>>>(d_obs.data(), d_out.data(), n_obs);
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out.data(),
                          h_out_gpu.size() * sizeof(BAJacobianOut),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float end_to_end_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&end_to_end_ms, ev0, ev1));

    std::cout << "CUDA H2D input only     : " << h2d_ms << " ms\n";
    std::cout << "CUDA kernel-only(avg)  : " << kernel_ms << " ms"
              << "  speedup " << cpu_ms / kernel_ms << "x\n";
    std::cout << "CUDA end-to-end(sample D2H): " << end_to_end_ms << " ms\n";
    std::cout << "Check first " << h_out_gpu.size()
              << " outputs: cpu_sum=" << cpu_sum
              << " gpu_sum=" << gpu_sum
              << " rel_err=" << rel_err << "\n\n";

    std::cout << "Notes:\n";
    std::cout << "  * Kernel-only measures Jacobian construction after inputs are on GPU.\n";
    std::cout << "  * End-to-end includes H2D for all observations and only a small D2H sample,\n";
    std::cout << "    which highlights the H2D transfer bottleneck for this low-AI workload.\n";
    std::cout << "  * Reusing d_obs/d_out buffers avoids per-iteration cudaMalloc/cudaFree overhead.\n";

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return 0;
}
