// ba_jacobian_compare.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>
#include <cstring>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

struct Obs {
  // 简化：假设相机坐标下点已知 (X,Y,Z)，只算投影雅可比
  float X, Y, Z;
  float u_meas, v_meas;
};

struct Out {
  float ru, rv;        // residual
  float J[2][3];       // d(u,v)/d(X,Y,Z) 简化雅可比
};

// ---------------- CPU baseline ----------------
void run_cpu_baseline(const std::vector<Obs>& obs, std::vector<Out>& out,
                      float fx, float fy, float cx, float cy) {
  size_t N = obs.size();
  for (size_t i = 0; i < N; ++i) {
    const auto& p = obs[i];
    float invZ = 1.0f / p.Z;
    float x = p.X * invZ;
    float y = p.Y * invZ;
    float u = fx * x + cx;
    float v = fy * y + cy;

    out[i].ru = u - p.u_meas;
    out[i].rv = v - p.v_meas;

    // Jacobian of (u,v) wrt (X,Y,Z)
    out[i].J[0][0] = fx * invZ;
    out[i].J[0][1] = 0.0f;
    out[i].J[0][2] = -fx * p.X * invZ * invZ;

    out[i].J[1][0] = 0.0f;
    out[i].J[1][1] = fy * invZ;
    out[i].J[1][2] = -fy * p.Y * invZ * invZ;
  }
}

// ---------------- CUDA kernel ----------------
__global__ void jacobian_kernel(const Obs* obs, Out* out, int N,
                                float fx, float fy, float cx, float cy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  Obs p = obs[i];
  float invZ = 1.0f / p.Z;
  float x = p.X * invZ;
  float y = p.Y * invZ;
  float u = fx * x + cx;
  float v = fy * y + cy;

  Out o;
  o.ru = u - p.u_meas;
  o.rv = v - p.v_meas;

  o.J[0][0] = fx * invZ;
  o.J[0][1] = 0.0f;
  o.J[0][2] = -fx * p.X * invZ * invZ;

  o.J[1][0] = 0.0f;
  o.J[1][1] = fy * invZ;
  o.J[1][2] = -fy * p.Y * invZ * invZ;

  out[i] = o;
}

// ---------------- CUDA single stream ----------------
void run_cuda_single_stream(const std::vector<Obs>& h_obs, std::vector<Out>& h_out,
                            float fx, float fy, float cx, float cy) {
  int N = (int)h_obs.size();
  // Reuse device buffers across calls to avoid malloc/free overhead.
  static Obs* d_obs = nullptr;
  static Out* d_out = nullptr;
  static int capacity = 0;
  if (N > capacity) {
    if (d_obs) CUDA_CHECK(cudaFree(d_obs));
    if (d_out) CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaMalloc(&d_obs, N * sizeof(Obs)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Out)));
    capacity = N;
  }

  CUDA_CHECK(cudaMemcpy(d_obs, h_obs.data(), N * sizeof(Obs), cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (N + block - 1) / block;
  jacobian_kernel<<<grid, block>>>(d_obs, d_out, N, fx, fy, cx, cy);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Out), cudaMemcpyDeviceToHost));
}

// ---------------- CUDA multi-stream pipeline ----------------
void run_cuda_multi_stream(const std::vector<Obs>& h_obs, std::vector<Out>& h_out,
                           float fx, float fy, float cx, float cy,
                           int n_stream = 4, int chunk = 1 << 20) {
  int N = (int)h_obs.size();
  // Reuse streams and buffers across calls to avoid repeated setup overhead.
  static int cached_streams = 0;
  static int cached_chunk = 0;
  static int pinned_capacity = 0;
  static std::vector<cudaStream_t> streams;
  static std::vector<Obs*> d_obs;
  static std::vector<Out*> d_out;
  static Obs* h_obs_pin = nullptr;
  static Out* h_out_pin = nullptr;

  bool need_reinit = (cached_streams != n_stream) || (cached_chunk != chunk);
  if (need_reinit) {
    if (!streams.empty()) {
      for (int i = 0; i < cached_streams; ++i) {
        if (d_obs[i]) CUDA_CHECK(cudaFree(d_obs[i]));
        if (d_out[i]) CUDA_CHECK(cudaFree(d_out[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
      }
      streams.clear();
      d_obs.clear();
      d_out.clear();
    }
    streams.resize(n_stream);
    d_obs.assign(n_stream, nullptr);
    d_out.assign(n_stream, nullptr);
    for (int i = 0; i < n_stream; ++i) {
      CUDA_CHECK(cudaStreamCreate(&streams[i]));
      CUDA_CHECK(cudaMalloc(&d_obs[i], chunk * sizeof(Obs)));
      CUDA_CHECK(cudaMalloc(&d_out[i], chunk * sizeof(Out)));
    }
    cached_streams = n_stream;
    cached_chunk = chunk;
  }

  if (N > pinned_capacity) {
    if (h_obs_pin) CUDA_CHECK(cudaFreeHost(h_obs_pin));
    if (h_out_pin) CUDA_CHECK(cudaFreeHost(h_out_pin));
    CUDA_CHECK(cudaHostAlloc(&h_obs_pin, N * sizeof(Obs), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out_pin, N * sizeof(Out), cudaHostAllocDefault));
    pinned_capacity = N;
  }
  std::memcpy(h_obs_pin, h_obs.data(), N * sizeof(Obs));

  int block = 256;
  int offset = 0;
  int turn = 0;

  while (offset < N) {
    int sid = turn % n_stream;
    int cur = std::min(chunk, N - offset);

    CUDA_CHECK(cudaMemcpyAsync(d_obs[sid], h_obs_pin + offset,
                               cur * sizeof(Obs), cudaMemcpyHostToDevice, streams[sid]));

    int grid = (cur + block - 1) / block;
    jacobian_kernel<<<grid, block, 0, streams[sid]>>>(d_obs[sid], d_out[sid], cur, fx, fy, cx, cy);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_out_pin + offset, d_out[sid],
                               cur * sizeof(Out), cudaMemcpyDeviceToHost, streams[sid]));

    offset += cur;
    ++turn;
  }

  for (auto& s : streams) CUDA_CHECK(cudaStreamSynchronize(s));
  std::memcpy(h_out.data(), h_out_pin, N * sizeof(Out));
}

double ms_now() {
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
  // 模拟 BA 中大量观测（可调大）
  int N = 8 * 1024 * 1024; // 8M observations
  std::vector<Obs> obs(N);
  std::vector<Out> out_cpu(N), out_gpu1(N), out_gpu4(N);

  float fx = 800, fy = 800, cx = 640, cy = 360;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dxy(-5.0f, 5.0f);
  std::uniform_real_distribution<float> dz(1.0f, 20.0f);
  std::normal_distribution<float> noise(0.0f, 1.0f);

  for (int i = 0; i < N; ++i) {
    obs[i].X = dxy(rng);
    obs[i].Y = dxy(rng);
    obs[i].Z = dz(rng);
    float u = fx * (obs[i].X / obs[i].Z) + cx;
    float v = fy * (obs[i].Y / obs[i].Z) + cy;
    obs[i].u_meas = u + noise(rng);
    obs[i].v_meas = v + noise(rng);
  }

  // Warmup to avoid first-run overhead.
  for (int i = 0; i < 3; ++i) {
    run_cpu_baseline(obs, out_cpu, fx, fy, cx, cy);
    run_cuda_single_stream(obs, out_gpu1, fx, fy, cx, cy);
    run_cuda_multi_stream(obs, out_gpu4, fx, fy, cx, cy, 4, 1 << 20);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  const int repeat = 10;
  double t0 = ms_now();
  for (int i = 0; i < repeat; ++i) {
    run_cpu_baseline(obs, out_cpu, fx, fy, cx, cy);
  }
  double t1 = ms_now();

  double t2 = ms_now();
  for (int i = 0; i < repeat; ++i) {
    run_cuda_single_stream(obs, out_gpu1, fx, fy, cx, cy);
  }
  double t3 = ms_now();

  double t4 = ms_now();
  for (int i = 0; i < repeat; ++i) {
    run_cuda_multi_stream(obs, out_gpu4, fx, fy, cx, cy, 4, 1 << 20);
  }
  double t5 = ms_now();

  // Kernel-only timing (exclude malloc/copies).
  Obs* d_obs = nullptr;
  Out* d_out = nullptr;
  int Nint = static_cast<int>(obs.size());
  int block = 256;
  int grid = (Nint + block - 1) / block;
  CUDA_CHECK(cudaMalloc(&d_obs, Nint * sizeof(Obs)));
  CUDA_CHECK(cudaMalloc(&d_out, Nint * sizeof(Out)));
  CUDA_CHECK(cudaMemcpy(d_obs, obs.data(), Nint * sizeof(Obs), cudaMemcpyHostToDevice));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  for (int i = 0; i < 10; ++i) {
    jacobian_kernel<<<grid, block>>>(d_obs, d_out, Nint, fx, fy, cx, cy);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_start));
  for (int i = 0; i < 100; ++i) {
    jacobian_kernel<<<grid, block>>>(d_obs, d_out, Nint, fx, fy, cx, cy);
  }
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float kernel_total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_total_ms, ev_start, ev_stop));
  float kernel_avg_ms = kernel_total_ms / 100.0f;
  CUDA_CHECK(cudaMemcpy(out_gpu1.data(), d_out, Nint * sizeof(Out), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_obs));
  CUDA_CHECK(cudaFree(d_out));

  // 简单数值检查
  double err = 0.0;
  for (int i = 0; i < 1000; ++i) {
    err += std::abs(out_cpu[i].ru - out_gpu4[i].ru) + std::abs(out_cpu[i].rv - out_gpu4[i].rv);
  }

  double cpu_ms = (t1 - t0) / repeat;
  double gpu1_ms = (t3 - t2) / repeat;
  double gpu4_ms = (t5 - t4) / repeat;

  std::cout << "CPU baseline(avg) : " << cpu_ms << " ms\n";
  std::cout << "CUDA 1 stream(avg): " << gpu1_ms << " ms, speedup " << (cpu_ms / gpu1_ms) << "x\n";
  std::cout << "CUDA 4 streams(avg): " << gpu4_ms << " ms, speedup " << (cpu_ms / gpu4_ms) << "x\n";
  std::cout << "CUDA kernel-only(avg): " << kernel_avg_ms << " ms, speedup " << (cpu_ms / kernel_avg_ms) << "x\n";
  std::cout << "Check err (1k sum): " << err << "\n";
  return 0;
}