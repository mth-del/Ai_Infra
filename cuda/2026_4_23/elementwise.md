int idx = blockIdx.x * blockDim.x + threadIdx.x;
给当前线程算一个 一维全局下标 idx：

* blockIdx.x：当前是第几个 block（沿 x 维）
* blockDim.x：每个 block 有多少线程（x 维）
* threadIdx.x：在当前 block 里是第几号线程

合起来就是「整条一维网格里，我是第几个线程」。


# elementwise_add_f32x4_kernel
int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

这里每个线程负责 从 idx 开始的连续 4 个元素：idx, idx+1, idx+2, idx+3。
所以把「线程一维编号」乘 4，得到每个线程负责的 起始下标。


float4 reg_a = FLOAT4(a[idx]);
从全局内存 a[idx..idx+3] 一次读出 128 bit，放进寄存器里的 float4（x,y,z,w 四个 float）

reg_c.{x,y,z,w} = reg_a.* + reg_b.*
在寄存器里做 4 路标量加（没有用到 SIMD 指令层面的显式向量加，但数据已是向量打包形式）。

FLOAT4(c[idx]) = reg_c;
把 reg_c 一次写回 c[idx..idx+3]，又是 128 bit 写。

**也可以写成 reg_c = make_float4(...) 或用 向量化 intrinsic，但当前写法等价、可读性好；性能关键仍是 全局内存的 128-bit 合并访问 是否满足对齐与合并条件。**


#elementwise_add_f16_kernel

half *a, *b, *c：三个半精度缓冲区，逻辑长度都是 N 个 half。
idx = blockIdx.x * blockDim.x + threadIdx.x：一维网格里每个线程的全局下标，一线程负责一个 half。

__hadd(a[idx], b[idx])：CUDA 提供的 单元素 FP16 加法（对 half），结果写回 c[idx]。


# elementwise_add_f16x2_kernel
* idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x)
线程编号 tid 映射到元素起始下标 idx，本线程负责 idx 和 idx+1 两个 half。

* HALF2(a[idx]) / HALF2(b[idx])
从 a[idx]、b[idx] 各读出一个 half2（两个 half 打包），相当于一次更宽的访存。

* reg_c.x/y = __hadd(...)
对 half2 的两个分量分别做 FP16 加法（等价于一条 __hadd2(reg_a, reg_b)，写法不同而已）。


