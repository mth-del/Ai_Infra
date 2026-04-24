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

* HALF2(c[idx]) = reg_c
把结果一次写回 c[idx] 起的两个 half。

# elementwise_add_f16x8_kernel
每个线程一次处理 连续 8 个 half，但实现上拆成 4 次 half2 读、4 次 half2 加、最多 4 次按条件 half2 写。

2. 为什么用 4 个 half2，而不是一次读 8 个 half
half2 = 64 bit：一次读 2 个 half。
4 组偏移 0,2,4,6 拼起来正好是 8 个连续 half。
相比「8 次标量 load」，指令和事务更少；相比「一条 128-bit 指令读 8 个 half」，这里是 显式拆成 4×half2，仍属于向量化访存思路

3. 写回为什么分四个 if

N 不一定是 8 的倍数 时，最后一个线程可能只有 部分 half2 有效
按 idx+0 / idx+2 / idx+4 / idx+6 分别判断，避免对 完全超出 N 的 half2 做 store（写侧做了 粗粒度尾部保护）

4. 读侧要注意的一点（面试说清楚很加分）
上面的 HALF2(a[idx+2]) … HALF2(a[idx+6]) 在 kernel 开头是 无条件执行的。
若 idx < N 但 idx+6 或 idx+7 已 ≥ N（最后一段不足 8 个元素），仍可能对 a/b 多读越界，仅靠后面的 store 判断救不了读。

因此这个 kernel 的隐含前提是：N 是 8 的倍数，或调度保证不会出现「idx 合法但上半段 half2 已越界」的线程；通用库会在 读之前也按边界拆分 或先算 valid mask。

# elementwise_add_f16x8_pack_kernel

half pack_a[8], pack_b[8], pack_c[8];
注释里写的 .local 是编译器常见落点之一：这类 地址可被取的局部数组，往往进 local memory（慢于纯寄存器），但这里主要目的是：有一块连续 128 字节对齐布局的缓冲区，方便下面用 float4 做 宽向量访存。



2. LDST128BITS 在干什么（结合文件宏）

宏大意是把 &pack_a[0] 当成 float4*，一次读写 128 bit：

也就是：全局内存 → 线程局部打包区，尽量 一条宽 load 完成 8 个 FP16 的读入。


#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
&(value)：取 value 左值在内存里的起始地址（value 必须是可取址的东西，例如 pack_a[0]、a[idx] 这种对象/元素）。
reinterpret_cast<float4 *>(...)：不改位模式，只把指针类型改成「指向一个 float4」。
[0]：解引用成 float4，在赋值语句里就可以当 左值或右值 用一整块 128 bit。


if ((idx + 7) < N) {
  LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
} else {
  for (int i = 0; idx + i < N; i++) {
    c[idx + i] = __hadd(a[idx + i], b[idx + i]);
  }
}

idx+7 < N：说明 idx..idx+7 这 8 个元素都合法，可以 一次 128-bit 写回 c[idx]，和读对称。
否则：这 8 个里 至少最后一个越界或整块不全是「满 8」（含 idx >= N 的线程），走 for 标量循环：只对 idx+i < N 的位置逐个 __hadd 写 c（注意 else 里又从 a[idx+i]、b[idx+i] 读了一遍，没复用 pack_*）。

















**知识点**
1. 向量类型（built-in vector types）
类型	含义	典型位宽
float2	2×float	64 bit
float3	3×float（注意对齐/布局，kernel 里不如 2/4 常用）	96 bit
float4	4×float	128 bit（最常用，和一次 L1/L2 线宽、合并访存很好对齐）
int2 / int4	2/4 个 int 打包	64 / 128 bit
uint2 / uint4	无符号整型打包	同上
char2 / short2 / longlong2 等	其他标量的 2/4 打包	按名类推

成员一般叫 .x .y .z .w（float3 有 .z，float4 四个都有）。

用途：reinterpret_cast<float4*>(ptr) 一次读 128 bit，等价于连续 4 个 float 的向量化 load/store（需满足 对齐：通常起始地址 16 字节对齐 更安全）。

2. 半精度与打包
类型 / API	含义
half	16-bit FP（IEEE binary16），CUDA device 常用
half2	2×half 打包，64 bit，类似 float2 的角色
__half_raw 等	更底层位型（少在业务 kernel 里手写）
__hadd / __hadd2	单 half / 单 half2 的加法 intrinsic
__hfma2 等	半精度 FMA 类 intrinsics

和 float4 的关系：8 个 half = 128 bit = 一个 float4 的位宽，所以常用 float4 当「搬运 8×FP16 的容器」（你代码里的 LDST128BITS 就是这种思路）。



3. BF16 / FP8（扩展精度族）
类型	含义
__nv_bfloat16	BF16 标量（8 exp + 7 frac，与 FP32 指数对齐）
__nv_bfloat162	2×BF16 打包
__nv_fp8_*（如 e4m3 / e5m2）	FP8 系列（架构/头文件支持有关，你 .cu 里 #include <cuda_fp8.h> 即为此）
面试一句：BF16 动态范围大、FP8 带宽/吞吐高，但都要注意硬件代际与数值范围。

FP32 向量化：优先想 float4（128 bit）。
FP16 向量化：half2（64 bit） 最常见；要一次 8 个 half 可用 两条 half2 或一条 float4 搬运。
类型名是“内存视图”：同一地址可以 half* 标量访问，也可以 half2* / float4* 宽访问，对齐与越界要自己保证。