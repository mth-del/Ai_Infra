# SGEMM 算子原理与 CUDA 优化记录

## 1. GEMM 是什么

GEMM（General Matrix Multiplication，通用矩阵乘法）是深度学习里计算量最大、最核心的算子。

全连接层、Attention 的 QK^T 和 AV、卷积展开后的计算，本质上都是 GEMM。

一句话定义：

```text
C = α × A × B + β × C

A: [M, K]
B: [K, N]
C: [M, N]
α, β: 标量系数（通常 α=1, β=0）
```

SGEMM 中的 S 指 Single precision，即 float32 版本。

---

## 2. 直觉理解

形象类比：

```text
A 是"M 个学生"的成绩单，每人有 K 门课的分数。
B 是"K 门课"对 N 个岗位的权重矩阵。
C[i][j] = 第 i 个学生在第 j 个岗位上的综合得分。

每个输出元素 C[i][j] = A 的第 i 行 · B 的第 j 列（内积）。
```

计算量：

```text
输出 M × N 个元素，每个元素需要 K 次乘加
总计算量 = 2 × M × K × N（FLOPs，乘 2 是因为乘和加各算一次）

举例：M=K=N=1024 时，总计算量 ≈ 21 亿次浮点运算
```

---

## 3. 为什么 GEMM 难以优化

朴素实现（三层 for 循环）：

```text
for i in M:
    for j in N:
        for k in K:
            C[i][j] += A[i][k] × B[k][j]
```

性能极差，原因：

```text
1. 每次计算 C[i][j] 都要从内存读 A 的一行（K 个数）和 B 的一列（K 个数）
2. A 和 B 的数据被反复读取，大量重复 global memory 访问
3. 算术强度极低：读 2K 个数只做 2K 次浮点，AI ≈ 1 FLOP/byte
4. GPU 的计算单元大量空转，等数据
```

GPU 的理论峰值算力通常在 100~300 FLOP/byte（取决于 L1/L2 带宽），
朴素实现完全无法利用这个优势。

---

## 4. 核心优化思路：Tiling（分块）

分块的本质是：**把大矩阵拆成小块，先把小块搬进 shared memory，在 shared memory 里反复用**。

```text
原来：每次从 global memory 读，用一次就扔掉
分块：一次从 global memory 读一个 TILE 进 shared memory
      这个 TILE 的数据可以被同一 block 里所有线程反复复用
      global memory 读取次数降低 TILE_SIZE 倍
```

示意图：

```text
A                    B
┌──┬──┬──┬──┐       ┌──┬──┬──┬──┐
│  │  │  │  │       │  │  │  │  │
├──┼──┼──┼──┤   ×   ├──┼──┼──┼──┤    =    C
│  │  │  │  │       │  │  │  │  │        ┌──┬──┬──┬──┐
└──┴──┴──┴──┘       └──┴──┴──┴──┘        │  │  │  │  │
   ↑  ↑  ↑                               └──┴──┴──┴──┘
   按 K 方向切成多个 TILE
   每次搬一片 A-TILE 和 B-TILE 到 shared memory
   block 内所有线程共享这片数据，做完这片再搬下一片
```

---

## 5. 关键参数与 CUDA 映射

```text
thread  → 负责计算 C 的 1 个或多个输出元素
block   → 负责计算 C 的一个 TILE（例如 16×16 或 32×32）
shared  → 缓存当前 TILE 对应的 A_tile 和 B_tile

典型配置（TILE_SIZE=16）：
  block = (16, 16) = 256 个线程
  每个 block 计算 C 的 16×16 = 256 个元素
  shared memory 用量 = 2 × 16 × 16 × 4 bytes = 2KB
```

访存次数对比：

```text
朴素版：每个元素读 K 次 A 数据 + K 次 B 数据
        总 global read = M × N × 2K 次

分块版：每个 TILE 只从 global 读一次
        总 global read = M × N × 2K / TILE_SIZE 次
        → 访存减少 TILE_SIZE 倍（16x 或 32x）
```

---

## 6. 进阶优化方向

```text
1. 向量化读写（float4）
   一条指令读 4 个 float，内存带宽利用率提升 4x

2. double buffering（双缓冲）
   用 2 块 shared memory 交替加载，隐藏 global 读取延迟
   计算和下一块数据加载同时进行

3. Thread Coarsening（线程粗化）
   每个线程计算多个输出元素（如 4×4 = 16 个）
   寄存器复用更充分，减少 shared memory 访问次数

4. Warp-level 优化
   利用 warp shuffle 在寄存器之间直接交换数据，跳过 shared memory

5. Tensor Core（硬件加速）
   Volta 及以上 GPU 支持 WMMA（Warp Matrix Multiply Accumulate）
   一条指令完成 16×16×16 的矩阵乘，比普通 FMA 快约 8 倍
   cuBLAS 的高性能本质上就是用 Tensor Core 实现的
```

---

## 7. 面试表达

```text
GEMM 的优化核心是提升算术强度（FLOP/byte）。

朴素实现每个数据只用一次就丢弃，算术强度约 1 FLOP/byte，
远低于 GPU 的 memory-to-compute 比值，所以 GPU 大量空转等数据。

分块（Tiling）把数据先搬进 shared memory，让同一块数据被
同 block 的多个线程反复复用，global 访存次数降低 TILE_SIZE 倍，
算术强度线性提升；配合向量化读写和 double buffering，
最终可以让 GPU 的算力利用率接近 cuBLAS 水平（80%+ 峰值算力）。
```
