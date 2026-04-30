# Softmax 算子原理与 CUDA 优化记录

## 1. Softmax 算子的原理

Softmax 的作用是把一组任意实数变成一组概率。

比如模型输出一行分数：

```text
[2.0, 1.0, 0.1]
```

这些数本身不是概率，因为可能大于 1，也可能为负数。Softmax 会把它们转换成：

```text
[0.66, 0.24, 0.10]
```

特点是：

```text
每个值都在 0 到 1 之间
所有值加起来等于 1
原来分数越大的位置，转换后的概率也越大
```

公式是：

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

形象理解：

```text
Softmax 像是在做“投票归一化”。
每个位置先根据自己的分数生成一个权重，分数越高权重越大；
然后所有位置一起平分总概率，最后得到一组加起来等于 1 的概率分布。
```

实际实现时通常不会直接算 `exp(x_i)`，因为如果 `x_i` 很大，指数可能溢出。

所以会先减去这一行的最大值：

```text
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

这样结果不变，但数值更稳定。

## 2. 根据 softmax.cu 的优化思路描述

当前 `softmax.cu` 中已经准备了 CUDA、half、bfloat16、fp8 相关头文件，以及 `float4`、`half2`、`bfloat162`、`int4` 等向量化读写宏。

这些宏说明优化方向主要是：

```text
减少全局内存访问次数
提高单次访存宽度
支持低精度数据类型
利用 warp/block 级并行做规约
```

### 2.1 优化背景

Softmax 在大模型里很常见，尤其是在 Attention 中。

Attention 里通常会先计算：

```text
QK^T
```

得到每个 token 对其他 token 的注意力分数，然后对这些分数做 Softmax，得到注意力权重。

Softmax 的计算流程一般包括三步：

```text
1. 找到一行里的最大值 max
2. 计算 exp(x - max)，并累加 sum
3. 每个元素除以 sum，得到最终概率
```

它的问题是：每个元素要被读写多次，而且每一行还需要做最大值规约和求和规约。

所以 Softmax 往往不是纯算力瓶颈，而是：

```text
访存瓶颈 + 规约瓶颈
```

形象理解：

```text
Softmax 不是难在“算一个 exp”，而是难在一整行数据要反复搬来搬去。
如果搬数据太慢，GPU 的计算单元就会等数据。
```

### 2.2 优化方案

#### 方案一：按行并行

Softmax 通常按最后一维做归一化，也就是一行一行处理。

可以让一个 CUDA block 或一个 warp 负责一行数据：

```text
一个 block/warp 处理一行
多个线程并行读取这一行的不同元素
```

这样每一行之间互不依赖，可以天然并行。

#### 方案二：先做 max 规约，保证数值稳定

第一步先并行求出当前行的最大值：

```text
row_max = max(x)
```

然后每个线程计算：

```text
exp(x - row_max)
```

这样可以避免指数溢出。

#### 方案三：再做 sum 规约，得到归一化分母

每个线程计算自己负责元素的 `exp(x - row_max)`，然后并行求和：

```text
row_sum = sum(exp(x - row_max))
```

最后每个元素除以 `row_sum`。

这两个规约可以用：

```text
warp shuffle
shared memory
block reduce
```

来减少线程同步和全局内存访问。

#### 方案四：向量化读写，提升访存带宽

`softmax.cu` 中定义了：

```text
FLOAT4(value)
HALF2(value)
BFLOAT2(value)
LDST128BITS(value)
```

这些宏的含义是让线程一次读写多个连续元素。

例如：

```text
float4  一次处理 4 个 float
half2   一次处理 2 个 half
bfloat162 一次处理 2 个 bfloat16
```

形象理解：

```text
普通读写像一次搬一个快递；
向量化读写像一次搬一箱快递。
路程一样，但一次搬得更多，整体效率更高。
```

#### 方案五：支持 half / bfloat16 / fp8 等低精度类型

大模型推理中常用 `FP16`、`BF16`，甚至 `FP8` 来减少显存占用和提升吞吐。

Softmax 通常可以：

```text
输入使用 half / bfloat16
中间规约使用 float
输出再转回 half / bfloat16
```

这样既能提升性能，又能尽量保证数值稳定。

### 2.3 优化结果

按照上述优化后，Softmax 的收益主要体现在：

```text
1. 向量化读写减少访存指令数量
2. warp/block 规约减少同步和中间数据写回
3. shared memory 减少重复访问 global memory
4. half/bfloat16 降低显存带宽压力
5. 减 max 的稳定实现避免 exp 溢出
```

最终效果可以总结为：

```text
Softmax 从“反复读写整行数据”的朴素实现，
优化成“每行并行处理、局部规约、向量化搬运”的 GPU 友好实现。
```

面试中可以这样说：

```text
我对 Softmax 的优化重点不是单纯优化 exp，而是减少访存和规约开销。
通过按行并行、max/sum 两阶段规约、向量化读写和低精度支持，让 kernel 更接近显存带宽上限。
```

### 2.4 适合在什么条件下使用

这种优化适合：

```text
1. 输入是二维或高维 tensor，但 Softmax 沿最后一维计算
2. 每一行长度中等，比如 128、256、512、1024、2048
3. 数据在内存中连续，方便 float4 / half2 向量化读写
4. 大模型 Attention 中对 score 做 Softmax
5. 推理场景中使用 FP16 / BF16，希望降低访存开销
```

不太适合：

```text
1. 行长度非常短，kernel launch 开销可能大于计算收益
2. 行长度非常长，需要更复杂的 block 级或多 block 规约
3. 数据不是连续布局，向量化读写收益会下降
4. 对数值精度要求极高，不能接受低精度输入输出
```

一句话总结：

```text
这个 Softmax 优化适合大模型推理中的 attention score 归一化场景，尤其是行维度固定、数据连续、使用 FP16/BF16 的 GPU 推理任务。
```
