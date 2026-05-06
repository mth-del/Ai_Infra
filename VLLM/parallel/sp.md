# 序列并行 SP 优化记录

## 1. 优化背景

当前 `sp.py` 演示的是大模型推理/训练中的 Sequence Parallel，简称 SP。

在 Transformer 中，输入通常是：

```text
shape = [batch_size, seq_len, d_model]
```

其中：

```text
batch_size：一次处理多少条样本
seq_len：序列长度，也就是 token 数量
d_model：每个 token 的 hidden 维度
```

当序列很长时，激活值会占用大量显存。

例如：

```text
batch_size = 1
seq_len = 2048
d_model = 512
```

如果继续扩展到更长上下文，比如 8K、32K、128K，单卡保存完整序列的中间激活会越来越吃力。

序列并行的核心思路是：

```text
把 seq_len 维度切开，让不同设备处理不同 token 片段。
```

形象理解：

```text
原来一本长书由一个人从头读到尾；
现在把书按页数切成几段，多个人分别读自己的部分；
最后把每个人的结果按原顺序拼回去。
```

当前代码用多线程模拟多设备，用一个简单的前馈网络块 `SimpleFeedForwardBlock` 模拟 Transformer 中可以按 token 独立处理的部分。

## 2. 优化思路

### 2.1 构造可切分的模型模块

代码中定义了一个简单的 FFN Block：

```text
Linear(d_model -> hidden_dim)
ReLU
Linear(hidden_dim -> d_model)
Residual Add
LayerNorm
```

对应代码中的 `SimpleFeedForwardBlock`。

这个模块适合用来演示序列并行，因为 FFN 对每个 token 的计算相对独立：

```text
每个 token 都经过同一组 FFN 权重
不同 token 之间没有 attention 那样的强依赖
```

所以可以把序列切成多个 token 片段分别计算。

### 2.2 串行基线

串行版本直接把完整输入送入模型：

```text
output = model(input_data)
```

输入形状是：

```text
[1, seq_len, d_model]
```

这种方式最简单，但所有 token 都由一个模型副本一次性处理。

### 2.3 按序列维度切分

序列并行版本先计算每个片段大小：

```text
chunk_size = seq_len // num_chunks
```

然后沿着 `seq_len` 维度切分输入：

```text
chunk = input_data[:, start_idx:end_idx, :]
```

例如：

```text
seq_len = 2048
num_chunks = 2
```

会切成：

```text
片段1: token [0:1024]
片段2: token [1024:2048]
```

每个片段的形状大致是：

```text
[1, 1024, 512]
```

### 2.4 多个模型副本并行处理

代码中为每个序列片段创建一个模型副本：

```text
model_copy = copy.deepcopy(self.model)
```

这样可以保证每个线程使用相同权重。

然后每个线程处理一个 token 片段：

```text
chunk_output = model(input_chunk)
```

形象理解：

```text
每个模型副本负责一段 token。
它们用相同的规则处理不同的 token 区间。
```

### 2.5 合并结果

所有线程完成后，把每个片段的输出按序列维度拼接：

```text
parallel_output = torch.cat(output_chunks, dim=1)
```

因为输入是按 token 顺序切分的，所以输出也要按 token 顺序拼回去。

最终得到：

```text
parallel_output shape = serial_output shape
```

## 3. 优化结果

本次实验配置：

```text
seq_len = 2048
d_model = 512
num_chunks = 2
```

终端输出显示：

```text
工作线程1: 处理 1024 个 token
工作线程2: 处理 1024 个 token
```

数值验证结果：

```text
串行输出范围: [-4.4356, 4.6646]
并行输出范围: [-4.4356, 4.6646]
最大绝对差异: 4.768372e-07
最大相对差异: 3.225070e-04
序列并行计算结果与串行计算结果一致
```

说明当前序列切分、并行计算、结果合并的逻辑是正确的。

性能结果：

```text
串行处理耗时: 0.0223 秒
序列并行 2 线程处理耗时: 0.0366 秒
```

当前实验中，序列并行版本比串行版本更慢。

这不是 SP 思想错误，而是因为当前实现只是 CPU 上的 Python 线程模拟。

主要原因：

```text
1. PyTorch 的 Linear/LayerNorm 底层已经有高性能 CPU 实现。
2. Python 线程引入了创建、调度、join 等额外开销。
3. 两个线程共享同一套 CPU cache 和内存带宽，不是真正多设备并行。
4. 当前计算规模不够大，线程开销超过了并行收益。
5. 真实 SP 通常依赖多 GPU 和高效通信，而不是单机 CPU 线程。
```

当前实验的价值是：

```text
验证 SP 的核心正确性：按 seq_len 切分，每个分片独立处理，最后按序列维度拼接，可以得到和串行版本一致的结果。
```

## 4. 进一步优化

### 4.1 扩展到真实多 GPU

当前实现使用 Python 线程模拟多设备。

下一步可以改成真实多 GPU：

```text
GPU 0 处理 token [0:1024]
GPU 1 处理 token [1024:2048]
最后通过通信收集并拼接结果
```

这样才能体现 SP 在显存和吞吐上的价值。

### 4.2 引入通信算子

真实 Sequence Parallel 不只是简单 `cat`。

在 Transformer 中，不同模块可能需要：

```text
AllGather
ReduceScatter
AllReduce
```

后续可以模拟这些通信过程，记录通信开销。

### 4.3 结合 Attention 场景

当前例子主要模拟 FFN 这类逐 token 操作。

Attention 更复杂，因为每个 token 可能要看其他 token：

```text
QK^T 需要跨 token 交互
Softmax 需要完整 attention score
KV Cache 需要按序列组织
```

后续可以区分：

```text
FFN / LayerNorm：更适合按序列切分
Attention：需要额外通信或更复杂的并行策略
```

### 4.4 统计更多指标

当前只记录了串行耗时、并行耗时和数值差异。

后续可以增加：

```text
不同 seq_len 下的耗时
不同 num_chunks 下的耗时
切分和合并开销
线程/设备利用率
显存占用变化
通信时间占比
```

### 4.5 和 TP / DP 联合理解

SP 通常不会孤立使用，而是和 TP、DP 组合：

```text
DP：不同模型副本处理不同请求或 batch
TP：按 hidden/channel 维度切权重和计算
SP：按 sequence/token 维度切激活
```

在大模型训练和长上下文推理中，SP 的核心价值是降低单卡激活显存压力。

## 5. 面试表达

可以这样总结：

```text
我用 PyTorch 实现了一个 Sequence Parallel 模拟实验，把输入按 seq_len 维度切成多个 token 片段，每个线程用相同模型副本处理一个片段，最后按序列维度拼接输出。实验验证了并行输出和串行输出数值一致，最大绝对误差在 1e-6 量级。

不过在 CPU + Python 线程环境下，并行版本反而更慢，因为 PyTorch baseline 已经高度优化，而线程调度、数据切分和结果合并带来了额外开销。这个实验主要验证 SP 的切分正确性；真实 SP 的收益主要体现在多 GPU 场景，通过按 sequence 维度切分激活，降低单卡显存占用，并配合 NCCL 通信完成同步。
```
