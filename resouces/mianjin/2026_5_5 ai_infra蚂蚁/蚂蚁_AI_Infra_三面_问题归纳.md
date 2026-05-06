# 蚂蚁 AI Infra 三面问题归纳

## 1. 面试整体方向

这一轮更偏大模型推理基础，不只是追问项目细节，而是看你是否真正理解：

```text
量化
KV Cache
Attention
推理 runtime
kernel profiling
性能瓶颈判断
```

核心考察点：

```text
你不是只会复现项目，而是能讲清楚背后的推理链路、数值问题和性能取舍。
```

## 2. 量化推理 Runtime

### 问题：你做的量化推理 runtime 具体改了什么？

面试官想确认：

```text
你是只把开源项目跑起来，还是理解了量化推理链路。
```

简洁回答：

```text
我围绕 W4A16 推理链路做了适配。权重从 FP16 压到 INT4，激活保持 FP16；推理时需要做 INT4 权重解包、scale 反量化，再参与矩阵计算。目标是在小显存消费卡上跑通 8B 模型前向，同时尽量控制精度损失。
```

形象理解：

```text
原来模型权重像一本很厚的书，FP16 每个字占 2 个字节；
W4A16 就是把书压缩成速记本，存的时候更省空间，用的时候再按比例还原出来计算。
```

### 问题：量化落在哪一部分？

```text
主要量化 Linear 层的 weight。
激活保持 FP16，所以叫 W4A16：Weight 4-bit，Activation 16-bit。
这样比 W4A4 更稳，因为激活是运行时动态变化的，分布更难控制。
```

### 问题：精度怎么评估？

```text
可以用 perplexity、下游任务准确率，或者对比 FP16 baseline 的输出误差。
LLM 常见数据集包括 WikiText、C4、MMLU、CMMLU、HumanEval 等。
核心不是只看能不能跑，而是看压缩后模型能力是否还能保持。
```

## 3. 量化基础

### 问题：W4A16、INT8、FP8 有什么区别？

```text
W4A16：权重 INT4，激活 FP16，省显存明显，精度相对稳。
INT8：权重和/或激活用 8-bit 整数，部署成熟，但需要 scale/zero point。
FP8：8-bit 浮点，有指数位，动态范围比 INT8 更自然，适合新 GPU 上训练和推理加速。
```

形象理解：

```text
INT4/INT8 像用固定刻度尺量东西，省空间但容易量不准。
FP8 像一把会自动换单位的尺子，既能表示小数，也能表示比较大的数。
```

### 问题：为什么量化会掉精度？

```text
量化本质是把连续的高精度数映射到有限几个离散值。
FP16 能表示很多细节，INT4 只有 16 个档位。
如果某层数值分布很散，或者有 outlier，量化误差就会变大。
```

## 4. KV Cache

### 问题：为什么需要 KV Cache？

```text
LLM decode 是一个 token 一个 token 生成。
如果没有 KV Cache，每生成一个新 token，都要重新计算所有历史 token 的 Key 和 Value。
KV Cache 把历史 K/V 存下来，新 token 只算自己的 Q/K/V，然后复用历史 K/V，避免重复计算。
```

形象理解：

```text
KV Cache 像聊天记录小抄。
没有它，每次回答都要重新读完整聊天记录；
有了它，只需要看新来的那句话，再翻已有小抄。
```

### 问题：KV Cache 大小怎么算？

常见公式：

```text
KV Cache ≈ batch_size × seq_len × num_layers × 2 × num_kv_heads × head_dim × dtype_bytes
```

其中 `2` 表示：

```text
Key Cache + Value Cache
```

简洁解释：

```text
KV Cache 大小主要由 batch、序列长度、层数、KV head 数、head_dim 和数据类型决定。
上下文越长、并发越高，KV Cache 越大。
```

## 5. Attention

### 问题：Attention 怎么理解？

```text
Attention 的核心是：当前 token 要决定自己应该关注历史里的哪些 token。
Q 是当前 token 的问题，K 是历史 token 的标签，V 是历史 token 的内容。
Q 和 K 做相似度，得到注意力权重，再用这个权重去加权 V。
```

形象理解：

```text
Q 像你现在的问题。
K 像每段历史内容的关键词。
V 像真正的历史内容。
Attention 就是先用问题去匹配关键词，再决定重点阅读哪些内容。
```

### 问题：Attention 计算流程是什么？

```text
Q, K, V = hidden_states 经过线性投影得到
score = Q @ K^T / sqrt(d)
prob = softmax(score)
output = prob @ V
```

一句话：

```text
先算当前 token 和历史 token 的相关性，再根据相关性加权汇总历史信息。
```

## 6. Attention 量化为什么难

### 问题：为什么 attention 比权重量化更难？

```text
权重是固定的，可以离线统计分布后量化。
attention 里的 Q/K/V 和 score 是运行时动态变化的，不同输入分布差异很大。
而且 attention 里有 softmax，softmax 对数值误差很敏感，小误差可能被放大成概率分布变化。
```

形象理解：

```text
权重量化像压缩一本固定的书，可以提前慢慢调。
Attention 量化像实时翻译现场对话，输入一直变，稍微听错一点，后面的理解可能全变。
```

更深入一点：

```text
attention 难量化主要因为动态范围大、outlier 多、softmax 对误差敏感，并且 QK^T 的误差会影响后续概率分布。
所以很多方案会优先量化权重或 KV Cache，而对 attention score / softmax 保守处理。
```

## 7. Profiling / Timing 工具

### 问题：你做 timing 工具的价值是什么？

```text
它不是为了替代 Nsight，而是为了快速判断瓶颈方向。
比如一个 kernel 是 latency bound、memory bound，还是 compute bound。
先用轻量 timing 快速定位，再决定要不要深入用 profiler 分析。
```

形象理解：

```text
Timing 工具像体温计，先快速判断哪里发烧；
Nsight 像 CT，后面再深入检查具体病灶。
```

## 8. 这轮面试核心考点

可以总结成：

```text
1. 是否真的理解量化，而不是只会跑脚本。
2. 是否理解 LLM decode 过程和 KV Cache。
3. 是否能讲清 attention 的计算和数值敏感性。
4. 是否知道性能分析要先判断瓶颈类型。
5. 能不能从项目经验抽象出通用方法论。
```

## 9. 最短复盘版

```text
这轮面试主要考大模型推理基础：量化、KV Cache、attention 和 profiling 方法论。

量化重点问 W4A16 到底量化了哪里、精度怎么测、INT8/FP8 有什么区别。
KV Cache 重点问为什么需要缓存，以及显存大小怎么算。
Attention 重点问 Q/K/V 怎么理解，为什么 softmax 和 attention 更难量化。
项目部分重点不是问你有没有做过，而是问你做这些优化背后的判断方法和取舍逻辑。
```

最核心一句话：

```text
AI Infra 三面不是只看你做过什么，而是看你能不能把项目背后的推理链路、数值问题和性能瓶颈讲明白。
```
