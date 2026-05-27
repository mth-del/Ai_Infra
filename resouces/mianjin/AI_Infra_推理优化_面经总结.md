# AI Infra 推理优化面经总结

## 1. 整体印象

这类面试不是只问项目经历，而是会沿着大模型推理链路连续追问：

```text
推理框架 -> Prefill/Decode -> 显存估算 -> Attention 优化 -> KV Cache -> 算子上限 -> 训练/后训练方向
```

核心考察点：

```text
你是否真正理解 LLM 推理系统，而不是只知道几个名词。
```

回答策略：

```text
先用一句话定义概念，再讲为什么需要它，最后讲工程上怎么做优化和取舍。
```

---

## 2. SGLang

### 问题：你懂不懂 SGLang？它主要解决什么问题？

面试官想确认：

```text
你是否理解现代 LLM serving 系统，而不是只会调用模型接口。
```

简洁回答：

```text
SGLang 是一个面向大模型推理服务的框架，重点解决高吞吐、低延迟和复杂请求编排问题。
它不仅是一个模型 server，还提供了类似程序化 prompt 的能力，并在后端做 continuous batching、KV Cache 管理和推理调度。
```

形象理解：

```text
普通推理框架像“一个人排队点餐”，请求一个个处理；
SGLang 更像“后厨调度系统”，把很多请求拼批、复用缓存、动态调度，让 GPU 尽量一直忙。
```

### 可能追问：SGLang 和 vLLM 有什么关系？

```text
vLLM 更强调高性能 serving runtime，核心能力是 PagedAttention、continuous batching 和 KV Cache 管理。
SGLang 在 serving runtime 之上更强调结构化生成和复杂推理流程编排，同时也可以使用高性能后端。
面试里可以说：二者都服务于高吞吐 LLM 推理，只是关注层次不同。
```

---

## 3. PD 和 PD 分离

### 问题：什么是 PD？什么是 PD 分离？

简洁回答：

```text
PD 指 Prefill 和 Decode 两个阶段。
Prefill 阶段一次性处理 prompt，计算量大，容易吃满 GPU 算力；
Decode 阶段一个 token 一个 token 生成，强依赖 KV Cache，显存和访存压力更明显。
PD 分离就是把 Prefill 和 Decode 拆到不同 GPU、不同进程或不同调度队列上，分别优化吞吐和延迟。
```

形象理解：

```text
Prefill 像读完整篇题目，适合大批量并行；
Decode 像一个字一个字写答案，每一步都依赖前一步。
两者工作模式不同，所以可以分开调度。
```

### 问题：PD 分离怎么控制显存？

回答要点：

```text
显存主要由模型权重、KV Cache、activation/workspace 和 batch 输入输出组成。
Prefill 侧更关注一次性计算和临时激活；
Decode 侧更关注长上下文下 KV Cache 的持续占用。
PD 分离后，可以给 Decode 实例预留更多 KV Cache 空间，给 Prefill 实例配置更大的 batch，提高吞吐。
```

加分句：

```text
如果 Decode 的 KV Cache 快满了，可以通过限制 max_num_seqs、max_model_len、batch token 数，或者做 KV Cache eviction/压缩来控制显存。
```

---

## 4. 训练和推理过程怎么估算显存

### 问题：训练过程怎么估算显存？

常见公式：

```text
训练显存 ≈ 参数 + 梯度 + 优化器状态 + activation + 临时 workspace
```

以 Adam 为例：

```text
参数 FP16:      2 bytes
梯度 FP16:      2 bytes
FP32 master:    4 bytes
Adam m:         4 bytes
Adam v:         4 bytes

合计约 16 bytes / parameter
```

简洁回答：

```text
训练显存最重的是优化器状态和 activation。
Adam 训练时每个参数不只是存权重，还要存梯度、FP32 master weight、m、v，所以远大于推理。
```

### 问题：推理过程怎么估算显存？

常见公式：

```text
推理显存 ≈ 模型权重 + KV Cache + workspace + 输入输出 buffer
```

KV Cache 公式：

```text
KV Cache = batch_size × seq_len × num_layers × 2 × num_kv_heads × head_dim × dtype_bytes
```

其中 `2` 是：

```text
K Cache + V Cache
```

面试表达：

```text
推理时没有梯度和优化器状态，所以模型权重本身比训练省很多。
但长上下文、多 batch 时，KV Cache 会成为主要显存瓶颈。
```

---

## 5. FlashAttention 1/2/3/4

### 问题：FlashAttention 的核心思想是什么？

简洁回答：

```text
FlashAttention 的核心是 IO-aware attention。
普通 attention 会把 QK^T 和 softmax 中间矩阵写到 HBM，显存读写非常大。
FlashAttention 通过 tiling，把 Q/K/V 分块搬到 SRAM/shared memory 中，在片上完成 softmax 和 PV，避免巨大中间矩阵落到 HBM。
```

形象理解：

```text
普通 attention 像每算一步都把草稿纸搬进搬出大仓库；
FlashAttention 是把一小块数据拿到桌面上，边算边归一化，最后只写最终结果。
```

### 问题：FlashAttention 1/2/3 大概区别是什么？

```text
FA1：证明 exact attention 可以通过 tiling 和 online softmax 大幅减少 HBM IO。
FA2：改进并行划分，减少非矩阵乘部分开销，提高 warp/block 级并行度。
FA3：面向 Hopper 架构优化，更好利用 Tensor Core、异步拷贝和新硬件特性。
FA4：如果被问到，可以谨慎说自己了解有限，重点关注最新实现如何进一步适配新 GPU 架构和低精度计算。
```

面试稳妥说法：

```text
我不会强行背版本号细节，但我理解主线：每一代都在减少 HBM IO、提高并行度、提高 Tensor Core 利用率，并适配新的 GPU 架构。
```

---

## 6. KV Cache 压缩和 SVD 压缩

### 问题：为什么要压缩 KV Cache？

简洁回答：

```text
因为 Decode 阶段每生成一个 token，都要保留历史 token 的 K/V。
KV Cache 大小随 batch、seq_len、layer 数线性增长。
长上下文和高并发时，KV Cache 往往比模型权重更容易成为显存瓶颈。
```

### 问题：常见 KV Cache 压缩方法有哪些？

```text
1. 量化：FP16/BF16 KV 压成 INT8/FP8/INT4。
2. 滑动窗口：只保留最近一段上下文。
3. Token eviction：丢弃不重要 token 的 KV。
4. 稀疏注意力：只访问部分历史 token。
5. 低秩压缩：用低秩矩阵近似 KV。
```

### 问题：SVD 压缩怎么理解？

简洁回答：

```text
SVD 压缩就是把一个大的 KV 矩阵近似分解成两个小矩阵，只保留最重要的低秩成分。
如果原矩阵是 K ≈ U_r × S_r × V_r^T，只保留 rank=r 的部分，就能减少存储。
```

形象理解：

```text
一张高清图片很多细节不一定都重要。
SVD 像只保留最主要的轮廓和纹理，用更少的数据近似原图。
```

注意风险：

```text
KV Cache 压缩会影响注意力结果，压得越狠越可能掉精度。
工程上要在显存、速度和模型效果之间做 trade-off。
```

---

## 7. 如何评估算子理论上限

### 问题：如何评估一个算子的理论上限？

简洁回答：

```text
我会用 Roofline 模型。
先算这个算子的 FLOPs，再算它需要读写多少 bytes，得到算术强度 AI = FLOPs / Bytes。
然后和硬件峰值算力、显存带宽比较，判断它是 compute-bound 还是 memory-bound。
```

公式：

```text
性能上限 = min(峰值算力, 显存带宽 × 算术强度)
```

面试表达：

```text
如果 AI 很低，算子大概率被显存带宽限制，优化重点是减少 HBM 读写、提高数据复用。
如果 AI 很高，算子更可能被计算单元限制，优化重点是提高 Tensor Core/SM 利用率。
```

### 可能追问：理论值和 profiler 实测为什么不同？

```text
理论值通常按算法逻辑访存算；
profiler 的 dram bytes 是实际打到 HBM 的流量。
中间有 L1/L2 cache、访存合并、事务粒度、预取和写回策略，所以两者可能不同。
```

---

## 8. 来了之后想干 RL 还是 SFT

### 问题：你更想做 RL 还是 SFT？

稳妥回答：

```text
我对 SFT 和 RL 都有兴趣，但如果结合 AI Infra 背景，我更关注训练和推理系统如何支撑这些算法。
SFT 更偏数据质量、指令跟随和能力注入；
RL 更偏偏好优化、奖励建模和复杂反馈下的策略优化。
```

如果想偏 Infra：

```text
从工程角度，我更想做能提升训练/推理效率的方向，比如 SFT/RL 训练中的显存优化、并行策略、推理服务和算子优化。
算法上我愿意补齐 RLHF、DPO、PPO、GRPO 这些方法，但我的优势是把算法需求落到高性能系统实现上。
```

面试表达：

```text
如果团队更偏 post-training，我可以从 SFT pipeline、RL 训练稳定性和推理采样效率切入；
如果团队更偏 serving，我更适合做推理 runtime、KV Cache、batching 和 kernel profiling。
```

---

## 9. 面试官可能连续追问路线

### 路线一：从 SGLang 追到 serving

```text
SGLang 是什么？
为什么需要 continuous batching？
Prefill 和 Decode 有什么区别？
KV Cache 怎么管理？
显存不够怎么办？
```

回答主线：

```text
LLM serving 的核心矛盾是请求动态变化、上下文长度不同、GPU 需要高利用率。
所以需要 batching、KV Cache 管理、调度和显存控制。
```

### 路线二：从 FlashAttention 追到算子

```text
Attention 为什么慢？
FlashAttention 减少了什么 IO？
online softmax 怎么保证数值正确？
怎么判断算子是 memory-bound 还是 compute-bound？
```

回答主线：

```text
普通 attention 的瓶颈不是只有计算量，还有 QK^T/softmax 中间矩阵的 HBM 读写。
FlashAttention 用 tiling 和 online softmax 把中间结果留在片上，减少 HBM IO。
```

### 路线三：从显存估算追到部署

```text
训练显存怎么算？
推理显存怎么算？
KV Cache 多大？
长上下文怎么省显存？
PD 分离为什么有用？
```

回答主线：

```text
训练显存主要看参数、梯度、优化器和 activation；
推理显存主要看权重和 KV Cache；
长上下文下 KV Cache 是核心瓶颈，所以需要分页、压缩、分离和调度。
```

---

## 10. 蔚来 AI Infra 实习面经：CUDA 算子优化追问

这类问题更偏底层 CUDA 和算子优化，面试官通常不是问概念定义，而是看你能不能把 shared memory、warp 访存和项目优化联系起来。

### 问题：项目经验讨论时，AI Infra 项目应该怎么讲？

回答思路：

```text
不要只说“我用了 CUDA 加速”，要按“瓶颈定位 -> 优化方法 -> profiler 验证 -> 边界条件”来讲。
```

可以这样组织：

```text
我先用 Nsight Systems / Nsight Compute 定位热点 kernel，看 kernel 耗时、SM 利用率、DRAM 吞吐、L2 命中率和 warp stall reason。
确认瓶颈后，再根据算子特点选择优化方式：如果是访存瓶颈，就减少 global memory 访问、做 coalesced access、用 shared memory 复用；如果是计算瓶颈，就提高 Tensor Core/SM 利用率。
最后用 before/after 的 kernel time、dram bytes、occupancy 和数值误差验证优化是否有效。
```

面试加分句：

```text
我不会凭直觉说某个地方慢，而是先 profile，再根据数据判断是 memory-bound、compute-bound，还是同步/atomic/copy 开销。
```

### 问题：共享内存 Bank Conflict 的产生原因与解决方案

简洁回答：

```text
Shared memory 被分成多个 bank，通常可以理解成 32 个 bank。
同一个 warp 的 32 个线程如果在同一条 shared memory 指令中访问同一个 bank 的不同地址，硬件就要串行处理，这就是 bank conflict。
```

典型例子：

```text
s[threadIdx.x]      -> 连续访问，32 个线程落到 32 个 bank，通常无冲突。
s[threadIdx.x * 32] -> stride=32，所有线程都落到同一个 bank，可能产生 32-way conflict。
```

例外：

```text
如果所有线程访问的是同一个 shared memory 地址，硬件会做 broadcast，不算 bank conflict。
```

解决方法：

```text
1. 让 warp 内线程访问连续地址。
2. 避免 stride 刚好是 bank 数量的倍数。
3. 二维 shared memory 做转置或列访问时加 padding，比如 tile[32][33]。
4. 用 Nsight Compute 查看 shared memory bank conflict 指标，确认是否真的成为瓶颈。
```

面试表达：

```text
Bank conflict 本质是 shared memory 物理分 bank，而线程访问模式让多个线程撞到同一个 bank。
优化时不是盲目加 shared memory，而是要保证访问模式对 bank 友好。
```

### 问题：同一 Warp 内不同线程的访存约束是什么？

简洁回答：

```text
GPU 的访存效率高度依赖 warp 内 32 个线程的访问模式。
如果相邻线程访问相邻地址，global memory 可以合并成少量 transaction，shared memory 也通常不会有 bank conflict。
如果线程访问离散、跨很大 stride，global memory transaction 会变多，shared memory 也可能出现 bank conflict。
```

Global memory 角度：

```text
最理想：thread 0 读 addr 0，thread 1 读 addr 4，thread 2 读 addr 8 ...
这样是 coalesced access，带宽利用率高。
```

Shared memory 角度：

```text
最理想：thread 0~31 分别访问不同 bank。
最差：thread 0~31 访问同一个 bank 的不同地址。
```

面试表达：

```text
warp 是 GPU 的基本执行单位，所以优化访存时不是只看单个线程，而是看整个 warp 的地址分布。
连续、对齐、合并，是 GPU 访存优化的核心。
```

### 问题：共享内存中的广播机制 Broadcast 是什么？

简洁回答：

```text
Broadcast 是 shared memory 的一个特殊情况。
如果同一个 warp 内多个线程访问同一个 shared memory 地址，硬件可以把这个值广播给所有线程，这不会产生 bank conflict。
```

例子：

```text
float x = s[0];
```

如果一个 warp 的 32 个线程都执行这句：

```text
所有线程访问同一个地址 s[0]，硬件广播一次即可。
```

和 bank conflict 的区别：

```text
同一个 bank + 同一个地址：broadcast，不冲突。
同一个 bank + 不同地址：bank conflict，需要串行。
```

面试表达：

```text
所以判断 bank conflict 时不能只看是不是同一个 bank，还要看是不是同一个地址。
同地址是 broadcast，不同地址才会冲突。
```

---

## 11. 暑期 AI Infra 面经：CUDA / Attention / vLLM 追问

这一组问题更偏手撕 CUDA、Attention 算子、推理框架适配和 GPU 硬件理解。核心不是背概念，而是能把“代码怎么写、为什么快、瓶颈在哪里”讲清楚。

### 问题：一道 CUDA，矩阵转置后做行规约，怎么设计？

回答思路：

```text
矩阵转置和行规约都容易受访存模式影响。
转置时读可以是 coalesced，但直接列写容易不连续；常见做法是用 shared memory tile。
每个 block 读一个 TILE，比如 32x32，先按连续地址读到 shared memory，再转置写出。
为了避免 shared memory 列访问 bank conflict，可以定义 tile[32][33] 加 padding。
```

行规约可以这样讲：

```text
转置后每一行连续，适合一个 block 或一个 warp 负责一行。
行内规约可以先在线程内累加，再用 warp shuffle 做 warp 内 reduce，最后跨 warp 用 shared memory 汇总。
```

面试表达：

```text
核心是让 global memory 读写尽量 coalesced，shared memory 避免 bank conflict，规约阶段用 warp shuffle 减少 shared memory 和同步开销。
```

### 问题：让你实现 Attention，你会怎么写？

简洁回答：

```text
最朴素 attention 是三步：
1. S = QK^T / sqrt(d)
2. P = softmax(S)
3. O = PV
```

但工程上不能直接把完整 `S` 写到 global memory：

```text
因为 S 的大小是 seq_len x seq_len，长序列下 HBM 读写和显存占用都很大。
```

优化思路：

```text
用 FlashAttention 的思路，对 Q/K/V 做 block tiling。
每次只加载一块 Q 和一块 K/V 到 shared memory，在片上做局部 QK、online softmax 和 PV 累加。
最终只把 O 写回 global memory，避免把完整 attention score 矩阵落到 HBM。
```

### 问题：CUDA 代码思路怎么讲？

回答模板：

```text
我会先讲线程映射：一个 block 负责什么，一个 thread 或 warp 负责什么。
然后讲数据从 global memory 到 register/shared memory 的路径。
接着讲同步点、访存是否 coalesced、是否有 bank conflict。
最后讲输出写回、是否需要 atomic，以及如何用 profiler 验证瓶颈。
```

简洁版：

```text
讲 CUDA 代码不要一行行念，要讲清楚数据划分、线程划分、内存层次和性能瓶颈。
```

### 问题：内存对齐如何实现？

简洁回答：

```text
内存对齐是为了让硬件用更少的 memory transaction 完成读写。
GPU 上相邻线程最好访问连续、对齐的地址，比如 float4 读写要求 16B 对齐。
```

常见做法：

```text
1. 分配内存时 cudaMalloc 本身会返回足够对齐的地址。
2. 数据结构大小尽量按 4/8/16 bytes 对齐。
3. 用 float4、half2、int4 做向量化读写时，保证地址满足对齐要求。
4. 对二维矩阵可以 padding 到 32、64、128 的倍数，方便 warp 合并访问。
```

面试表达：

```text
对齐的本质是减少 transaction 数量，提高有效带宽；不对齐会导致一次逻辑访问拆成多次内存事务。
```

### 问题：GQA 的作用是什么？是否减少计算量？

简洁回答：

```text
GQA 是 Grouped Query Attention。
它让多个 query heads 共享同一组 key/value heads，是 MHA 和 MQA 之间的折中。
```

它主要减少的是：

```text
KV Cache 显存占用
Decode 阶段读取 KV Cache 的带宽
```

是否减少计算量：

```text
Prefill 阶段 QK^T 的主计算量不一定按相同比例下降，因为 query head 仍然存在。
Decode 阶段更明显的收益是 KV Cache 读写更少，访存压力下降。
```

面试表达：

```text
GQA 的核心价值不是简单说“减少计算量”，更准确是减少 K/V head 数，从而降低 KV Cache 显存和 decode 阶段访存带宽。
```

### 问题：Warp 内通信速度以及内存层级怎么理解？

简洁回答：

```text
Warp 内通信最快的是寄存器级 shuffle，不需要经过 shared memory。
其次是 shared memory，同一个 block 内可见，延迟较低。
再往外是 L1/L2 cache 和 global memory，延迟逐级升高。
```

常见速度直觉：

```text
register / shuffle：最快，warp 内直接交换
shared memory：很快，但要注意 bank conflict
L2 cache：全 GPU 共享，延迟中等
global memory：带宽高但延迟大
host memory：最慢，走 PCIe/NVLink
```

面试表达：

```text
如果只是 warp 内规约，我优先用 shuffle；如果要 block 内多个 warp 共享数据，再用 shared memory。
```

### 问题：FlashAttention3 和 TMA 有什么关系？

简洁回答：

```text
TMA 是 Hopper 架构引入的 Tensor Memory Accelerator，可以更高效地把多维 tensor tile 从 global memory 异步搬到 shared memory。
FlashAttention3 面向 Hopper 做优化，会利用异步数据搬运和 Tensor Core 计算重叠，减少等待，提高吞吐。
```

形象理解：

```text
以前搬数据更多要靠线程自己发起 load；
TMA 像专门的搬运引擎，把大块 tensor tile 搬到 shared memory，让计算单元更专注做矩阵乘。
```

谨慎回答：

```text
我没有完整精读 FA3 所有源码，但理解它的主线是利用 Hopper 的 TMA、异步流水和 Tensor Core，把数据搬运和计算更充分重叠。
```

### 问题：vLLM 的模型适配过程是什么？

简洁回答：

```text
vLLM 适配一个模型，主要要处理模型结构、权重加载、attention 实现、KV Cache 布局和 tokenizer/config 对齐。
```

可以按步骤讲：

```text
1. 解析 HuggingFace config，确认 hidden size、num layers、num heads、num_kv_heads、rope 参数等。
2. 实现或复用对应的 model class，包括 embedding、attention、MLP、norm 和 lm_head。
3. 做权重映射，把 HF checkpoint 的参数名映射到 vLLM 内部模块。
4. 适配 attention 后端，包括 PagedAttention、GQA/MQA、RoPE 和 KV Cache layout。
5. 跑数值对齐，对比 HF baseline 的 logits 或生成结果。
```

面试表达：

```text
模型适配不是只改配置，最容易出问题的是权重命名、RoPE 参数、GQA 的 KV head 数和 KV Cache 布局。
```

### 问题：PagedAttention 解决什么问题？原理是什么？

简洁回答：

```text
PagedAttention 解决的是 KV Cache 管理问题。
传统 KV Cache 给每个请求分配连续大块显存，容易因为请求长度不同造成显存碎片和浪费。
PagedAttention 借鉴操作系统分页，把 KV Cache 切成固定大小 block，通过 block table 管理逻辑 token 到物理 KV block 的映射。
```

形象理解：

```text
不用给每个请求提前分一整本空白本子，而是按页分配，用多少页拿多少页。
```

收益：

```text
1. 减少 KV Cache 显存碎片。
2. 支持 continuous batching 下动态加入和结束请求。
3. 方便 prefix cache、beam search 等共享 KV block。
```

### 问题：模型量化相关会怎么答？

简洁回答：

```text
模型量化是用更低 bit 表示权重或激活，降低显存占用和访存带宽。
推理里常见 W8A8、W4A16、FP8。
```

关键区分：

```text
W4A16：权重 4-bit，激活 FP16，省显存，精度较稳。
W8A8：权重和激活都 8-bit，吞吐更好，但校准更重要。
FP8：有指数位，动态范围比 INT8 更自然，适合新 GPU。
```

面试表达：

```text
量化不只是把 dtype 改小，还要处理 scale、zero point、outlier、per-channel/per-token 量化，以及反量化和 GEMM 融合。
```

### 问题：RTX Pro 5000 相比 L20 有什么优势？

回答时先声明：

```text
具体要看是哪一代 RTX Pro 5000 和具体部署场景，不能只看名字。
```

比较维度：

```text
1. 架构代际：是否支持更新的 Tensor Core、FP8、Transformer Engine 等。
2. 显存容量和带宽：决定大模型权重和 KV Cache 容量。
3. 算力：FP16/BF16/FP8 Tensor Core 峰值。
4. 部署形态：L20 更偏数据中心推理卡，RTX Pro 更偏工作站/专业图形和本地开发。
5. 软件生态：驱动、虚拟化、MIG/多租户能力、数据中心稳定性。
```

面试表达：

```text
如果是 AI Infra 部署，我会优先比较显存容量、显存带宽、Tensor Core 低精度能力、功耗和数据中心部署特性，而不是只看单卡理论算力。
```

### 问题：CUDA Graph 是什么？vLLM/SGLang 为什么会用？

简洁回答：

```text
CUDA Graph 是把一串 CUDA 操作提前捕获成图，后续重复执行时不用每次都从 CPU 逐个 launch kernel。
它主要减少 CPU launch overhead，适合 decode 这种小 kernel、多次重复执行的场景。
```

为什么 LLM 推理会用：

```text
LLM decode 每生成一个 token 都会执行一批相似 kernel。
如果 batch shape 稳定，可以用 CUDA Graph 复用执行图，减少 CPU 调度开销和 kernel launch latency。
```

限制：

```text
CUDA Graph 对 shape 和内存地址比较敏感。
动态 batch、动态 seq_len、请求频繁变化时，需要做 padding、bucket 或维护多套 graph。
```

### 问题：如何设计 block 数量？

简洁回答：

```text
block 数量由数据规模和每个 block 负责的工作量决定。
一般先确定一个 block 处理多少元素，再用 ceil(N / block_size) 计算 grid size。
```

例子：

```text
int block = 256;
int grid = (N + block - 1) / block;
kernel<<<grid, block>>>(...);
```

进一步考虑：

```text
1. blockDim 通常从 128/256/512 试。
2. 保证有足够多 block 填满所有 SM。
3. 关注寄存器、shared memory、occupancy。
4. 不是 occupancy 越高越好，最终要看实际 kernel time 和 profiler。
```

面试表达：

```text
block size 是理论约束加 profiling sweep 的结果，不是固定答案。
```

### 问题：优化 CUDA Kernel 时通常从哪些方面入手？

面试官想确认：

```text
你是否有系统化的 kernel 优化思路，而不是只会说 shared memory、调 block size。
```

简洁回答：

```text
我一般先 profile 定位瓶颈，再判断 kernel 是 memory-bound、compute-bound，还是受同步、分支、atomic、launch 开销影响。
然后分别从访存、计算、并行度、同步和调度几个方向优化。
```

常见切入点：

```text
1. 访存优化：
   保证 global memory coalesced access，减少非连续访问；
   用 shared memory / register tiling 复用数据；
   减少不必要的 global load/store；
   使用 vectorized load/store，例如 float4。

2. 计算优化：
   提高算术强度，让同一份数据做更多计算；
   对 GEMM/Attention 使用 tiling、Tensor Core、算子融合；
   避免重复计算，把中间量放在寄存器里复用。

3. 并行度和 occupancy：
   合理选择 blockDim/gridDim；
   控制寄存器和 shared memory 使用量，避免 occupancy 过低；
   但不盲目追求 100% occupancy，最终以 kernel time 为准。

4. 分支和线程发散：
   减少 warp 内不同线程走不同分支；
   对不规则 workload 做重排或分桶，让同一个 warp 处理相似任务。

5. 同步和原子操作：
   减少 __syncthreads()；
   尽量把全局 atomic 改成 block 内归约后再写回；
   用 warp-level primitive 做局部 reduce。

6. 调度和端到端：
   小 kernel 可以考虑融合，减少 launch overhead；
   H2D/D2H 可以用 stream 和 pinned memory 做重叠；
   用 CUDA Graph 减少重复 launch 的 CPU 调度开销。
```

面试表达：

```text
我的优化顺序通常是：先用 Nsight 看瓶颈，再看访存是否合并、DRAM 吞吐是否高、SM 是否吃满、warp stall 原因是什么。
如果是 memory-bound，就优先减少 HBM 访问和提高数据复用；
如果是 compute-bound，就看 Tensor Core、指令吞吐和算子融合；
如果是调度瓶颈，就考虑 stream、CUDA Graph 或 kernel fusion。
```

### 问题：如何减少计算单元到 global memory 的内存转运？

简洁回答：

```text
核心是提高数据复用，减少 HBM 读写。
```

常见方法：

```text
1. 用 shared memory / register tiling，把数据搬到片上后多次复用。
2. 做算子融合，避免中间结果写回 global memory。
3. 使用 coalesced access 和向量化读写，提高有效带宽。
4. 减少不必要的 global store，只写最终结果。
5. 对 attention/GEMM 这类算子，用 tiling 和 double buffering 重叠搬运与计算。
```

面试表达：

```text
GPU 优化里很大一部分不是让每个线程算得更多，而是让数据少走远路，尽量留在 register/shared memory/L2 这些更近的层级里。
```

---

## 12. 百度 AI Infra 暑期一面：项目 + 推理 + CUDA

### 项目：加速时的 vLLM 细节

面试官想确认：

```text
你是否真的用过 vLLM，还是只跑过 demo。
```

简洁回答要点：

```text
vLLM 核心是 PagedAttention + continuous batching：KV 按 block 分页管理，动态 batch 里不同请求长度不一也能拼在一起跑。
可以提：max_num_seqs、max_model_len、KV dtype、W8A16/W4 权重量化、prefix caching、spec decode、CUDA Graph 减少 launch 开销。
```

### 项目：遇到的难点

回答结构：

```text
选一个具体瓶颈：例如长上下文下 KV 显存、吞吐与延迟权衡、或某类 kernel 的 memory-bound。
说明如何定位（Nsight / 日志 / 指标），如何改（调度、量化、算子、参数），最后用数据验证。
```

### 八股：KV Cache 为什么存在、解决什么问题

```text
Decode 每步只新增一个 token，若没有 KV Cache，就要对整段历史重新算 K/V，复杂度随长度平方增长。
KV Cache 存历史 token 的 K/V，新步只算当前 token 的 Q/K/V 并与历史 K/V 做 attention，把重复计算变成 O(1) 的增量更新（相对整段重算）。
```

### 八股：KV Cache 怎么算

```text
KV ≈ 2 × batch × seq_len × num_layers × num_kv_heads × head_dim × dtype_bytes
其中 2 表示 K 与 V 各一份；GQA 时用 num_kv_heads 而不是 num_heads。
```

### 八股：CUDA 内存层次结构

```text
寄存器（每线程）→ shared memory（每 block）→ L1/L2 cache → global memory（HBM）→ 主机内存（PCIe）。
越近越快越小；优化目标是减少 global 访问、提高合并访存与片上复用。
```

### 八股：block 级规约（例如求一组线程中的最大值）

```text
典型流程：每个线程先算局部值 → block 内用 shared memory 做 tree reduction，或 warp shuffle 做 warp 内 max，再由一个 warp 汇总到 shared 写一个 block 结果。
最后用单个线程写 global，或若需要全局 max 再二级规约 / atomic。
```

### 八股：decode 阶段输出 token 怎么选

```text
贪心：每步取 argmax，快但多样性差。
采样：temperature、top-k、top-p（nucleus）、min-p 等，平衡质量与多样性。
束搜索 beam search：牺牲算力换质量。
部分框架还支持对比解码、speculative decoding 等。
```

### 八股：更倾向框架层还是算子层

稳妥回答：

```text
算子层决定单 kernel 上限，框架层决定调度、显存与端到端吞吐。
我兴趣在 X（二选一或都沾），但会配合另一端：算子人要懂 batching/KV，框架人要懂瓶颈算子。
```

### 八股：PD 分离的大致流程

```text
Prefill：吃 prompt，算力密集，可大 batch 高吞吐。
Decode：逐 token，KV 常驻显存，延迟敏感。
分离后：请求先到 prefill 节点算完首包或首段 KV，再把 KV 或中间状态交给 decode 节点续写；调度上可分别扩缩容、分别调显存与算力配比。
```

### 八股：推理加速手段有哪些

```text
模型侧：量化、蒸馏、稀疏、GQA/MQA、架构裁剪。
系统侧：continuous batching、PagedAttention、PD 分离、KV 压缩、prefix cache、CUDA Graph、多 stream、算子融合。
算法侧：speculative decoding、早停等。
```

---

## 13. 最后总结模板

如果面试官让你整体总结，可以这样说：

```text
大模型推理优化我会从三个层面看：

第一是模型和显存层面：权重量化、KV Cache 估算和压缩；
第二是 runtime 调度层面：continuous batching、PD 分离、stream 和 cache 管理；
第三是 kernel 层面：FlashAttention、算子融合、Roofline 分析和 profiling。

核心方法不是凭直觉优化，而是先用 profiler 定位瓶颈，再判断是 compute-bound、memory-bound 还是调度/拷贝瓶颈，最后做针对性优化。
```
