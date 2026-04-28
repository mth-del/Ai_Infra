# vLLM Scheduler Demo 梳理

这个 demo 用一个简化版调度器模拟 vLLM 推理服务中的核心流程：多个请求进入系统后，由 Scheduler 决定哪些请求可以进入 batch、哪些请求可以占用 KV cache、每轮 decode 后哪些请求结束并释放资源。

真实 vLLM 的 scheduler 会处理 prefix cache、preemption、chunked prefill、LoRA、多优先级队列、CUDA graph 等复杂逻辑。这个 demo 的目标是先跑通最小主链路：`waiting -> running -> finished`。

## 代码结构

### 1. Tokenizer 和 Prompt 构造

代码一开始从本地加载 Qwen tokenizer：

```python
tokenizer = AutoTokenizer.from_pretrained("/home/timsea/huggingface/Qwen3-0.6B", use_fast=True)
```

然后将普通文本 prompt 转成 Qwen 的 chat template 格式：

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
```

这里的 `add_generation_prompt=True` 会在 prompt 末尾加上 assistant 生成起始标记，表示模型接下来应该开始回复。

### 2. Fake Model

`run_fake_model()` 用随机 token 模拟模型输出：

```python
def run_fake_model(seqs, max_len: int = 15, min_val: int = -1, max_val: int = 999, eos: int = -1):
    token_ids = [
        eos if len(seq) >= max_len else np.random.randint(min_val, max_val + 1)
        for seq in seqs
    ]
    return token_ids
```

它不执行真实 Transformer，只根据当前 sequence 长度决定是否返回 `eos`。这个函数的作用是让 Scheduler demo 可以独立跑通。

注意：当前 `min_val=-1`，而 `eos=-1`，所以随机生成时可能第一轮就生成 eos，导致请求很快结束。如果想观察多轮 decode，可以把 `min_val` 改成 `0`。

### 3. Config

`config` 保存 Scheduler 和 KV cache 的核心配置：

```python
@dataclass
class config:
    model: str = "dummy"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
```

重要字段：

- `max_num_batched_tokens`：一轮调度中最多处理多少 token。
- `max_num_seqs`：同时 running 的最大请求数。
- `max_model_len`：单个请求允许的最大上下文长度。
- `kvcache_block_size`：每个 KV cache block 容纳多少 token。
- `num_kvcache_blocks`：总共有多少个物理 KV cache block。

`__post_init__()` 中有几个约束：

```python
assert self.kvcache_block_size % 256 == 0
assert 1 <= self.tensor_parallel_size <= 8
assert self.max_num_batched_tokens >= self.max_model_len
```

这个 demo 里如果设置 `max_num_batched_tokens=128`，就需要同时设置 `max_model_len=128`，否则会触发断言。

### 4. SamplingParams

`SamplingParams` 保存每个请求的采样参数：

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
```

当前 demo 实际只用到了：

- `max_tokens`：最多生成多少 token。
- `ignore_eos`：遇到 eos 是否继续生成。

### 5. Sequence

`Sequence` 表示一个请求对应的 token 序列。

核心字段：

- `seq_id`：请求 id。
- `status`：请求状态，可能是 `WAITING`、`RUNNING`、`FINISHED`。
- `token_ids`：prompt token 和 completion token 都存在这里。
- `num_prompt_tokens`：prompt 长度。
- `num_tokens`：当前总 token 数。
- `block_table`：逻辑 token block 到物理 KV cache block 的映射。
- `max_tokens`：最多生成多少 completion token。

几个重要 property：

```python
num_completion_tokens = num_tokens - num_prompt_tokens
prompt_token_ids = token_ids[:num_prompt_tokens]
completion_token_ids = token_ids[num_prompt_tokens:]
num_blocks = ceil(num_tokens / block_size)
```

这里的 `block_table` 是理解 PagedAttention 的关键。模型里的 token 是连续的，但 KV cache 可以按 block 离散存储。`block_table` 记录一个 sequence 用了哪些物理 block。

### 6. Block 和 BlockManager

`Block` 表示一个物理 KV cache block：

```python
class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.token_ids = {}
```

`BlockManager` 负责管理所有物理 block：

- `blocks`：所有 KV cache block。
- `free_block_ids`：当前空闲 block id 队列。
- `can_allocate(seq)`：判断是否有足够 block 分给某个 sequence。
- `allocate(seq)`：给 sequence 分配新的 block。
- `free(seq)`：请求结束后释放它占用的 block。

核心逻辑：

```python
need = seq.num_blocks - len(seq.block_table)
```

也就是说，当 sequence 的 token 变长后，如果需要更多 block，就只分配新增的 block。

释放时要注意：

```python
block = self.blocks[block_id]
```

这里应该从 `BlockManager.blocks` 取物理 block，而不是调用 `seq.block(block_id)`。`seq.block(i)` 是按逻辑块切 token，不是取物理 KV cache block。

### 7. Scheduler

`Scheduler` 是整个 demo 的核心。

它维护三个队列：

```python
self.waiting: deque[Sequence] = deque()
self.running: list[Sequence] = []
self.finished: list[Sequence] = []
```

生命周期如下：

```text
add_request()
    -> waiting
schedule()
    -> waiting 中可分配 KV cache 的请求进入 running
step()
    -> 对 running 中的 batch 调 fake model
    -> append token
    -> 判断 eos / max_tokens / KV cache 是否不足
    -> finished 请求释放 KV cache
```

### 8. schedule()

`schedule()` 做两件事。

第一步，将 waiting 请求转入 running：

```python
while self.waiting and len(self.running) < self.max_num_seqs:
    seq = self.waiting[0]
    if not self.block_manager.can_allocate(seq):
        break
    self.waiting.popleft()
    self.block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    self.running.append(seq)
```

第二步，从 running 中构造本轮 batch：

```python
if seq.num_completion_tokens == 0:
    cost = seq.num_prompt_tokens
else:
    cost = 1
```

这里模拟了 prefill 和 decode 的区别：

- Prefill：第一次处理 prompt，需要消耗整个 prompt 长度。
- Decode：后续每轮只生成一个 token，所以 cost 为 1。

### 9. step()

`step()` 表示一次调度执行。

核心步骤：

1. 调用 `schedule()` 得到本轮 batch。
2. 调用 `run_fake_model()` 得到每个 sequence 的 next token。
3. 将 next token append 到 sequence。
4. 检查是否需要新增 KV cache block。
5. 检查是否遇到 eos 或达到 `max_tokens`。
6. 将 finished 请求移出 running，并释放 KV cache。

当前代码中需要特别注意这一段：

```python
still_running = []
for seg in self.running:
    if seq.is_finished:
        self.block_manager.free(seq)
        self.finished.append(seq)
    else:
        still_running.append(seq)
```

这里 `for seg in self.running` 里面却使用了 `seq`，会导致重复处理上一个循环残留的 sequence。应该统一变量名：

```python
still_running = []
for seq in self.running:
    if seq.is_finished:
        self.block_manager.free(seq)
        self.finished.append(seq)
    else:
        still_running.append(seq)
```

## 一次请求的执行流程

以一个 prompt 为例：

1. 文本 prompt 通过 tokenizer 转成 token ids。
2. `scheduler.add_request()` 创建 `Sequence`，放入 `waiting`。
3. `scheduler.step()` 调用 `schedule()`。
4. 如果 KV cache 足够，请求从 `waiting` 进入 `running`。
5. 第一次执行是 prefill，cost 等于 prompt token 数。
6. fake model 生成一个 token。
7. token 被 append 到 sequence。
8. 如果达到 eos 或 `max_tokens`，请求变成 `FINISHED`。
9. Scheduler 释放这个请求占用的 KV cache block。
10. 最后在 `scheduler.finished` 中打印结果。

## 和真实 vLLM 的对应关系

| Demo 概念 | vLLM 中的真实概念 |
| --- | --- |
| `Sequence` | 一个请求或一个生成序列 |
| `block_table` | PagedAttention 中的 block table |
| `BlockManager` | KV cache block allocator |
| `waiting` | 等待调度的新请求 |
| `running` | 正在占用 KV cache 的请求 |
| `finished` | 已完成并可释放资源的请求 |
| `max_num_batched_tokens` | 控制 batch token 总量 |
| `max_num_seqs` | 控制并发 sequence 数 |
| `prefill cost = prompt length` | 首次处理完整 prompt |
| `decode cost = 1` | 每轮为每个请求生成一个 token |

## AI Infra 常见问题和答案

### Q1: LLM 推理为什么需要 Scheduler？

因为线上推理通常同时有多个请求，每个请求的 prompt 长度、生成长度、到达时间都不同。Scheduler 负责决定哪些请求一起组成 batch、哪些请求可以进入 GPU 执行、哪些请求因为资源不足需要等待，从而提高 GPU 利用率并控制延迟。

### Q2: Prefill 和 Decode 有什么区别？

Prefill 是处理用户输入 prompt 的阶段，需要一次性计算整段 prompt 的 attention，因此计算量和 prompt 长度相关。Decode 是逐 token 生成阶段，每轮通常只为每个请求生成一个新 token，但需要读取历史 KV cache。

### Q3: 为什么 LLM 推理要用 KV cache？

自回归生成时，第 t 个 token 会 attend 到前面所有 token。如果每次都重新计算历史 token 的 K/V，成本很高。KV cache 会缓存历史 token 的 key/value，decode 时只计算新 token 的 K/V，然后复用历史缓存。

### Q4: KV cache 为什么容易成为瓶颈？

KV cache 会随着 batch size、sequence length、layer 数、hidden size 增长而快速变大。长上下文和高并发场景下，显存可能主要被 KV cache 占用，而不是模型权重。

### Q5: PagedAttention 解决了什么问题？

传统 KV cache 常要求连续显存，容易产生碎片和浪费。PagedAttention 将 KV cache 拆成固定大小 block，通过 block table 建立逻辑 token 位置到物理 block 的映射，类似操作系统的分页机制，可以更灵活地分配和复用显存。

### Q6: `max_num_batched_tokens` 和 `max_num_seqs` 分别控制什么？

`max_num_batched_tokens` 控制一轮调度中处理的 token 总数，主要影响显存和计算量。`max_num_seqs` 控制同时运行的请求数量，主要影响并发度。前者偏 token 维度，后者偏请求维度。

### Q7: 为什么 decode 阶段 batch 很重要？

Decode 每个请求每轮只生成一个 token，单请求执行很难打满 GPU。把多个请求合成 batch，可以提升矩阵计算规模，提高 GPU 利用率。

### Q8: Continuous Batching 是什么？

Continuous batching 指推理服务不是等一整个 batch 全部完成后再接新请求，而是在每轮 decode 之间动态加入新请求、移除完成请求。这样可以减少排队时间，提高吞吐。

### Q9: 为什么长 prompt 会影响其他请求延迟？

长 prompt 的 prefill cost 很高。如果直接和短请求放在同一轮执行，短请求可能被长 prefill 拖慢。因此真实系统通常会做 chunked prefill、prefill/decode 分离或更复杂的调度策略。

### Q10: 什么是 Chunked Prefill？

Chunked prefill 是把长 prompt 拆成多个小块分多轮处理，而不是一次性处理完整 prompt。这样可以避免一个超长 prompt 独占 batch token budget，让 decode 请求也能穿插执行，降低尾延迟。

### Q11: Throughput 和 Latency 如何权衡？

更大的 batch 通常提高 throughput，但可能增加单个请求等待时间。更激进的调度可以降低 latency，但可能降低 GPU 利用率。推理系统需要在吞吐、首 token 延迟、平均延迟、尾延迟之间做权衡。

### Q12: TTFT 和 TPOT 是什么？

TTFT 是 Time To First Token，表示从请求进入系统到生成第一个 token 的时间，主要受排队和 prefill 影响。TPOT 是 Time Per Output Token，表示生成阶段每个 token 的平均耗时，主要受 decode 性能影响。

### Q13: 为什么 attention 在 decode 阶段仍然可能很慢？

虽然 decode 每轮只处理一个新 token，但这个 token 需要 attend 到全部历史 token。上下文越长，读取 KV cache 的数据越多，memory bandwidth 压力越大。

### Q14: Tensor Parallel 是什么？

Tensor parallel 是把一个模型层内部的大矩阵切到多个 GPU 上并行计算。它可以让单个大模型放进多卡显存，也可以提升单次推理计算吞吐，但会引入 GPU 间通信开销。

### Q15: Pipeline Parallel 和 Tensor Parallel 有什么区别？

Tensor parallel 是切分单层计算，多个 GPU 共同完成同一层。Pipeline parallel 是按层切分模型，不同 GPU 负责不同层。前者通信频繁但粒度细，后者适合更深模型但可能有 pipeline bubble。

### Q16: 模型权重显存和 KV cache 显存有什么区别？

模型权重显存基本是固定的，加载模型后变化不大。KV cache 显存随请求数和上下文长度动态增长，是在线推理中最关键的动态资源。

### Q17: 为什么量化可以提升推理效率？

量化把 FP16/BF16 权重压缩成 INT8、INT4 等格式，可以减少显存占用和内存带宽压力。有些硬件还能直接加速低精度矩阵乘法。但量化可能带来精度损失，并且需要合适的 kernel 支持。

### Q18: speculative decoding 是什么？

Speculative decoding 使用一个小模型先草拟多个 token，再用大模型一次性验证。如果草拟 token 被接受，就能减少大模型调用轮数，从而提升 decode 吞吐。

### Q19: 为什么推理服务需要限流和 admission control？

如果所有请求都直接进入 running，KV cache 可能被打满，导致 OOM 或严重排队。Admission control 根据当前显存、队列长度、token budget 等判断请求是否可以进入系统。

### Q20: 一个 LLM 推理服务的核心模块有哪些？

常见模块包括 tokenizer、request queue、scheduler、model executor、KV cache manager、sampling、streaming output、metrics、admission control、模型并行通信和故障恢复逻辑。