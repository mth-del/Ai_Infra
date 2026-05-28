# ai_infra

个人 **AI Infra** 学习与实验仓库：CUDA 算子、大模型推理（vLLM / KV Cache / 并行）、PyTorch/CUTLASS 练习，以及面试面经整理。

---

## 环境

```bash
# Python 3.12+，依赖见 pyproject.toml
uv sync          # 或 pip install -e .
source .venv/bin/activate
```

主要依赖：`torch >= 2.11`。

CUDA 示例需本机安装 `nvcc`，架构按显卡修改 `-arch=sm_XX`（如 RTX 4060 常用 `sm_89`）。

---

## 目录结构

```text
ai_infra/
├── cuda/                    # CUDA 算子手写与笔记（按日期/主题分目录）
├── cutitle/                 # CUTLASS / PyTorch 侧小实验（matmul、rmsnorm、reduce 等）
├── VLLM/                    # 推理框架：Attention、KV Cache、并行、调度、MTP
├── minisglang/              # 精简推理相关笔记（如 KV Cache）
├── zhuoyu/                  # 项目向 demo：BA Jacobian、BEV lifting、ORB CUDA
├── pytorch/                 # PyTorch 注意力等基础实现
├── transfromer/             # Transformer 组件（attention、位置编码）
├── llama.cpp/               # llama.cpp 相关笔记与资料
├── resouces/mianjin/        # 面经归纳（推理优化、蚂蚁三面等）
├── prompt/                  # 提示词 / 学习提纲
├── paper/                   # 论文 PDF（FlashAttention、SGLang 等）
├── summary/                 # 算法思维导图等
├── main.py                  # 入口占位
├── pyproject.toml
└── README.md
```

---

## 模块说明

### `cuda/` — CUDA 算子

按主题分目录，多数含 `.cu` + `.md`（原理与优化记录），部分带 PyTorch 对照。

| 目录 | 内容 |
|------|------|
| `2026_4_23_element/` | Elementwise |
| `2026_4_24_rms_norm/` | RMSNorm |
| `2026_4_25_sigmoid/` | Sigmoid |
| `2026_4_29_softmax/` | Softmax |
| `2026_5_7_sgemm/` | SGEMM（naive / tiling / vectorize），见 `sgemm.md` |
| `202_5_15_reduce/` | Reduce（进行中） |
| `cuda_examples/` | 示例：GEMM、2D reduce、warp shuffle reduce，`Makefile` 统一编译 |

编译示例（以 SGEMM 为例，需按实际路径调整）：

```bash
nvcc -O3 -arch=sm_89 cuda/2026_5_7_sgemm/segmm.cu -o /tmp/segmm
```

带 PyTorch 扩展的算子可参考 README 下方「与 PyTorch 联编」一节。

### `cutitle/` — CUTLASS / 高层算子实验

Python 脚本，用于矩阵乘、RMSNorm、Reduce、FlashAttention 等方向的实验与 profiling。

### `VLLM/` — 大模型推理

| 子目录 | 内容 |
|--------|------|
| `Attention/` | MHA / MQA / GQA、`attention.md` |
| `KV-Cache/` | KV 原理、`kv_cache_utils.py`、`prefix_cache.py` |
| `parallel/` | DP / TP / SP 笔记与示例脚本 |
| `VLLM_Scheduler/` | 调度器笔记与 `scheduler.py` |
| `MTP/` | 推测解码（Speculative Decoding） |
| `PD分离/` | Prefill-Decode 分离、KV 传递（xmind） |

### `minisglang/`

精简版推理链路笔记，如 `kv cache/kv_cache.md`。

### `zhuoyu/` — 工程向 CUDA Demo

| 文件 | 说明 |
|------|------|
| `ba_jacobian_compare.cu` | BA 重投影 Jacobian 构建；含 CPU / CUDA naive / shared memory；可编译为 `bev_lifting` 等 benchmark |
| `ba_jacobian_kernel_demo.cu` | Jacobian kernel 拆分 demo |
| `orb_fast_cuda_demo.cu` | ORB FAST 检测 CUDA 化 |
| `orb_fast_cuda_streams_demo.cu` | 多 Stream + 显存 buffer 复用 |
| `BA.md` | BA / ORB 优化记录与 benchmark 数据 |

编译运行示例：

```bash
nvcc -O3 -arch=sm_89 zhuoyu/ba_jacobian_compare.cu -o zhuoyu/bev_lifting
./zhuoyu/bev_lifting
```

### `pytorch/`、`transfromer/`

- `pytorch/`：Self-Attention 等基础实现与说明。
- `transfromer/`：`attention.py`、`positionalEncoding.py`（目录名为历史拼写，保留不改）。

### `resouces/mianjin/`

面试问题归纳，例如：

- `AI_Infra_推理优化_面经总结.md` — SGLang、PD 分离、FlashAttention、KV 压缩、CUDA 八股等
- `2026_5_5 ai_infra蚂蚁/蚂蚁_AI_Infra_三面_问题归纳.md`

### 其他

- `paper/`：FlashAttention、SGLang 等 PDF
- `prompt/`：`ai_infra.md`、`langGPT.md` 学习提纲
- `llama.cpp/`：llama 相关 xmind / 论文

---

## 性能分析

```bash
# Nsight Systems：端到端 timeline
nsys profile --stats=true python cutitle/reduce.py

# Nsight Compute：单 kernel 指标（需 counter 权限，必要时 sudo + ncu 绝对路径）
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./zhuoyu/bev_lifting
```

---

## 与 PyTorch 联编（可选）

部分 `.cu` 需要 Torch 头文件时，可先取 include 路径再编译：

```bash
TORCH_INC=$(python -c "from torch.utils.cpp_extension import include_paths; print(include_paths()[0])")
TORCH_API_INC=$(python -c "from torch.utils.cpp_extension import include_paths; print(include_paths()[1])")
PY_INC=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")

nvcc -std=c++17 -c cuda/2026_4_24_rms_norm/rms_norm.cu -o /tmp/rms_norm.o \
  -I"$TORCH_INC" -I"$TORCH_API_INC" -I"$PY_INC"
```

---

## 学习路线（建议）

```text
1. cuda/ 基础算子（element → reduce → gemm）+ cuda_examples
2. VLLM/ Attention + KV-Cache + parallel
3. cutitle/ 与 FlashAttention / 推理 profiling
4. zhuoyu/ 结合 BA / BEV 做 kernel 优化与 Nsight 验证
5. resouces/mianjin/ 对照面经查漏补缺
```

---

## 备注

- 仓库为**个人笔记 + 代码实验**，非单一可发布产品；各子目录独立编译运行。
- 学术网络代理等环境配置请按本机情况自行处理，不写入仓库默认流程。


## 学习文档

### VLLM
[nanovllm框架](https://zhuanlan.zhihu.com/p/2008285806222132143)


### Sglang



### 算子优化


### 技术原理
[paged_attention](https://www.zhihu.com/search?type=content&q=paged_attention)
[flash_attention]()
[continuous_batching]()
[KV_Cache压缩在nano-vllm实现](https://zhuanlan.zhihu.com/p/2031044402987184600)
[]

## 开源项目
[vllm](https://github.com/vllm-project/vllm)
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
[sglang](https://github.com/sgl-project/sglang)
[kvllm](https://github.com/TheToughCrane/nano-kvllm)