# RadixPrefixCache 核心代码讲解

下面这份代码实现的是一个基于 Radix Tree（基数树）的前缀缓存，用来做 LLM 推理里的 KV Cache 复用。

## 1. 整体目标

在大模型推理中，不同请求经常共享前缀（例如 system prompt 或公共上下文）。  
如果每次都重新 prefill，会浪费大量计算。

这个模块的目标是：

```text
把“前缀 token 序列 -> KV Cache 物理页索引”的映射存到 Radix Tree
请求到来时先查最长前缀并复用
显存紧张时按 LRU 驱逐未使用的缓存段
```

---

## 2. 核心类与职责

### 2.1 `RadixTreeNode`

每个节点代表一段连续前缀，包含：

```text
_key      : 这段前缀 token（Tensor）
_value    : 对应 KV 物理页索引（Tensor）
_length   : 段长度
children  : 子节点字典
_parent   : 父节点
ref_count : 引用计数（>0 表示被请求占用，不可驱逐）
timestamp : 最近访问时间（LRU）
```

关键方法：

- `get_match_len(input_ids)`：比较当前节点 `_key` 与输入，返回匹配长度。
- `split_at(pos)`：在部分匹配时把节点切成两段（前缀段 + 剩余段）。

---

### 2.2 `RadixCacheHandle`

`handle.node` 指向匹配到的树节点。  
`get_matched_indices()` 从该节点一路回溯到根，拼接整条路径的 `_value`，得到可复用的 KV 索引。

---

### 2.3 `RadixPrefixCache`

是缓存管理器，负责：

- 查询：`match_prefix`
- 插入：`insert_prefix`
- 锁保护/解锁：`lock_handle`
- 驱逐：`evict`
- 统计：`size_info`

内部状态：

```text
evictable_size : 可驱逐 token 数
protected_size : 受保护 token 数（正在被请求使用）
root_node      : 根节点（ref_count=1，永不驱逐）
```

---

## 3. key_fn：children 字典如何索引

`children` 的 key 不是整个序列，而是“该段首页的 page 粒度内容”：

```python
if page_size == 1:
    key = x[0].item()
else:
    key = tuple(x[:page_size].tolist())
```

好处是可以快速决定“下一步走哪个子节点”，减少遍历开销。

---

## 4. 核心流程一：前缀匹配 `_tree_walk`

`_tree_walk(input_ids)` 返回 `(node, prefix_len)`，表示：

- 最长匹配长度 `prefix_len`
- 匹配终点节点 `node`

逻辑：

1. 从 `root_node` 开始，按 `key_fn` 在 `children` 查下一跳。
2. 找不到子节点：直接返回当前节点和匹配长度。
3. 找到子节点后，用 `get_match_len` 计算实际匹配长度，并对齐到 `page_size`。
4. 如果只匹配到该节点的一部分，调用 `split_at` 把节点切开并返回。
5. 如果完全匹配该节点，继续向下走。
6. 每次访问命中节点都更新 `timestamp`（用于 LRU）。

---

## 5. 核心流程二：查询与插入

### 5.1 `match_prefix`

调用 `_tree_walk` 找最长公共前缀，返回 `MatchResult(RadixCacheHandle(...))`。  
后续可通过 handle 取到复用 KV 的物理索引。

### 5.2 `insert_prefix`

1. 先把输入裁到 `page_size` 对齐长度（尾部不足一页不入树）。
2. `_tree_walk` 找到已有前缀长度 `prefix_len`。
3. 若 `prefix_len < insert_len`，创建新节点挂到树上，保存未命中的后缀段。
4. 新节点初始是可驱逐状态，`evictable_size += new_node.length`。

---

## 6. 核心流程三：引用计数与保护区

`lock_handle(handle, unlock=False)` 会沿着 `handle.node -> root` 路径逐层更新 `ref_count`。

- 上锁（`unlock=False`）：
  - `ref_count` 从 0 变 1：从 evictable 移到 protected
- 解锁（`unlock=True`）：
  - `ref_count` 从 1 变 0：从 protected 移回 evictable

这保证“正在被请求使用的缓存不会被驱逐”。

---

## 7. 核心流程四：驱逐 `evict`

`evict(size)` 的策略是：**只驱逐 `ref_count==0` 的叶子节点，按最久未使用优先（LRU）**。

步骤：

1. `_collect_leave_nodes_for_evict()` 收集所有可驱逐叶子。
2. 用 `heapq` 按 `timestamp` 建最小堆（最旧先出）。
3. 循环弹出节点直到累计驱逐长度达到 `size`。
4. 删除该节点在父节点的引用，并回收其 `value`（物理页索引）。
5. 若父节点因此变成叶子且 `ref_count==0`，继续加入堆（向上级联回收）。

返回值是被驱逐的 KV 索引拼接张量，用于上层真正释放物理页。

---

## 8. 节点分裂为什么必要

当树中已有段与新输入“前半段相同、后半段不同”时，必须分裂节点。

示例：

```text
已有：How can I help you
新来：How can I do this
```

需要拆成：

```text
How can I   (公共前缀)
├── help you
└── do this
```

否则无法正确表示分叉路径。

---

## 9. 时间复杂度直觉

- 前缀匹配：与命中路径长度成正比（通常远小于全序列重算成本）。
- 插入：一次匹配 + 可能一次节点创建/分裂。
- 驱逐：主要受可驱逐叶子数量影响，堆操作是 `O(log n)`。

这套结构用少量管理开销换取了大量 prefill 复用收益。

---

## 10. 端到端工作流

```text
请求到来
  -> match_prefix 找最长可复用前缀
  -> lock_handle 保护命中路径
  -> 对未命中部分执行 prefill / decode
  -> insert_prefix 把新产生的 KV 写回树
  -> unlock_handle 释放保护

显存不足
  -> evict(size) 按 LRU 回收 ref_count=0 的叶子段
```

一句话总结：  
这是一个“按 page 对齐、支持最长前缀复用、引用计数保护、LRU 叶子驱逐”的 KV Cache 前缀树实现。
