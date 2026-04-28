# 代码位置：vllm/vllm/v1/core/kv_cache_utils.py
# KVCacheBlock定义：

from typing import Optional


class KVCacheBlock:
    block_id: int
    # 表示有多少请求使用该block
    ref_cnt:int = 0

    # _block_hash用作已完成计算块的唯一标识
    _block_hash:Optional[BlockHashWithGroupId] = None
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None
    is_null: bool = False