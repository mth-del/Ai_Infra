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

# 当前版本模块的定义位置：
# BlockPool     vllm/vllm/v1/core/block_pool.py 
# FreeKVCacheBlockQueue   vllm/vllm/v1/core/kv_cache_utils.py
# KVCache Coordinator vllm/vllm/v1/core/kv_cache_coordinator.py

'''
1. 块池（Block Pool）：存储KVCacheBlock，block数量一般在初始化时决定，可以降低CPU侧的操作次数
2. 空闲队列（Free Block Queue）：空闲块的队列，仅存储头尾节点指针信息。
3. 缓存协调模块（KVCache Coordinator）：协调不同的KV cache组。
'''



# vllm/vllm/v1/engine/core.py
# _initialize_kv_caches函数：
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = \
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                available_gpu_memory = [
                    self.available_gpu_memory_for_kv_cache
                ] * len(kv_cache_specs)
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = (
                    self.model_executor.determine_available_memory())
                self.available_gpu_memory_for_kv_cache = \
                    available_gpu_memory[0]


# 代码位置：vllm/vllm/v1/core/kv_cache_utils.py
def get_num_blocks(vllm_config: VllmConfig, num_layers: int,
                   available_memory: int, page_size: int) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks

# 其中同构的attention的page_size计算：
# 代码位置：vllm/vllm/v1/kv_cache_interface.py
    def page_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        return coef * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)              


# block tabel
