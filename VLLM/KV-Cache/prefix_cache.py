#  scheduer 中调用kv manager
#  vllm/vllm/v1/core/sched/scheduler.py
#  触发匹配计算
                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

# vllm/vllm/v1/core/kv_cache_manager.py
# 进行命中率计算：

        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(request.block_hashes,
                                                    max_cache_hit_length))

#  vllm/vllm/v1/core/sched/scheduler.py
# scheduler申请slots：

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    # 新算和命中一起申请slot
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

# vllm/vllm/v1/core/kv_cache_manager.py
# manger标记匹配的blocks：

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)  

# vllm/vllm/v1/core/block_pool.py
# pool里面管理的block的引用计数（ref_cnt ）+1：

    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null:
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

# vllm/vllm/v1/core/single_type_kv_cache_manager.py
# pool释放请求对应的blocks：

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        # 此处有个按照逆序来淘汰策略，但该方式目前看非最佳（可留言讨论@kaiyuan）
        ordered_blocks = reversed(req_blocks)         

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

# vllm/vllm/v1/core/block_pool.py
# 释放时，引用计数-1：

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n([
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ])