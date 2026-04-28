from asyncio import FastChildWatcher
from pyclbr import Class
from transformers import AutoTokenizer
# 从本地加载Qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/timsea/huggingface/Qwen3-0.6B", use_fast=True)

prompts = [
    "hi, I'm timsea",
    "Do you subscribe InfraTech?",
]
# 把纯文本 prompt 转成 Qwen 聊天模板格式（带 role 和 generation 前缀）
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,  # 先返回字符串，不直接转 token id
        add_generation_prompt=True, # 在末尾加 assistant 生成起始标记
    )
    for prompt in prompts
]
# 再把模板字符串编码成 token id 序列并打印
# for prompt in prompts:
#     print(tokenizer.encode(prompt))

import numpy as np
def run_fake_model(seqs, max_len: int = 15, min_val:int =-1,max_val: int = 999, eos: int=-1):
    token_ids = [eos if len(seq) >= max_len else np.random.randint(min_val, max_val + 1) for seq in seqs]
    return token_ids

# seqs = [[536, 332, 844, 933, 736, 476, 87], [1000]]
# print(run_fake_model(seqs))



import os
from dataclasses import dataclass
@dataclass
class config:
    model: str = "dummy"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_num_batched_tokens >= self.max_model_len

# 采样参数
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

# 请求序列
from copy import copy
from enum import Enum,auto
from itertools import count
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens =len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens
    def __getitem__(self,key):
        return self.token_ids[key]
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]


# KV 管理器
# paged attention原理管理kv cache
# 以block为单位管理KV cache 每个block 包含若干tokens cache空间
# 由block mannager统一管理
from collections import deque

class Block:
    def __init__(self,block_id:int):
        self.block_id = block_id
        self.ref_count = 0
        self.token_ids = {}
    
    def reset(self):
        self.ref_count = 0
        self.token_ids = []

class BlockManager:
    def __init__(self,num_blocks:int,block_size:int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        # 空闲的block
        self.free_block_ids = deque(range(num_blocks))

    # 是否可以分配的空间
    def can_allocate(self,seq:Sequence) -> bool:
        need = seq.num_blocks - len(seq.block_table)
        return len(self.free_block_ids) >= need
    # 分配空间
    def allocate(self, seq:Sequence):
        need = seq.num_blocks - len(seq.block_table)
        if need > len(self.free_block_ids):
            raise RuntimeError("No enough KV cache blocks")
        
        for _ in range(need):
            block_id = self.free_block_ids.popleft()
            block = self.blocks[block_id]
            block.ref_count = 1
            seq.block_table.append(block_id)
    # 释放sequence
    def free(self, seq:Sequence):
        # 按逻辑块切分的
        for block_id in seq.block_table:
            # 物理kv cache block
            block = self.blocks[block_id]
            block.reset()
            self.free_block_ids.append(block_id)

        seq.block_table.clear()


class Scheduler:
    def __init__(self,cfg:config):
        self.max_num_seqs = cfg.max_num_seqs 
        self.max_num_batched_tokens = cfg.max_num_batched_tokens
        self.eos = cfg.eos
    
        Sequence.block_size = cfg.kvcache_block_size
        num_blocks = cfg.num_kvcache_blocks

        if num_blocks <= 0:
            num_blocks = 64
        
        # block mannager
        self.block_manager = BlockManager(
            num_blocks= num_blocks,
            block_size = cfg.kvcache_block_size
        )

        self.waiting:deque[Sequence] = deque()
        self.running:list[Sequence] = []
        self.finished:list[Sequence] = []

    def add_request(self,token_ids:list[int],sampling_params:SamplingParams):
        seq = Sequence(token_ids,sampling_params)
        self.waiting.append(seq)
        return seq.seq_id


    def schedule(self)->list[Sequence]:
        batch = []

        # 把waiting中的kv cache请求调入running
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            # 无法分配空间
            if not self.block_manager.can_allocate(seq):
                break

            self.waiting.popleft()
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

        # 从runing中运行
        num_batched_tokens = 0
        for seq in self.running:
            if seq.is_finished:
                continue

            # prefill 阶段消耗promt长度，decode阶段每轮消耗1token
            if seq.num_completion_tokens == 0:
                cost = seq.num_prompt_tokens
            else:
                cost = 1
            if num_batched_tokens + cost > self.max_num_batched_tokens:
                break
            
            batch.append(seq)
            num_batched_tokens += cost
        
        return batch
    
    def step(self):
        batch = self.schedule()
        if not batch:
            return False
        
        next_tokens = run_fake_model(
            batch,
            eos = self.eos
        )

        for seq, token_id in zip(batch, next_tokens):
            seq.append_token(token_id)

            if self.block_manager.can_allocate(seq):
                self.block_manager.allocate(seq)
            else:
                seq.status = SequenceStatus.FINISHED

            if token_id == self.eos and not seq.ignore_eos:
                seq.status = SequenceStatus.FINISHED

            if seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
            
        
        still_running = []
        for seg in self.running:
            if seq.is_finished:
                self.block_manager.free(seq)
                self.finished.append(seq)
            else:
                still_running.append(seq)
            
        self.running = still_running
        return True

    def has_unfinished_requests(self):
        return bool(self.waiting or self.running)

if __name__ == "__main__":
    cfg = config(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        max_model_len=128,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        eos=-1,
    )

    scheduler = Scheduler(cfg)

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        scheduler.add_request(
            token_ids,
            SamplingParams(max_tokens=8),
        )

    while scheduler.has_unfinished_requests():
        scheduler.step()

    for seq in scheduler.finished:
        print("seq_id:", seq.seq_id)
        print("prompt tokens:", seq.prompt_token_ids)
        print("completion tokens:", seq.completion_token_ids)
        print()    

