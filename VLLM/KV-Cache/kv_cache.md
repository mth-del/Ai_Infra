vllm prefix cache:https://zhuanlan.zhihu.com/p/1896927732027335111
vllm pd分离：https://zhuanlan.zhihu.com/p/1906741007606878764


 KV cache group：
 为了适配异构形态、或者不同Attention模块（MHA/MLA/GQA/slide等）的kv cache混合使用设计了kv cache group概念，对应的block size可以设置不同，用不同的block table来管理每个KV cache group的映射关系。不同KV cache group之间协同工作由KVCacheCoordinator管理。

DCP（Decode Context Parallel）
为了减少KV cache的冗余存储，开启Attention序列并行时，可根据CP的数量让不同设备存储KV cache的部分数据。

当前计算逻辑单个token的存储位置满足: rank_id = token_idx % cp_world_size。


一句话版
MHA：每个头都有独立的 Q/K/V，表达力最强，但 KV Cache 最大、解码带宽压力最高。
GQA：多个 Q 头共享一组 K/V（按组共享），在效果和成本之间做折中。
MLA：把 K/V 先压到低秩潜空间再用于注意力，进一步降 KV Cache 和带宽，适合长上下文推理。
Sliding Window（如果你说的 slide 是这个）：注意力只看最近窗口 token，显存和计算从全局变局部，长序列更可控。
面试对比版（20 秒）
“如果按推理成本看：MHA 最贵、质量最好；GQA 通过 K/V 组共享把成本降下来，质量通常接近 MHA；MLA 进一步做低秩压缩，长上下文下省显存更明显；Sliding Window 则直接限制可见范围，用局部注意力换取线性可扩展。”
