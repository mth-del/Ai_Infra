MHA（muti-Head Attention多头注意力机制）
独立的query,key,Value
https://zhuanlan.zhihu.com/p/1890128241
![alt text](Snipaste_2026-04-25_13-59-52.png)

「MHA 就是把 hidden 维拆成多个子空间，每个子空间各自做 softmax 注意力，再把结果拼接并线性投影；目的是让模型并行地从不同角度聚合上下文信息。」

Attention(Q, K, V) = softmax(Q·K^T/√d_k)·V
如果不进行缩放，当d_k较大时，点积的结果可能会变得非常大，这会导致在应用softmax函数时产生的梯度非常小



# MHA
![alt text](Snipaste_2026-04-25_16-44-35.png)
1. 把 (Q,K,V) 线性投影成多组 ((Q_i, K_i, V_i))，每一组做一遍 Scaled Dot-Product Attention，得到一份「上下文表示」。
2. 单头 attention 的公式不变
3. 多头再拼起来，再线性融合

batch_size = hidden_states.size(0)
size(0) = 第 0 维长度
如果 hidden_states 形状是常见的 [B, T, C]（batch, seq_len, hidden_dim），那它取到的就是 B
在 MHA 里后面通常会用 batch_size 去做 view/reshape，比如把 [B, T, C] 拆成 [B, num_heads, T, head_dim]。

hidden_dim（也常写 d_model）就是每个 token 的特征向量长度，也叫“隐藏层维度”
* batch：一次多少条样本
* seq_len：每条样本多少个 token
* hidden_dim：每个 token 用多少维数字表示


x = hidden.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
把输入从 [B, T, hidden_dim] 变成多头注意力常用的 [B, H, T, D]。
* 把最后一维 hidden_dim 拆成 num_heads * head_dim


embed_dim 就是 embedding 向量的维度，通常也等同于 Transformer 里的 hidden_dim / d_model


MQA是MHA的一种变体，也是用于自回归解码的一种注意力机制。
MQA 让所有的Head之间共享同样的一份 K 和 V 矩阵（意味K和V的计算唯一），只让 Q 保留了原始多头的性质
