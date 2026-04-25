from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embe_dim):
        super(SelfAttention, self).__init__()
        self.dim = embe_dim;
        self.head_dim = embe_dim;

        # 定义三个权重矩阵
        self.Wq = nn.Linear(embe_dim, self.head_dim) # Q投影
        self.Wk = nn.Linear(embe_dim, self.head_dim) # Q投影
        self.Wv = nn.Linear(embe_dim, self.head_dim) # Q投影

    def forward(self, inputs):
        # 输入形状（batch_size, seq_len, embed_dim）
        batch_size,seq_len, _ = inputs.shape

        # 计算Q/K/V矩阵形状
        Q = self.Wq(inputs)
        K = self.Wk(inputs)
        V = self.Wv(inputs)

        # Q*K^
        attention_score = torch.matmul(Q,K.transpose(-1, -2))

        # 计算注意力权重，维度沿最后一个轴
        attention_weights = F.softmax(attention_score, dim=-1)

        # 计算输出
        outputs = torch.matmul(attention_weights, V)
        return outputs

x = torch.randn(2,3,4)
self_attention = SelfAttention(embe_dim=4)
output = self_attention(x);
print("output shapes",output.shape)