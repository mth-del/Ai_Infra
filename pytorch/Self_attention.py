import torch
from torch import nn
import math
class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v] b后者的k和v都是encoder的输出，所以k和v的形状总是相同的
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attention = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attention, V)

        return context, attention