import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # 输入一样的维度
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

    def mh_split(self, hidden):
        batch_size = hidden.shape[0];
        x = hidden.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        return x
    
    def forward(self, hidden_states, mask=None):

        batch_size = hidden_states.size(0)
        # 线性变换
        q,k,v = self.wq (hidden_states), self.wk(hidden_states),self.wv(hidden_states)

        # 多头切分
        q, k,v = self.mh_split(q), self.mh_split(k), self.mh_split(v)

        # 注意力计算
        scores = torch.matmul(q,k.transpose(-2,-1) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        output  = torch.matmul(attention,v)

        # 拼接多头
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads * self.head_dim)

        # 线性变换
        output = self.wo(output)
        return output
    
x = torch.rand(2,3,36)
print(x)
output = MultiHeadAttention(36,6)
# 关键点：这是调用 nn.Module.__call__，再自动进入 forward(x, mask=None)。
y = output(x)
print(y.shape)


