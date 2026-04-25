from turtle import forward
from numpy import dtype
import torch
import torch.nn as nn

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GroupQueryAttention,self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.group_num = 4
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, self.group_num * self.head_dim)
        self.wv = nn.Linear(embed_dim, self.group_num * self.head_dim)
        
        self.wo = nn.Linear(embed_dim, embed_dim)
    def split(self,hidden,group_num = None):
        batch_size, seq_len = hidden.size()[:2]
        if group_num == None:
            x = hidden.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
            return x
        else:
            x =  hidden.view(batch_size,seq_len,group_num,self.head_dim)
            # hidden: [B, T, group_num * D]
            # x:[B,T,G,D]“每个 batch 里、每个 token 上，有 G 组向量，每组长度 D
            x = x.permute(0,2,1,3)
            x = x[:,:,None,:,:]
            x = x.expand(batch_size,group_num,self.num_heads // group_num, seq_len, self.head_dim)
            x = x.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            return x
    def forward(self, hidden_states, mask=None):
        batch_size = hidden_states.size(0)
        
        # 线性变换
        q = self.wq(hidden_states)
        k = self.wk(hidden_states)
        v = self.wv(hidden_states)

        # 多头切分
        q = self.split(q)
        k = self.split(k,self.group_num)
        v = self.split(v,self.group_num)

        # 注意力计算
        scores = torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype = torch.float32))
        print("scores:", scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0 ,float('-inf'))

        attention = torch.softmax(scores,dim=-1)
        output = torch.matmul(attention, v)

        # 合并多头
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.head_dim)

        # 线性变换
        output = self.wo(output)

        return output
# B=3, T=12, C=512]。
x =torch.ones(3,12,512)
atten = GroupQueryAttention(512,8)
y = atten(x)
print(y.shape)