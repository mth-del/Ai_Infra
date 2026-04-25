import torch 
import torch.nn as nn

class MultiQuerySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiQuerySelfAttention,self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)
        # k和v只做投影一份
        self.wk = nn.Linear(embed_dim, self.head_dim)
        self.wv = nn.Linear(embed_dim, self.head_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

    def q_h_split(self , hidden, head_num=None):
        # 取hidden的前两个维度
        batch_size, seq_len = hidden.size()[:2]

        # q进行拆分多头
        if head_num == None:
            x = hidden.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            return x
        else:
            return hidden.view(batch_size, seq_len, head_num, self.head_dim).transpose(1, 2)
    
    def forward(self, hidden_states, mask=None):
        batch_size = hidden_states.size(0)

        # 线性变换
        q = self.wq(hidden_states)
        k = self.wk(hidden_states)
        v = self.wv(hidden_states)

        # 多头切分
        q = self.q_h_split(q)
        k = self.q_h_split(k,1)
        v = self.q_h_split(v,1)

        # 注意力机制
        scores = torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))
        print("scores", scores.shape)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores,dim=-1)
        output = torch.matmul(attention,v)
        # seq_len, num_heads, head_dim
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.head_dim)
        # 线性变换
        output = self.wo(output)
        return output

x = torch.rand(3,12,512)
atten = MultiQuerySelfAttention(512,8)
y = atten(x)
print(y.shape)
