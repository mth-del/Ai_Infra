import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0,d_model,2)* -(math.log(10000) / d_model)
        )
        # 偶数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        #  奇数维度
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self,x):
        x = x + self.pe[:,: x.size(1)].requires_grad_(False)

        return self.dropout(x)
