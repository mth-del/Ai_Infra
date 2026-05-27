import torch
import math

def attention(query,key,value,mask=None,dropout=None):
    d_k = query.size(-1) 
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)

    if mask is  not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = scores.softmax(scores)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn