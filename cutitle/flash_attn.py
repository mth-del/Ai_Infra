import torch
import cuda.tile as ct



@ct.kernel
def flash_attn(Q:ct.Array, K:ct.Array, V:ct.Array, O:ct.Array,
                tileS:ct.Constant, tileD:ct.Constant):
                 
    # block mapping on O[B(block_z 1:1), S(block_x 1:tilees)， H(block_y 1:1)， D]
    block_x, block_y,block_z = ct.bid(0),  ct.bid(1) , ct.bid(2)
    tile_Q = ct.load(Q, (block_z, block_x, block_y, 0), (1, tileS, 1, tileD)).reshape((tileS, tileD))

    accumlator = ct.full((tileS,tileD), 0.0, dtype=ct.float32)
    for seq_iter in range(ct.cdiv(K.shape[1], tileS)):
        tile_K = ct.load(K, (block_z, block_x, block_y, 0), (1, tileS, 1, tileD)).reshape((tileS, tileD))
        tile_K = ct.transpose(tile_K, 0, 1)

        tile_QK = ct.matmul(tile_Q, tile_K) # [tileS, tileS]
        tile_V = ct.load(V, (block_z, block_x, block_y, 0), (1, tileS, 1, tileD)).reshape((tileS, tileD))

        accumlator = ct.mma(tile_QK, tile_V, accumlator)
        
    accumlator = accumlator.reshape((1,tileS,1, tileD))
    ct.store(O,(block_z,block_x,block_y,0), accumlator)
# Q =  [batch size, sequence size, num of head, head dim]

B, S, H ,D = 1, 128, 8, 64
Q = torch.randn(size=[B, S, H, D], device = "cuda", dtype = torch.float32)
K = torch.randn(size=[B, S, H, D], device = "cuda", dtype = torch.float32)
V = torch.randn(size=[B, S, H, D], device = "cuda", dtype = torch.float32)
O = torch.empty_like(Q)

_Q = Q.permute(0, 2, 1, 3)
_K = K.permute(0, 2, 1, 3)
_V = V.permute(0, 2, 1, 3)


real = torch.softmax(_Q @ _K.transpose(-1,-2), dim=1)@ _V
real = real.permute(0, 2, 1, 3)

ref = _Q @ _V.transpose(-1,-2) @ _V
ref = ref.permute(0, 2, 1, 3)

ct.launch(torch.cuda.current_stream(),
        (ct.cdiv(S,32),H,B),
        flash_attn,
        (Q,K,V,O,32,D)
)

print(ref - O)       
        


