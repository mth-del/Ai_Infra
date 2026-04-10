import cuda.tile as ct
import torch



@ct.kernel
def rmsnorm(x:ct.Array, w:ct.Array, o:ct.Array, tile_size:ct.Constant,dim: int,eps:float):
    block_x = ct.bid(0)
    # paddingMode=zero funciton is add 0 to complete the tile
    # block_x is the startpoint of 0 dim ; 0 is the startpoint of 1 dim
    tile_x = ct.load(x,(block_x,0),(1,tile_size),padding_mode=ct.PaddingMode.ZERO)
    tile_w = ct.load(w,(0, ),(tile_size, ),padding_mode=ct.PaddingMode.ZERO)

    tile_x = tile_x.astype(ct.float32)
    # in cuda compute 1/dim is faster than /dim
    tile_var = ct.sum(tile_x * tile_x * (1/dim))
    # rsqrt is sqrt and then inverse  and eps prevent division by zero
    tile_rsqrt = ct.rsqrt(tile_var + eps)

    tile_x = tile_x * tile_rsqrt
    tile_x = tile_x * tile_w


    ct.store(o,(block_x,0),tile_x.astype(o.dtype))


M, N = 128, 1024
X = torch.randn(size=[M,N], device = "cuda", dtype=torch.bfloat16)
W = torch.randn(size=[N], device = "cuda", dtype=torch.float32)
O = torch.empty_like(X)

# reference function
real = torch.nn.functional.rms_norm(X, normalized_shape=[N], weight=W,eps=1e-7)
ct.launch(torch.cuda.current_stream(), (M,), rmsnorm, (X,W,O,1024,1024,1e-7))
print(real - O)