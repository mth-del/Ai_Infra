import cuda.tile as ct
import torch


@ct.kernel
def matmul(
    A: ct.Array, B: ct.Array, O: ct.Array,
    tileM: ct.Constant, tileN: ct.Constant, tileK: ct.Constant,
    transposeA: ct.Constant, transposeB: ct.Constant
):
    """
    ### stanrd matmul

    Q = A * B

    ### input layout
    A: [M , k]
    B: [K, N]
    Q; [M,N]

    ### block mapping at matix O
    [(block_x, 1:tileM),(block_y, 1:tileN)]

    ### parameter:
    transposeA: A 是否需要专置
    transposeB: B 是都需要专置
    """
    block_idx_x, block_idx_y = ct.bid(0),ct.bid(1)
    # if don't tanspose the shape is rhe last dim 
    K = A.shape[0] if transposeA else A.shape[-1]
    # 向上取整的一个数
    num_iter_k = ct.cdiv(K, tileK)

    accumulator = ct.full(shape=(tileM, tileN), fill_value = 0.0, dtype=ct.float32)
    for iter_k in range(num_iter_k):
        # if transpose order = F else order  = C
        tile_A = ct.load(A,(block_idx_x, iter_k),(tileM,tileK),order ="F" if transposeA else "C")
        tile_B = ct.load(B,(iter_k, block_idx_y),(tileK,tileN),order ="F" if transposeB else "C")   

        accumulator = ct.mma(tile_A, tile_B, accumulator)
    
    ct.store(O, (block_idx_x,block_idx_y), accumulator.astype(O.dtype))



if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    M, K, N = 128, 96, 64
    tileM, tileN, tileK = 32, 32, 32
    transposeA, transposeB = False, False

    A = torch.randn((M,K), device = device, dtype = torch.float32)
    B = torch.randn((K,N), device = device, dtype = torch.float32)
    O = torch.empty((M,N), device = device, dtype = torch.float32)

    grid = (ct.cdiv(M, tileM), ct.cdiv(N, tileN))

    ct.launch(
            torch.cuda.current_stream(),
            grid,
            matmul,
            (A,B,O,tileM,tileN,tileK,transposeA,transposeB),
    )

    ref = A @ B

    max_abs_error =  (0 - ref).abs().max().item()
    mean_abs_error =  (0 - ref).abs().mean().item()

    print(f"max_abs_error = {max_abs_error:.6f}, mean_abs_error = {mean_abs_error:.6f}")