import torch
import cuda.tile as ct
import time


@ct.kernel
def sum_v1(A:ct.Array, O:ct.Array, tile_size: ct.Constant):
    # how to split blocks
    # block mapping at A [M(block_x 1:1)),N]
    block_x = ct.bid(0)
    # after reshape tile_A is one dim 
    tile_A = ct.load(A, 
                    (block_x, 0), (1, tile_size), 
                    padding_mode = ct.PaddingMode.ZERO
                    ).reshape((tile_size,))
    tile_A = ct.sum(tile_A)
    ct.store(O,(block_x, ), tile_A)

@ct.kernel
def sum_v2(A:ct.Array, O:ct.Array, tile_size: ct.Constant):
    # how to split blocks
    # block mapping at A [M(block_x 1:1)),N(block_y 1:tile_size)]
    block_x, block_y = ct.bid(0), ct.bid(1)
    # after reshape tile_A is one dim 
    tile_A = ct.load(A, 
                    (block_x, block_y), (1, tile_size), 
                    padding_mode = ct.PaddingMode.ZERO
                    ).reshape((tile_size,))
    tile_A = ct.sum(tile_A)
    # 对输出 O[block_x] 做原子加
    ct.atomic_add(O,(block_x, ), tile_A)

@ct.kernel
def sum_v3(A: ct.Array,O: ct.Array, tileM: ct.Constant, tileN: ct.Constant):
    block_x = ct.bid(0)
    # after reshape tile_A is one dim 
    tile_A = ct.load(A, 
                    (block_x, 0), (tileM, tileN), 
                    padding_mode = ct.PaddingMode.ZERO
                    )
    tile_B = ct.full(shape=(tileN,tileM), fill_value=1.0, dtype=A.dtype)
    """
    tile_A:
    [[1,2]
     [3,4]]
    
    tile_B
    [[1,1]
     [1,1]]

    tile_sum
    [[3,3]
     [7,7]]   # 最后的结果在一列上
    """
    # tensorcore very fast
    tile_sum = ct.matmul(tile_A,tile_B)
    tile_sum = ct.extract(tile_sum, (0,0) , (tileM,1)).reshape((tileM,))
    ct.store(O, (block_x, ),tile_sum)


def bench_cuda_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return total_ms / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    M,N = 1048576, 32
    tile_size = 32
    tileM, tileN = 32, 32

    A = torch.randn(size=(M,N),  device="cuda", dtype=torch.float32)
    ref = torch.sum(A,dim=-1) #[M]


    O1 = torch.zeros(size=[M], device="cuda",dtype=torch.float32)
    ct.launch(torch.cuda.current_stream(),(M,),sum_v1,(A,O1,32))

    O2 = torch.zeros(size=[M], device="cuda",dtype=torch.float32)
    ct.launch(torch.cuda.current_stream(),(M,ct.cdiv(N,32)),sum_v2,(A,O2,32))

    O3 = torch.zeros(size=[M], device="cuda",dtype=torch.float32)
    ct.launch(torch.cuda.current_stream(),(ct.cdiv(M,32), ),sum_v3,(A,O3,32,32))   

    # launch 封装
    # if M >> N V1 is closer to V2 
    def run_v1():
        ct.launch(
            torch.cuda.current_stream(),
            (M,),                 # 每行一个block
            sum_v1,
            (A, O1, 1048576),
        )
    
    # N is bigger than M
    def run_v2():
        O2.zero_()               # 避免atomic_add叠加历史值
        ct.launch(
            torch.cuda.current_stream(),
            (M, ct.cdiv(N, tile_size)),
            sum_v2,
            (A, O2, tile_size),
        )
    # M is bigger than N
    def run_v3():
        O3.zero_()               # 避免atomic_add叠加历史值
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(M, tileM),),
            sum_v3,
            (A, O3, tileM, tileN),
        )
    def run_torch():
        _ = torch.sum(A, dim=-1)
    

    # 正确性（先单次跑）
    run_v1(); torch.cuda.synchronize()
    run_v2(); torch.cuda.synchronize()
    run_v3(); torch.cuda.synchronize()

    print("v1 max err:", (O1 - ref).abs().max().item())
    print("v2 max err:", (O2 - ref).abs().max().item())
    print("v3 max err:", (O3 - ref).abs().max().item())

    # 性能
    t_v1 = bench_cuda_ms(run_v1)
    t_v2 = bench_cuda_ms(run_v2)
    t_v2 = bench_cuda_ms(run_v3)
    t_th = bench_cuda_ms(run_torch)


    print(f"sum_v1:   {t_v1:.4f} ms")
    print(f"sum_v2:   {t_v2:.4f} ms")
    print(f"sum_v3:   {t_v2:.4f} ms")
    print(f"torch.sum:{t_th:.4f} ms")  