#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
// 
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

//  A = [M,K]
// B =[K,N]
// C = A * B = [M,N]
__global__  void segmm_native_f32_kernel(float* a, float* b, float* c, int M, int N, int K)
{
    // 这个需要形象解释一下
    // 每一个c[m][n] 对应一个线程
    int m = blockIdx.x * blockDim.x + threadIdx.x;  // 第m行
    int n = blockIdx.y * blockDim.y + threadIdx.y;  // 第n列

    if(m < M && n < N){
        float p_sum = 0.0f;
        for(int k= 0 ; k < K ;k++){
            // 不太理解为什么这样
            // 需要确定目前计算的在a中的位置
            p_sum += a[m*K +k] * b[k*N + n];
        }
        c[m * N + n] = p_sum; // c[m,n]
    }
}

// summary:原始地址；在block中的地址；写入shared_memory;写入global_memory
template<const int BM =32, const int BN = 32 , const int BK =32>
__global__ void segmm_sliced_f32_kernel(float* a, float* b, float* c, int M, int N, int K){
    // 一个block处理一块计算
    // 使用shared memory 将K进行分块

    __shared__ float s_a[BM][BK], s_b[BK][BN];

    // 当前 block 在 grid 中的 x 编号（对应 C 矩阵的列方向 block）
    int bx = blockIdx.x;
    // 当前 block 在 grid 中的 y 编号（对应 C 矩阵的行方向 block）
    int by = blockIdx.y;

    // 假设 block = （16，16） --- 压缩到一维：256
    /*
        （0，0）
        （0，1）
          ...
         (0,15)
    */
    // 当前线程在 block 内的 x 编号（0 ~ blockDim.x-1）
    int tx = threadIdx.x;
    // 当前线程在 block 内的 y 编号（0 ~ blockDim.y-1）
    int ty = threadIdx.y;
    // blockDim.x表示就是列的维度,二维坐标是（ty,tx）->tid = ty*blockDim.x + tx
    int tid = threadIdx.y * blockDim.x + tx;


    int load_smem_a_m = tid / BM;
    int load_smem_a_k = tid % BK;
    int load_smem_b_k = tid / BK;
    int load_smem_b_n = tid % BN;

    // global memory grid 
    /*
    set grid = (16,16)
    bx = blockIdx.x  // 列
    by = blockIdx.y  // 行

    by*BM + load_smen_a_m -> 第几个block+第几个thread
    */
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;


    float sum = 0.0f;


    for(int bk = 0 ; bk < (K+BK -1)/BK; bk++){
        // A[行][列] = a[行*K + 列] 行先存储
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        // 计算线程在A中的全局地址，加载到shared_memory
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];

        // 计算线程在B的全局地址
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];

        // 等待线程同步
        __syncthreads();
#pragma unroll
        for(int k = 0 ; k < K ; k++){
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;

            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }

    // 把结果写回来
    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = store_gmem_c_m * N + load_gmem_b_n;
    c[store_gmem_c_addr] = sum;
}



// 增加计算密度
template<const int BM = 128, const int BN = 128, const int BK = 8,const int TM = 8, const int TN = 8>

__global__ void sgemm_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,
                                                    int M, int N, int K){
    // block Tile :16*16 上处理c上大小为128*128的目标块
    // thread Tile :每个目标块负责计算8*8个元素
    // K tile: 将K进行分块，每块BK大小，迭代
    // vectorize:减少load和store指令，使用float4

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    
    int tid = threadIdx.y * blockDim.x + tx;   // tid within the block
    __shared__ float s_a[BM][BK], s_b[BK][BN]; // 2*128*8*4=8KB
    
    
    /******[0]计算shred_memory中的索引******/
    // 明确shared_memoy是怎么存储的s_a[BM][BK] = s_a[128][8];s_b[BK][BN] = s_b[8][128]

    // 计算索引：s_a每行8个，每个线程读取4个，需要两个线程；128行需要256个线程
    // 决定加载那一行
    int load_smem_a_m = tid / 2;
    // 决定加载这一行的前半段还是后半段
    int load_smem_a_k = (tid % 2 == 0 )? 0 : 4;


    // 对于s_b
    // 128 / 32 = 4 每个线程读取4个需要32个线程 按B行主序 一共8行需要 32*8=256个线程
    // 256 / 32 = 8（0～7）
    int load_smem_b_k = tid / 32;
    // 256 % 32*4 = 0 4 .... 124
    int load_smem_b_n = (tid % 32)*4;

    /*******[1]计算全局内存中的索引**********/
    // 全局的a和c的行
    int load_gmem_a_m = by * BM + load_smem_a_m;
    // 全局的a和b的列
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    float r_c[TM][TN] ={0.0};
    /*******[2]先对k进行分块，每块BK大小**********/
    for(int bk = 0 ; bk < (K+BK -1) / BK ; bk++){
        /**1.加载数据到共享内存s_a**/
        // a全局的列
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * N + load_gmem_a_k;
        FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
        /*2.加载数据到共享内存s_b***/
        // b全局的行
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * K + load_smem_b_n;
        FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
        /*3.线程同步*/
        __syncthreads();
        /*4.每个线程负责计算BM*BN中的TM*TN个元素*/
#pragma unroll
        for(int k =0 ; k < BK ; k++){
#pragma unroll
            for(int m = 0 ; m < TM ; m++){
#pragma unroll 
                for(int n = 0 ; n < TN ; n++){
                    // k 0~7 ty and tx range from 0 to 15, 16*8=128
                    // 128*8 128/TM(8) = 16 M方向
                    int comp_smem_a_m = ty * TM + m;
                    // 8*128 128/TN(8) = 16 N方向  
                    int comp_smem_b_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }


    /*******[3]每个线程负责计算BM*BN（12x128）中的TM*TN = 8*8个元素**********/
#pragma unroll
    for(int m = 0; m < TM; ++m){
        int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}