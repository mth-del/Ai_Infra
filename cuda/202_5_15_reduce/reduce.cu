dim3  blaock(BLOCK_SIZE);
dim3  grid_size(N,BLOCK_SIZE)

__global__ void reduce_v1(const float* input ){
    int idx = blockDim.x * blockIdx.x + threadIdx.x
    if(idx<N) atomicadd()
}
