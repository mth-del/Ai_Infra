# nvcc required. Override: make CUDA_ARCH=sm_89
CUDA_ARCH ?= sm_75
NVCC = nvcc
FLAGS = -O3 -std=c++17 --use_fast_math -arch=$(CUDA_ARCH)

TARGETS = reduce_warp_shuffle reduce_2d_cols gemm_optimized layernorm softmax online_softmax

all: $(TARGETS)

reduce_warp_shuffle: reduce_warp_shuffle.cu
	$(NVCC) $(FLAGS) -o $@ $<

reduce_2d_cols: reduce_2d_cols.cu
	$(NVCC) $(FLAGS) -o $@ $<

gemm_optimized: gemm_optimized.cu
	$(NVCC) $(FLAGS) -o $@ $<

layernorm: layernorm.cu
	$(NVCC) $(FLAGS) -o $@ $<

softmax: softmax.cu
	$(NVCC) $(FLAGS) -o $@ $<

online_softmax: online_softmax.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
