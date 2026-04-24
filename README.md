# ai_infra
ai infra实际过程

# 学术资源加速
source /etc/network_turbo


# 取消学术加速，如果不再需要建议关闭学术加速，因为该加速可能对正常网络造成一定影响
unset http_proxy && unset https_proxy


# nsys使用
nsys profile --stats=true python  /root/mth/code_space/ai_infra/cutitle/reduce.py

# 导入子模块
TORCH_INC=$(python -c "from torch.utils.cpp_extension import include_paths; print(include_paths()[0])")
TORCH_API_INC=$(python -c "from torch.utils.cpp_extension import include_paths; print(include_paths()[1])")
python -c "import sysconfig; print(sysconfig.get_paths()['include'])"
PY_INC=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")

nvcc -std=c++17 -c cuda/2026_4_24_rms_norm/rms_norm.cu -o /tmp/rms_norm.o \
  -I"$TORCH_INC" \
  -I"$TORCH_API_INC" \
  -I"$PY_INC"

