# 查看 CUDA 安装位置
ls /usr/local/ | grep cuda

# 设置环境变量（假设是 cuda-12.1，根据实际情况调整）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 再次安装 flash-attn
pip install flash-attn --no-build-isolation
