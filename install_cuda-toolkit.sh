# 1. 下载并安装 CUDA 仓库密钥
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 2. 更新包列表
sudo apt update

# 3. 安装 CUDA Toolkit 12.1（与 PyTorch 2.4 兼容）
sudo apt install cuda-toolkit-12-1 -y

# 4. 设置环境变量（添加到 ~/.bashrc）
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 5. 立即生效
source ~/.bashrc

# 6. 验证安装
nvcc --version