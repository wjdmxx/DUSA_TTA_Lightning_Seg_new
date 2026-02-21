#!/bin/bash

# 定义目标环境绝对路径
ENV_PATH="/home/qingju/data/liyuanxi/envs/DUSA_Lightning_Seg_mm"

echo "===================================================="
echo "  [1/7] 正在清理并创建 Python 3.10 隔离环境..."
echo "===================================================="
conda env remove -p $ENV_PATH -y
rm -rf $ENV_PATH
conda create -p $ENV_PATH python=3.10 -y

# 激活环境
eval "$(conda shell.bash hook)"
conda activate $ENV_PATH

echo "===================================================="
echo "  [2/7] 配置基础构建工具与 Numpy..."
echo "===================================================="
# 锁死底层依赖，防止新版触发 C++ ABI 崩溃
pip install "setuptools<66.0.0" wheel pip --upgrade
pip install "numpy<2.0.0"

echo "===================================================="
echo "  [3/7] 安装 PyTorch 2.4.0..."
echo "===================================================="
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "===================================================="
echo "  [4/7] 注入 NVIDIA H20 硬件底层修复补丁..."
echo "===================================================="
# 替换默认 cuBLAS，彻底解决 H20 上运行 SD3 时的 core dumped 浮点异常
pip install nvidia-cublas-cu12==12.5.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "===================================================="
echo "  [5/7] 安装 OpenMMLab 体系..."
echo "===================================================="
pip install mmengine
# 精准拉取匹配 Torch 2.4.0 的 MMCV 2.2.0 官方预编译轮子，彻底告别源码编译报错
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
pip install "mmsegmentation>=1.0.0"

echo "===================================================="
echo "  [6/7] 安装 Hugging Face 体系..."
echo "===================================================="
# 完美对齐你验证过的高版本生态
pip install accelerate==1.12.0 diffusers==0.36.0 transformers==4.57.5 sentencepiece protobuf peft

echo "===================================================="
echo "  [7/7] 安装辅助开发工具与安全的 OpenCV..."
echo "===================================================="
pip install hydra-core omegaconf wandb Pillow einops rich tqdm scipy matplotlib
# 锁定 OpenCV 版本，防止其悄悄拉取 Numpy 2.x 破坏底层环境
pip install "opencv-python==4.9.0.80" "opencv-python-headless==4.9.0.80"

echo "===================================================="
echo "  环境配置全部完成！"
echo "  环境路径: $ENV_PATH"
echo "===================================================="