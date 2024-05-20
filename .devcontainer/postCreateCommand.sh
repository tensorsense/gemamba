# install tmux
apt update
apt install -y tmux

# install Python packages
pip3 install -r requirements.txt
pip3 install git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d

pip3 install -e llava/model/multimodal_encoder/videomamba/causal-conv1d
pip3 install -e llava/model/multimodal_encoder/videomamba/mamba
pip3 install git+https://github.com/huggingface/transformers  # Phi3 is not in a pip release yet

# fix async_io warning
apt install -y libaio-dev

# fix cutlass warning
git clone https://github.com/NVIDIA/cutlass.git ~/cutlass

# download pretrained weights for videomamba
# wget https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_m16_25M_f8_res224.pth

nvidia-smi

