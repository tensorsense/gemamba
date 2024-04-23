FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN pip3 install ninja \
 && pip3 install -v git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

#  ROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# # install python
# RUN apt update && apt install -y \
#     python3 \
#     python3-pip \
#     python3-venv \
#     && rm -rf /var/lib/apt/lists/*

# # install tmux
# RUN apt update && apt install -y tmux

# # update apt and install packages
# RUN apt update && \
#     apt install -yq \
#         ffmpeg \
#         dkms \
#         build-essential

# # add user tools
# RUN sudo apt install -yq \
#         jq \
#         jp \
#         tree \
#         tldr

# # # Setup conda
# # RUN cd /tmp && \
# #     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
# #     bash ./Miniconda3-latest-Linux-x86_64.sh -b && \
# #     rm ./Miniconda3-latest-Linux-x86_64.sh