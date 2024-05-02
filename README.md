# Gemamba

This repository contains training code for the Gemamba multimodal language model.

Gemamba is the first multimodal LLM to combine a Mamba-based video encoder with performant and flexible [Gemma](https://huggingface.co/google/gemma-2b-it) transformer LLM in a LLaVA-style architecture.

## Getting started

We recommend using [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) to create the environment using pre-made configuration.

1. Install [PyTorch](https://pytorch.org).

2. Install Python dependencies.

```bash
pip3 install -r requirements.txt
```

Install VideoMamba dependencies:

```bash
pip3 install -e llava/model/multimodal_encoder/videomamba/causal-conv1d
pip3 install -e llava/model/multimodal_encoder/videomamba/mamba
```

\[optional\] Update transformers to get Phi3 support:

```bash
pip3 install git+https://github.com/huggingface/transformers
```

3. Download pretrained weights for VideoMamba:

```bash
wget https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_m16_25M_f8_res224.pth
```

4. Refer to `run_finetune.ipynb` to learn how to load a checkpoint and run inference.

## Pretrained checkpoints

Pretrained checkpoint for the model can be found here: [HF 🤗](https://huggingface.co/TensorSenseAI/gemamba-v0).

- The model's projector has been pretrained for 1 epoch on the [Valley](https://github.com/RupertLuo/Valley) dataset.
- LLM and the projector have been jointly fine-tuned using the [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/docs/train_video_chatgpt.md) dataset.

## Training

We inherit most of the training workflow from the original LLaVA. Please refer to `scripts/train` to see configurations used for training the model. See `scripts/eval` for scripts used to calculate benchmark scores.