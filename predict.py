import os
import sys
from time import time
from pathlib import Path

sys.path.append(Path(".").resolve().as_posix())

import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
)

# These imports are necessary to run the code and register models in the autoclass
from llava.model import (
    LlavaGemmaForCausalLM,
    LlavaGemmaConfig,
)

from flask import Flask, request, jsonify

app = Flask(__name__)


# CHECKPOINT_PATH = "checkpoints/llava_gemma_mamba_v18_adapter_vcgpt"
CHECKPOINT_PATH = "checkpoints/llava_gemma_mamba_v26_adapter25M_ft"
CONV_MODE = "gemma"
DEFAULT_DEVICE = torch.device("cuda")
DEFAULT_DTYPE = torch.bfloat16

disable_torch_init()
model_path = os.path.expanduser(CHECKPOINT_PATH)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)

tokenizer.pad_token = "<pad>"
model.config.tokenizer_padding_side = "left"
model.get_model().to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
model.get_model().mm_projector.to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)


@app.route("/predict", methods=["POST"])
def predict():
    json_input = request.get_json(force=True)

    temperature = json_input.get("temperature", 0.1)
    max_new_tokens = json_input.get("max_new_tokens", 128)

    # 1. Parse input prompts
    inputs = json_input["inputs"]
    texts = [i["text_prompt"] for i in inputs]
    video_paths = [i["video_path"] for i in inputs]
    assert len(texts) == len(
        video_paths
    ), "Found a mismatch between the number of text prompts and videos"

    # 2. Prepare batched inputs
    text_inputs = _prepare_text_batch(texts)
    video_tensor = _prepare_video_batch(video_paths)

    # 3. Run the inference
    start_time = time()
    with torch.inference_mode():  # , torch.amp.autocast(DEFAULT_DEVICE.type):
        output_ids = model.generate(
            **text_inputs,
            images=video_tensor,
            # image_sizes=[image.size],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    proctime = time() - start_time
    ntokens = len(output_ids[0])

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return jsonify(
        {"predicted_texts": outputs, "proctime": proctime, "ntokens": ntokens}
    )


def _build_prompt(text):
    # insert special image tokens into the text prompt
    text = f"{DEFAULT_IMAGE_TOKEN}\n{text}"

    # construct conversation
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def _prepare_text_batch(texts):
    prompts = []
    for text in texts:
        prompts.append(_build_prompt(text))

    # # tokenize the prompt
    inputs = tokenizer_image_token(
        prompts, tokenizer, padding_side="left", image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
    )

    for k, v in inputs.items():
        inputs[k] = v.to(DEFAULT_DEVICE)

    return inputs


def _prepare_video_batch(video_paths):
    video_tensor = image_processor(video_paths, return_tensors="pt")["pixel_values"].to(
        DEFAULT_DEVICE, dtype=DEFAULT_DTYPE
    )
    return video_tensor


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Listen on all interfaces
