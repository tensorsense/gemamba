import os
from typing import Union
from transformers import PretrainedConfig

num_frames = 8
img_size = 224
batch_size = 64
max_txt_l = 32

# model_pth = "videomamba_m16_5M_f8_res224.pth"
model_pth = "videomamba_m16_25M_f8_res224.pth"

config_dict = {
    # preprocessor settings
    "num_frames": num_frames,
    "num_frames_test": num_frames,
    "batch_size": batch_size,
    "max_txt_l": max_txt_l,
    "inputs": {
        "image_res": img_size,
        "video_input": {
            "num_frames": num_frames,
            "sample_type": "rand",
            "num_frames_test": num_frames,
            "sample_type_test": "middle",
            "random_aug": False,
        },
        "max_txt_l": {"image": max_txt_l, "video": max_txt_l},
        "batch_size": {"image": batch_size, "video": batch_size},
        "batch_size_test": {"image": batch_size, "video": batch_size},
    },
    # model settings
    "text_enc": "bert",
    "model": {
        "model_cls": UMT_VIDEOMAMBA,
        "multimodal": {"enable": True},
    },
    # train test settings
    "evaluate": False,
    "deep_fusion": False,
    "evaluation": {
        "eval_frame_ensemble": "concat",
        "eval_x_only": False,
        "k_test": 128,
        "eval_offload": False,
    },
    # technicalities
    "fp16": True,
    "bf16": True,
    "gradient_checkpointing": True,
    "device": "cuda",
    "mode": "pt",
    "output_dir": None,
    "resume": False,
    "debug": False,
    "log_freq": 1,
    "seed": 42,
    "zero_shot": True,
    "save_latest": False,
    "auto_resume": False,
    # "pretrained_path": model_pth,
    "distributed": False,
}


class VideoMambaTextConfig(PretrainedConfig):
    """
    "text_encoder": {
        "name": "bert_base",
        "pretrained": "bert-base-uncased",
        "config": "llava/model/multimodal_encoder/videomamba/configs/config_bert.json",
        "d_model": 768,
        "fusion_layer": 9,
    },
    """

    model_type = "videomamba_text_model"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        # get the text config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["text_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class VideoMambaVisionConfig(PretrainedConfig):
    """
    "vision_encoder": {
        "name": "videomamba_middle",
        "img_size": img_size,
        "patch_size": 16,
        "depth": 32,
        "embed_dim": 576,
        "drop_path_rate": 0.25,
        "ssm_cfg": None,
        "norm_epsilon": 1e-5,
        "fused_add_norm": True,
        "rms_norm": True,
        "residual_in_fp32": True,
        "bimamba": True,
        "pool_type": "cls+avg",
        "kernel_size": 1,
        "num_frames": num_frames,
        "ckpt_num_frame": 8,
        "use_checkpoint": False,
        "checkpoint_num": 0,
        "clip_decoder_embed_dim": 576,
        "clip_output_dim": 512,
        "clip_norm_type": "l2",
        "clip_return_layer": 1,
        "clip_student_return_interval": 1,
        "pretrained": model_pth,
        "clip_teacher": "none",
        "clip_img_size": img_size,
        "clip_return_interval": 1,
        "video_mask_type": "none",
        "video_mask_ratio": 0.0,
        "video_double_mask_ratio": 0.0,
        "image_mask_type": "none",
        "image_mask_ratio": 0.0,
        "image_double_mask_ratio": 0.0,
        "keep_temporal": True,
    },
    """

    model_type = "videomamba_vision_model"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class VideoMambaConfig(PretrainedConfig):
    """
    "embed_dim": 512,
    "temp": 0.07,
    """

    model_type = "videomamba"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_config = VideoMambaTextConfig(**text_config)
        self.vision_config = VideoMambaVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
