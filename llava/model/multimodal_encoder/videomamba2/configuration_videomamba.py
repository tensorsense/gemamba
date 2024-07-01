import os
from typing import Union
from transformers import PretrainedConfig
from transformers import BertConfig
from transformers import logging
import torch


logger = logging.get_logger(__name__)


class VideoMambaTextConfig(BertConfig):

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
        self.name = "bert_base"
        self.pretrained = "bert-base-uncased"
        self.config = (
            "llava/model/multimodal_encoder/videomamba/configs/config_bert.json"
        )
        self.d_model = 768
        self.fusion_layer = 9

        # bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
        # bert_config.encoder_width = (
        #     model_config.vision_encoder.get.d_model
        #     if model_config.vision_encoder.get("d_model", 0)
        #     else model_config.vision_encoder.embed_dim
        # )
        # bert_config.gradient_checkpointing = checkpoint
        # bert_config.fusion_layer = model_config.text_encoder.fusion_layer



        # config bert

        {
            "architectures": ["BertForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "fusion_layer": 9,
            "encoder_width": 768,
            "cross_module": "ca",
        }

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

    model_type = "videomamba_vision_model"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = torch.device("cpu")
        self.dtype = torch.float
        self.channels=3
        self.initializer_cfg = None
        self.add_pool_norm = True

        self.name = "videomamba_middle"
        self.img_size = 224
        self.patch_size = 16
        self.depth = 32
        self.embed_dim = 576
        self.drop_path_rate = 0.25
        self.ssm_cfg = None
        self.norm_epsilon = 1e-5
        self.fused_add_norm = True
        self.rms_norm = True
        self.residual_in_fp32 = True
        self.bimamba = True
        self.pool_type = "cls+avg"
        self.kernel_size = 1
        self.num_frames = 8
        self.ckpt_num_frame = 8
        self.use_checkpoint = False
        self.checkpoint_num = 0
        self.clip_decoder_embed_dim = 576
        self.clip_output_dim = 512
        self.clip_norm_type = "l2"
        self.clip_return_layer = 1
        self.clip_student_return_interval = 1
        # self.pretrained = model_pth
        self.clip_teacher = "none"
        self.clip_img_size = 224
        self.clip_return_interval = 1
        self.video_mask_type = "none"
        self.video_mask_ratio = 0.0
        self.video_double_mask_ratio = 0.0
        self.image_mask_type = "none"
        self.image_mask_ratio = 0.0
        self.image_double_mask_ratio = 0.0
        self.keep_temporal = True

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

    model_type = "videomamba"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        # projection_dim=512,
        # logit_scale_init_value=2.6592,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_config = VideoMambaTextConfig(**text_config)
        self.vision_config = VideoMambaVisionConfig(**vision_config)

        self.embed_dim = 512
        self.temp = 0.07

        # self.projection_dim = projection_dim
        # self.logit_scale_init_value = logit_scale_init_value
        # self.initializer_factor = 1.0
