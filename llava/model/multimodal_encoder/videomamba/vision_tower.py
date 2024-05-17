import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .hf_parts.processing_videomamba import VideoMambaVideoProcessor
from .models.umt_videomamba import UMT_VIDEOMAMBA
from .models.backbones.bert.tokenization_bert import BertTokenizer
from .utils.easydict import EasyDict
from .models.backbones.videomamba.videomamba import PretrainVideoMamba

num_frames = 8
img_size = 224
batch_size = 64
max_txt_l = 32

# model_pth = "videomamba_m16_5M_f8_res224.pth"
model_pth = "videomamba_m16_25M_f8_res224.pth"

config_dict = {
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
    "text_enc": "bert",
    "model": {
        "model_cls": UMT_VIDEOMAMBA,
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
        "text_encoder": {
            "name": "bert_base",
            "pretrained": "bert-base-uncased",
            "config": "llava/model/multimodal_encoder/videomamba/configs/config_bert.json",
            "d_model": 768,
            "fusion_layer": 9,
        },
        "multimodal": {"enable": True},
        "embed_dim": 512,
        "temp": 0.07,
    },
    "evaluate": False,
    "deep_fusion": False,
    "evaluation": {
        "eval_frame_ensemble": "concat",
        "eval_x_only": False,
        "k_test": 128,
        "eval_offload": False,
    },
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

DEFAULT_VIDEOMAMBA_CONFIG = EasyDict(config_dict)


class VideoMambaEncoder(torch.nn.Module):
    def __init__(self, vision_encoder):  # , vision_proj):
        super().__init__()

        self.vision_encoder = vision_encoder
        # self.vision_proj = vision_proj

    def forward(self, x, mask=None, use_image=False, keep_temporal=False):
        vision_embeds, pooled_vision_embeds, _ = self.vision_encoder(
            x, mask, use_image, keep_temporal
        )

        # vision_embeds is batch, 1568, 576
        # pooled_vision_embeds is batch, frames, 576

        # vision_proj = self.vision_proj(pooled_vision_embeds)
        return pooled_vision_embeds


class VideoMambaVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, **kwargs):
        super().__init__()

        self._dtype = torch.float32
        self.is_loaded = False

        self.mamba_config = DEFAULT_VIDEOMAMBA_CONFIG

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        # self.hidden_size = None

        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            # self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.cfg_only = self.mamba_config

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        tokenizer = BertTokenizer.from_pretrained(
            self.mamba_config.model.text_encoder.pretrained
        )

        self.video_processor = VideoMambaVideoProcessor(
            config=self.mamba_config, tokenizer=tokenizer
        )

        encoder_name = self.mamba_config.model.vision_encoder.name

        if not "videomamba" in encoder_name:
            raise NotImplementedError

        config = DEFAULT_VIDEOMAMBA_CONFIG.model

        checkpoint = torch.load(config.vision_encoder.pretrained, map_location="cpu")

        new_state_dict = {}

        for k, v in checkpoint.items():
            # if "vision_encoder" in k or "vision_proj" in k:
            if "vision_encoder" in k:
                new_state_dict[k] = v

        vision_encoder = PretrainVideoMamba(
            img_size=config.vision_encoder.img_size,
            patch_size=config.vision_encoder.patch_size,
            depth=config.vision_encoder.depth,
            embed_dim=config.vision_encoder.embed_dim,
            drop_path_rate=config.vision_encoder.drop_path_rate,
            ssm_cfg=config.vision_encoder.ssm_cfg,
            norm_epsilon=config.vision_encoder.norm_epsilon,
            fused_add_norm=config.vision_encoder.fused_add_norm,
            rms_norm=config.vision_encoder.rms_norm,
            residual_in_fp32=config.vision_encoder.residual_in_fp32,
            bimamba=config.vision_encoder.bimamba,
            pool_type=config.vision_encoder.pool_type,
            kernel_size=config.vision_encoder.kernel_size,
            num_frames=config.vision_encoder.num_frames,
            use_checkpoint=config.vision_encoder.use_checkpoint,
            checkpoint_num=config.vision_encoder.checkpoint_num,
            clip_decoder_embed_dim=config.vision_encoder.clip_decoder_embed_dim,
            clip_output_dim=config.vision_encoder.clip_output_dim,
            clip_return_layer=config.vision_encoder.clip_return_layer,
            clip_student_return_interval=config.vision_encoder.clip_student_return_interval,
            add_pool_norm=True,  # TO GET POOLED FEATURES
        )

        # vision_proj = torch.nn.Linear(config.vision_encoder.embed_dim, config.embed_dim)
        model = VideoMambaEncoder(vision_encoder)  # , vision_proj)
        model.to(torch.device(self.mamba_config.device))
        if self.mamba_config.fp16:
            if self.mamba_config.get("bf16", True):
                model = model.to(torch.bfloat16)
                self._dtype = torch.bfloat16
            else:
                model = model.half()
                self._dtype = torch.half

        msg = model.load_state_dict(new_state_dict, strict=True)
        print(msg)

        self.vision_tower = model
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        print("DONE LOADING")

    def extract_video_features(self, image):
        """
        Args:
        image (torch.Tensor): The input images.
        test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """

        T = image.shape[2]
        # use_image = True if T == 1 else False
        use_image = False

        # print(image.shape)

        # image = image.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        keep_temporal = self.config.model.vision_encoder.keep_temporal

        vision_features = self.vision_tower(
            image,
            None,
            use_image,
            keep_temporal,
        )

        return vision_features

    def feature_select(self, video_forward_outs):
        # video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
        return video_forward_outs  # return all

    @torch.no_grad()
    def forward(self, videos):

        # return image_features
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_forward_out = self.extract_video_features(
                    video.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                )

                video_feature = self.feature_select(video_forward_out).to(video.dtype)
                video_features.append(video_feature)
        else:

            video_forward_outs = self.extract_video_features(
                videos.to(device=self.device, dtype=self.dtype),
            )

            video_features = self.feature_select(video_forward_outs).to(videos.dtype)

        return video_features

    @property
    def dummy_feature(self):
        raise NotImplementedError
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self.mamba_config.device
        # return self.vision_tower.device

    @property
    def config(self):
        # if self.is_loaded:
        #     return self.vision_tower.config
        # else:
        #     return self.mamba_config
        return self.mamba_config

    @property
    def hidden_size(self):
        return self.mamba_config.model.vision_encoder.embed_dim
        # return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        raise NotImplementedError
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        raise NotImplementedError
        return (self.config.image_size // self.config.patch_size) ** 2
