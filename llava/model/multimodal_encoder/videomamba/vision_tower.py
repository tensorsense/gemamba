import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .hf_parts.processing_videomamba import VideoMambaVideoProcessor
from .models.umt_videomamba import UMT_VIDEOMAMBA
from .models.backbones.bert.tokenization_bert import BertTokenizer
from .utils.easydict import EasyDict
from .models.backbones.videomamba import build_videomamba

num_frames = 8
img_size = 224
batch_size = 64
max_txt_l = 32

model_pth = "llava/model/multimodal_encoder/videomamba/videomamba_m16_k400_mask_ft_f8_res224.pth"

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
    "pretrained_path": model_pth,
    "distributed": False,
}

DEFAULT_VIDEOMAMBA_CONFIG = EasyDict(config_dict)


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
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

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

        # need to register config and autoprocessor first
        # self.video_processor = VideoMambaVideoProcessor.from_pretrained(
        #     self.vision_tower_name
        # )

        self.video_processor = VideoMambaVideoProcessor(
            config=self.mamba_config, tokenizer=tokenizer
        )

        # model = UMT_VIDEOMAMBA(
        #     config=self.mamba_config, tokenizer=tokenizer, is_pretrain=False
        # )
        # model = model.to(torch.device(self.mamba_config.device))

        encoder_name = self.mamba_config.model.vision_encoder.name
        # logger.info(f"Build vision_encoder: {encoder_name}")
        if "videomamba" in encoder_name:
            model = build_videomamba(self.mamba_config.model)

        if self.mamba_config.fp16:
            if self.mamba_config.get("bf16", True):
                model = model.to(torch.bfloat16)
                self._dtype = torch.bfloat16
            else:
                model = model.half()
                self._dtype = torch.half

        checkpoint = torch.load(self.mamba_config.pretrained_path, map_location="cpu")

        # print(checkpoint.keys())

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        print("LOADING WHAT SEEMS LIKE THE CORRECT STATE DICT")

        print(type(model))

        # self.hidden_size = model.embed_dim

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        self.vision_tower = model
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        print("DONE LOADING")

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == "patch":
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == "cls_patch":
    #         image_features = image_features
    #     else:
    #         raise ValueError(f"Unexpected select feature: {self.select_feature}")
    #     return image_features

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

        T = image.shape[1]
        # use_image = True if T == 1 else False
        use_image = False
        
        # print(image.shape)

        image = image.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        keep_temporal = self.config.model.vision_encoder.keep_temporal

        vision_embeds, pooled_vision_embeds, _ = self.vision_tower(
            image,
            None,
            use_image,
            keep_temporal,
        )

        # return vision_embeds, pooled_vision_embeds
        return pooled_vision_embeds

    def feature_select(self, video_forward_outs):
        # video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
        return video_forward_outs  # return all

    @torch.no_grad()
    def forward(self, videos):
        # if type(images) is list:
        #     image_features = []
        #     for image in images:
        #         image_forward_out = self.vision_tower(
        #             image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
        #             output_hidden_states=True,
        #         )
        #         image_feature = self.feature_select(image_forward_out).to(image.dtype)
        #         image_features.append(image_feature)
        # else:
        #     image_forward_outs = self.vision_tower(
        #         images.to(device=self.device, dtype=self.dtype),
        #         output_hidden_states=True,
        #     )
        #     image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # return image_features
        if type(videos) is list:
            video_features = []
            for video in videos:
                # video_forward_out = self.vision_tower(
                #     video.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                #     # output_hidden_states=True,
                # )

                video_forward_out = self.extract_video_features(
                    video.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                )

                video_feature = self.feature_select(video_forward_out).to(video.dtype)
                video_features.append(video_feature)
        else:
            # video_forward_outs = self.vision_tower(
            #     videos.to(device=self.device, dtype=self.dtype),
            #     # output_hidden_states=True,
            # )

            video_forward_outs = self.extract_video_features(
                videos.to(device=self.device, dtype=self.dtype),
            )

            video_features = self.feature_select(video_forward_outs).to(videos.dtype)

        return video_features

    # def __init__(self, video_tower, args, delay_load=False, cache_dir='./cache_dir'):
    #     super().__init__()

    #     self.is_loaded = False

    #     self.video_tower_name = video_tower
    #     self.select_layer = args.mm_vision_select_layer
    #     self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

    #     self.cache_dir = cache_dir

    #     if not delay_load:
    #         self.load_model()
    #     else:
    #         self.cfg_only = LanguageBindVideoConfig.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)

    # ############################################################
    # def load_model(self, device_map=None):
    #     model = LanguageBindVideo.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)
    #     self.video_processor = LanguageBindVideoProcessor(model.config)

    #     # model = LanguageBindImage.from_pretrained('LanguageBind/LanguageBind_Image', cache_dir=self.cache_dir)
    #     self.video_tower = model.vision_model
    #     self.video_tower.requires_grad_(False)

    #     self.is_loaded = True

    # def feature_select(self, video_forward_outs):
    #     video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
    #     return video_features  # return all
    #     # b, t, n, c = video_features.shape
    #     # if self.select_feature == 'patch':
    #     #     video_features = video_features[:, :, 1:]
    #     # else:
    #     #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     # return video_features

    # @torch.no_grad()
    # def forward(self, videos):
    #     if type(videos) is list:
    #         video_features = []
    #         for video in videos:
    #             video_forward_out = self.video_tower(video.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
    #             video_feature = self.feature_select(video_forward_out).to(video.dtype)
    #             video_features.append(video_feature)
    #     else:
    #         video_forward_outs = self.video_tower(videos.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
    #         video_features = self.feature_select(video_forward_outs).to(videos.dtype)

    #     return video_features

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
        return self.vision_tower.embed_dim
        # return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        raise NotImplementedError
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        raise NotImplementedError
        return (self.config.image_size // self.config.patch_size) ** 2
