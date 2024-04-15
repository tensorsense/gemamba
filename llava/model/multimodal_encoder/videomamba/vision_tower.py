import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from hf_parts.processing_videomamba import VideoMambaVideoProcessor
from models.umt_videomamba import UMT_VIDEOMAMBA
from models.backbones.bert.tokenization_bert import BertTokenizer


class VideoMambaVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

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

        # self.image_processor = CLIPImageProcessor.from_pretrained(
        #     self.vision_tower_name
        # )

        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)

        self.video_processor = VideoMambaVideoProcessor.from_pretrained(
            self.vision_tower_name
        )

        model = UMT_VIDEOMAMBA(config=config, tokenizer=tokenizer, is_pretrain=pretrain)
        model = model.to(torch.device(config.device))

        if config.fp16:
            if config.get("bf16", True):
                logger.info("Change to bfloat16 for model")
                model = model.to(torch.bfloat16)
            else:
                logger.info("Change to float16 for model")
                model = model.half()

        # self.vision_tower = CLIPVisionModel.from_pretrained(
        #     self.vision_tower_name, device_map=device_map
        # )

        self.vision_tower = model
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == "patch":
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == "cls_patch":
    #         image_features = image_features
    #     else:
    #         raise ValueError(f"Unexpected select feature: {self.select_feature}")
    #     return image_features

    def feature_select(self, video_forward_outs):
        video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
        return video_features  # return all

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
                video_forward_out = self.video_tower(
                    video.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                video_feature = self.feature_select(video_forward_out).to(video.dtype)
                video_features.append(video_feature)
        else:
            video_forward_outs = self.video_tower(
                videos.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
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
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
