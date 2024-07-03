import torch
import torch.nn as nn

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .videomamba2 import VideoMambaVisionModel, VideoMambaVisionConfig, VideoMambaProcessor


class VideoMambaVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, **kwargs):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = VideoMambaVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.video_processor = VideoMambaProcessor.from_pretrained(self.vision_tower_name).video_processor
        self.vision_tower = VideoMambaVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features

    @torch.no_grad()
    def forward(self, videos):
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_feature = self.vision_tower(video.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                # image_feature = self.feature_select(image_forward_out).to(image.dtype)
                video_features.append(video_feature)
        else:
            video_features = self.vision_tower(videos.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return video_features

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