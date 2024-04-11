import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindVideoTower

# ==== UPDATED FOR VISION TOWER ====

# def build_vision_tower(vision_tower_cfg, **kwargs):
#     vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
#     is_absolute_path_exists = os.path.exists(vision_tower)
#     if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
#         return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_vision_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(
        video_tower_cfg, "mm_vision_tower", getattr(video_tower_cfg, "vision_tower", None)
    )
    if video_tower.endswith("LanguageBind_Video_merge"):
        return LanguageBindVideoTower(
            video_tower, args=video_tower_cfg, cache_dir="./cache_dir", **kwargs
        )
    raise ValueError(f"Unknown video tower: {video_tower}")

# ==================================