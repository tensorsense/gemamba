import os
from .videomamba.vision_tower import VideoMambaVisionTower

# ==== UPDATED FOR VISION TOWER ====

def build_vision_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(
        video_tower_cfg,
        "mm_vision_tower",
        getattr(video_tower_cfg, "vision_tower", None),
    )
    # if video_tower.endswith("LanguageBind_Video_merge"):
    #     return LanguageBindVideoTower(
    #         video_tower, args=video_tower_cfg, cache_dir="./cache_dir", **kwargs
    #     )
    if video_tower.endswith("videomamba"):
        return VideoMambaVisionTower(
            video_tower, video_tower_cfg, cache_dir="./cache_dir", **kwargs
        )
    raise ValueError(f"Unknown video tower: {video_tower}")

# ==================================
