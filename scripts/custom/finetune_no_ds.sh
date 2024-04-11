cd /data/vlm_sandbox/custom_llava

export PYTHONPATH=$PYTHONPATH:/data/vlm_sandbox/custom_llava
export HF_TOKEN=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
export DS_SKIP_CUDA_CHECK=1

python3 llava/train/train.py \
    --model_name_or_path google/gemma-2b-it \
    --version gemma \
    --data_path /data/data/train_gpt.json \
    --video_folder /data/data \
    --vision_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava_gemma_v0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072  \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir" \
    
    # --model_name_or_path lmsys/vicuna-7b-v1.5 \
    # --pretrain_mm_mlp_adapter ./checkpoints/Video-LLaVA-Pretrain-7B/mm_projector.bin \
    # --tune_mm_mlp_adapter True \
    # --freeze_backbone True \
    # --tokenizer_model_max_length 3072 \