cd /data/gemamba/

export PYTHONPATH=$PYTHONPATH:/data/gemamba/
export HF_TOKEN=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
export CUTLASS_PATH=~/cutlass
# export CUDA_VISIBLE_DEVICES=0

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path google/gemma-2b-it \
    --version gemma \
    --vision_tower videomamba \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --dataloader_num_workers 64 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir" \
    --data_path /data/valley/train_json/videochatgpt_tune_fixed.json \
    --video_folder /data/videochatgpt \
    --output_dir ./checkpoints/llava_gemma_mamba_debug \
    --num_train_epochs 1 \
    --learning_rate 3e-5 \
    --pretrain_mm_mlp_adapter mm_projector.bin \


    # --resume_from_checkpoint ./checkpoints/llava_gemma_mamba_v14_full_valley \
    # --data_path /data/valley/train_json/valley_exist.json \
    # --video_folder /data/valley \
    # --tune_mm_mlp_adapter True \  saves only the adapter into the checkpoint
