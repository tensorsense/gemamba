cd /data/gemamba/

export PYTHONPATH=$PYTHONPATH:/data/gemamba/
export HF_TOKEN=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
export CUTLASS_PATH=~/cutlass
# export DS_SKIP_CUDA_CHECK=1
# export CUDA_VISIBLE_DEVICES=0,1,2

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path google/gemma-2b-it \
    --version gemma \
    --data_path /data/valley/train_json/valley_exist.json \
    --video_folder /data/valley \
    --vision_tower videomamba \
    --freeze_backbone True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava_gemma_mamba_v2_pt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

        # --tune_mm_mlp_adapter True \  saves only the adapter into the checkpoint
        # --data_path /data/valley/train_json/valley_exist.json \
            # --freeze_backbone True \