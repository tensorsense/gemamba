

CKPT_NAME="llava_gemma_mamba_v18_adapter_vcgpt/checkpoint-1000"
model_path="checkpoints/${CKPT_NAME}"
# model_base="checkpoints/llava_gemma_mamba_base_3"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="/data/data"
video_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/videos/all"
gt_file_question="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_q1000.json"
gt_file_answers="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_a.json"
output_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${CKPT_NAME}"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

export PYTHONPATH="llava:$PYTHONPATH"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llava/eval/video/run_inference_video_qa.py \
      --model_path ${model_path} \
      --model_base ${model_base} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --temperature '0.2' \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge1000.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done