

if [ ! -f /data/data/vlm_sandbox/.env ]
then
    export $(grep -v '^#' .env | xargs)
fi

GPT_Zero_Shot_QA="/data/data"
output_name="llava_gemma_mamba_v11_lora"
pred_path="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/1_0.json" # merge.jsonl when it's finished
output_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/gpt"
output_json="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/results.json"
echo $pred_path

# api_key=""
# api_base=""
num_tasks=8

export PYTHONPATH="llava:$PYTHONPATH"

python3 llava/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --azure_deployment "gpt-35-turbo-1106" \
    --num_tasks ${num_tasks}


    # --api_key ${AZURE_OPENAI_ENDPOINT} \
    # --api_base ${AZURE_OPENAI_API_KEY} \