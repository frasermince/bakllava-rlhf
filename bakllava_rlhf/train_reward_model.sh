HUGGINGFACE_CACHE="/workspace/.cache/huggingface"
export DATA_DIR=../LLaVA-RLHF/data
export HF_DATASETS_CACHE="${HUGGINGFACE_CACHE}/datasets"
export HF_HOME="${HUGGINGFACE_CACHE}/misc"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE}/transformers"

export CUDA_VISIBLE_DEVICES=0#,1,2,3,4,5,6,7
export GPUS_PER_NODE=1
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
NUM_EPOCHS=1

echo $CUDA_VISIBLE_DEVICES

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    bakllava_rlhf/train_reward_model.py \
    --model_max_length 2048 \
    --bits 16 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --query_len 1280 \
    --response_len 768 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --reward_prompt_file "./prompts/fact_rlhf_reward_prompt.txt" \
    --image_to_caption_file "$DATA_DIR/image_to_caption.json" \
    --num_train_epochs $NUM_EPOCHS \
    --image_aspect_ratio 'pad'


