export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
export HF_HOME="/workspace/.cache/huggingface/misc"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

poetry run torchrun \
    -- --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    bakllava_rlhf/train_reward_model.py 
