export HUGGINGFACE_CACHE="/workspace/.cache/huggingface"
export DATA_DIR=./data
export HF_DATASETS_CACHE="${HUGGINGFACE_CACHE}/datasets"
export HF_HOME="${HUGGINGFACE_CACHE}/misc"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE}/transformers"
export WANDB_LOG_MODEL=checkpoint
export WANDB_PROJECT=multimodal-rlhf

export CUDA_VISIBLE_DEVICES=0#,1,2,3,4,5,6,7
export GPUS_PER_NODE=1
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python bakllava_rlhf/run_reward_bench.py