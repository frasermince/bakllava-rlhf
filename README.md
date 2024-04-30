# LLAVA RLHF Implementation

This implementation will eventually consist of the following steps

1. Choosing different base LLAVA models - This could include bakllava, llava 1.6 or custom llava models using other modalityies
2. Performing a SFT (Supervised Fine Tuning) step - This is often done before RLHF to improve results
3. Performing a reward model training step
4. Performing a PPO step.

This will look slightly different if we use different algorithms but is a useful starting point.

## Installation

Run poetry install
If you want to use flash attention I suggest creating a virtualenv and then running:
```
PIP_NO_BUILD_ISOLATION=1 pip install -e .
```

## Data setup

To setup the data needed for the reward model run the following

```
mkdir data
cd data
```

Use the following to download the coco train dataset

```
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

Download the image captions file image_to_caption.json from https://huggingface.co/datasets/zhiqings/LLaVA-RLHF-Data/ and place it in the data directory

## Training the Reward Model

To train the reward model run the following

```
sh train_reward_model.sh
```

I am pointing the huggingface cache to a custom location for training on RUNPROD. Feel free to comment out to use the default location. Comment out the following lines found in bakllava_rlhf/train_reward_model.sh

```
HUGGINGFACE_CACHE="/workspace/.cache/huggingface"
export HF_DATASETS_CACHE="${HUGGINGFACE_CACHE}/datasets"
export HF_HOME="${HUGGINGFACE_CACHE}/misc"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE}/transformers"
```
