from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    # Idefics2ForConditionalGeneration,
    AutoProcessor,
)
from dataclasses import dataclass, field
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Literal, Tuple
from datasets import load_dataset
from peft import LoraConfig, PeftModelForCausalLM
import json
from PIL import Image
# from torchvision.transforms.functional import pil_to_tensor
import os
import bitsandbytes as bnb
from bakllava_rlhf.multi_modal_reward_trainer import MultiModalRewardTrainer, RewardDataCollatorWithPadding
from peft.tuners.lora import LoraLayer
import os
import time
import sys


starting_prompt = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""

def value_prompt(captions):
    caption_string = ""
    for caption in captions:
        caption_string += f"{caption}\n"
    return f"""
USER: Please evaluate the quality of your last response. There are several dimensions you should consider in your evaluation:

1. Accurate: The AI should provide factual and accurate information from the image, and refrain from making statements that are not supported by the image or inconsistent with the image. Specifically, the AI's response should be fully supported by the combination of the following captions:
{caption_string}
2. Helpful: The AI’s response should precisely serve the user's needs and interests, while grounding the response in the image.
3. Language Natural: The AI should employ language that flows smoothly and is free from repetitive or awkward constructs.
4. Concise: The AI should efficiently address the task or answer the question, communicating the necessary information with brevity and clarity.

A good response should be accurate, helpful, language natural, and concise. ASSISTANT: Following your definitions, the quality score of my last response is
"""

def add_message_with_role_string(conversation, is_start):
  if conversation["from"] == "human":
      role_string = "USER"
  elif conversation["from"] == "gpt":
      role_string = "ASSISTANT"
  else:
      role_string = "ASSISTANT"
  # if prompt == "":
  #     prompt += f"{starting_prompt}\n{role_string}:\n"
  # else:
  #     prompt += f"{role_string}:"
  newline = "\n"
  return f"{role_string}:{newline if is_start else ''}{conversation['value']}{newline}"
  # prompt += f"{conversation['value']}\n"


def preprocess_data(example, processor, image_path, caption_map):
    new_example = {}
    first = example["output_1"]
    second = example["output_2"]
    choice = example["preference"]
    image = example["image"]
    conversations = example["conversations"]
    if choice == 1:
        chosen = first
        rejected = second
    elif choice == 2:
        chosen = second
        rejected = first
    else:
        raise ValueError("Choice must be 1 or 2")
    if conversations[0]["from"] == "human":
        starting_index = 0
    else:
        starting_index = 1

    assert conversations[-1]["from"] == "gpt"
    conversations = conversations[:-1]
    
    prompt = f"{starting_prompt}\n"
    is_start = True
    for conversation in conversations[starting_index:]:
        prompt += add_message_with_role_string(conversation, is_start)
        if is_start == True:
          is_start == False
        # if conversation["from"] == "human":
        #     role_string = "USER"
        # elif conversation["from"] == "gpt":
        #     role_string = "ASSISTANT"
        # else:
        #     role_string = "ASSISTANT"
        # if prompt == "":
        #     prompt += f"{starting_prompt}\n{role_string}:\n"
        # else:
        #     prompt += f"{role_string}:"
        # prompt += f"{conversation['value']}\n"

    filename = image.filename.split("/")[-1]
    prompt_ending = value_prompt(caption_map[filename])
    prompt_chosen = prompt + f"ASSISTANT: {chosen['value']}\n{prompt_ending}"
    prompt_reject = prompt + f"ASSISTANT: {rejected['value']}\n{prompt_ending}"

    # with Image.open(os.path.join(image_path, image)) as raw_image:
    processed_chosen = processor(
        prompt_chosen,
        image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )#.to(0, torch.bfloat16)
    processed_rejected = processor(
        prompt_reject,
        image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )#.to(0, torch.bfloat16)

    # new_example["input_ids_chosen"] = torch.tensor(processed_chosen["input_ids"])
    # new_example["attention_mask_chosen"] = torch.tensor(processed_chosen["attention_mask"])
    # new_example["pixel_values_chosen"] = torch.tensor(processed_chosen["pixel_values"])
    # new_example["pixel_attention_mask_chosen"] = torch.tensor(processed_chosen["pixel_attention_mask"])

    # new_example["input_ids_rejected"] = torch.tensor(processed_rejected["input_ids"])
    # new_example["attention_mask_rejected"] = torch.tensor(processed_rejected["attention_mask"])
    # new_example["pixel_values_rejected"] = torch.tensor(processed_rejected["pixel_values"])
    # new_example["pixel_attention_mask_rejected"] = torch.tensor(processed_rejected["pixel_attention_mask"])

    new_example["input_ids_chosen"] = processed_chosen["input_ids"]
    new_example["attention_mask_chosen"] = processed_chosen["attention_mask"]
    new_example["pixel_values"] = processed_chosen["pixel_values"]
    # new_example["pixel_attention_mask_chosen"] = processed_chosen["pixel_attention_mask"]

    new_example["input_ids_rejected"] = processed_rejected["input_ids"]
    new_example["attention_mask_rejected"] = processed_rejected["attention_mask"]
    new_example["pixel_values_rejected"] = processed_rejected["pixel_values"]
    # new_example["pixel_attention_mask_rejected"] = processed_rejected["pixel_attention_mask"]

    new_example["length"] = processed_chosen["input_ids"].shape

    return new_example

def preprocess_data_batch(example, processor, image_path, caption_map):
    first = example["output_1"]
    second = example["output_2"]

    choice = example["preference"]
    images = []
    for i in example["image"]:
      images.append([i])
    conversations_batch = example["conversations"]
    chosen = []
    rejected = []
    start = time.time()
    for i, c in enumerate(choice):
      if c == 1:
        chosen.append(first[i])
        rejected.append(second[i])
      else:
        chosen.append(second[i])
        rejected.append(first[i])

    
    # if choice == 1:
    #     chosen = first
    #     rejected = second
    # elif choice == 2:
    #     chosen = second
    #     rejected = first
    # else:
    #     raise ValueError("Choice must be 1 or 2")

    start = time.time()
    chosen_prompts = []
    rejected_prompts = []
    for i, conversations in enumerate(conversations_batch):
      if conversations[0]["from"] == "human":
          starting_index = 0
      else:
          starting_index = 1

      assert conversations[-1]["from"] == "gpt"
      conversations = conversations[:-1]

      prompt = f"{starting_prompt}\n"
      is_start = True
      for conversation in conversations[starting_index:]:
          prompt += add_message_with_role_string(conversation, is_start)
          if is_start == True:
            is_start == False

      filename = images[i][0].filename.split("/")[-1]
      prompt_ending = value_prompt(caption_map[filename])
      prompt_chosen = prompt + f"ASSISTANT: {chosen[i]['value']}\n{prompt_ending}"
      prompt_reject = prompt + f"ASSISTANT: {rejected[i]['value']}\n{prompt_ending}"
      chosen_prompts.append(prompt_chosen)
      rejected_prompts.append(prompt_reject)


    # with Image.open(os.path.join(image_path, image)) as raw_image:

    # TODO ensure truncation true is correct
    start = time.time()
    processed_chosen = processor(
        chosen_prompts,
        images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )#.to(0, torch.bfloat16)
    processed_rejected = processor(
        rejected_prompts,
        images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )#.to(0, torch.bfloat16)

    assert torch.all(processed_chosen["pixel_attention_mask"] == processed_rejected["pixel_attention_mask"])

    # new_example["input_ids_chosen"] = torch.tensor(processed_chosen["input_ids"])
    # new_example["attention_mask_chosen"] = torch.tensor(processed_chosen["attention_mask"])
    # new_example["pixel_values_chosen"] = torch.tensor(processed_chosen["pixel_values"])
    # new_example["pixel_attention_mask_chosen"] = torch.tensor(processed_chosen["pixel_attention_mask"])

    # new_example["input_ids_rejected"] = torch.tensor(processed_rejected["input_ids"])
    # new_example["attention_mask_rejected"] = torch.tensor(processed_rejected["attention_mask"])
    # new_example["pixel_values_rejected"] = torch.tensor(processed_rejected["pixel_values"])
    # new_example["pixel_attention_mask_rejected"] = torch.tensor(processed_rejected["pixel_attention_mask"])
    # assert torch.all(processed_chosen["pixel_values"] == processed_rejected["pixel_values"])
    # assert torch.all(processed_chosen["pixel_attention_mask"] == processed_rejected["pixel_attention_mask"])

    data_result = {
        "input_ids_chosen": processed_chosen["input_ids"],
        "input_ids_rejected": processed_rejected["input_ids"],
        "attention_mask_chosen": processed_chosen["attention_mask"],
        "attention_mask_rejected": processed_rejected["attention_mask"],
        "pixel_values": processed_chosen["pixel_values"],
        "pixel_attention_mask": processed_chosen["pixel_attention_mask"],
    }
    return data_result
