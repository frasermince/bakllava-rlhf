from transformers import (
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
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
import os
import bitsandbytes as bnb
from bakllava_rlhf.multi_modal_reward_trainer import MultiModalRewardTrainer, RewardDataCollatorWithPadding
from peft.tuners.lora import LoraLayer

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


# class BinaryRewardModelingDataset(Dataset):
#     def __init__(
#         self,
#         data: Sequence[dict],
#         tokenizer: transformers.PreTrainedTokenizer,
#         df_postprocessor: Optional[Callable] = None,
#         query_len: Optional[int] = None,
#         response_len: Optional[int] = None,
#         use_data_frame: bool = True,
#         data_args: Optional[Dict] = None,
#     ):
#         super(BinaryRewardModelingDataset, self).__init__()
#         list_data_dict = json.load(open(data_args.dataset_path, "r"))
#         self.tokenizer = tokenizer
#         self.list_data_dict = list_data_dict
#         self.data_args = data_args

#         self.query_len = query_len
#         self.response_len = response_len
#         self.use_data_frame = use_data_frame

#         self.reward_model_prompt = None
#         if data_args.reward_prompt_file is not None:
#             with open(data_args.reward_prompt_file, "r") as f:
#                 self.reward_model_prompt = " " + f.read().strip()

#         self.image_to_caption_mapping = None
#         if data_args.image_to_caption_file is not None:
#             with open(data_args.image_to_caption_file, "r") as f:
#                 self.image_to_caption_mapping = json.load(f)

#     def __len__(self):
#         # return len(self.input_ids)
#         return len(self.list_data_dict)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
#         if "image" in sources[0]:
#             image_file = self.list_data_dict[i]["image"]
#             image_folder = self.data_args.image_folder
#             processor = self.data_args.image_processor
#             image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
#             if self.data_args.image_aspect_ratio == "pad":

#                 def expand2square(pil_img, background_color):
#                     width, height = pil_img.size
#                     if width == height:
#                         return pil_img
#                     elif width > height:
#                         result = Image.new(
#                             pil_img.mode, (width, width), background_color
#                         )
#                         result.paste(pil_img, (0, (width - height) // 2))
#                         return result
#                     else:
#                         result = Image.new(
#                             pil_img.mode, (height, height), background_color
#                         )
#                         result.paste(pil_img, ((height - width) // 2, 0))
#                         return result

#                 image = expand2square(
#                     image, tuple(int(x * 255) for x in processor.image_mean)
#                 )
#                 image = processor.preprocess(image, return_tensors="pt")[
#                     "pixel_values"
#                 ][0]
#             else:
#                 image = processor.preprocess(image, return_tensors="pt")[
#                     "pixel_values"
#                 ][0]
#             _sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]), self.data_args
#             )
#         else:
#             _sources = copy.deepcopy([e["conversations"] for e in sources])

#         sources_ = copy.deepcopy(sources)
#         sources_[0]["conversations"] = _sources

#         data_dict = preprocess_for_reward_modeling(
#             sources_,
#             tokenizer=self.tokenizer,
#             has_image=("image" in self.list_data_dict[i]),
#             mask_target=False,
#             query_len=self.query_len,
#             response_len=self.response_len,
#             use_data_frame=self.use_data_frame,
#             reward_model_prompt=self.reward_model_prompt,
#             image_to_caption_mapping=self.image_to_caption_mapping,
#         )
#         if isinstance(i, int):
#             data_dict = dict(
#                 input_ids=data_dict["input_ids"][0],
#                 labels=data_dict["labels"][0],
#                 choice=data_dict["choice"][0],
#                 index_0=data_dict["index_0"][0],
#                 index_1=data_dict["index_1"][0],
#             )

#         # image exist in the data
#         if "image" in self.list_data_dict[i]:
#             data_dict["image"] = image.to(torch.bfloat16)
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

#         return data_dict


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    # from LLaVA
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None, metadata={"help": "Dataset name"})
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(
        default=500,
        metadata={
            "help": "Number of examples to split out from training to use for evaluation."
        },
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)


# LLAVA-RLHF used
# BATCH_SIZE=4
# GRAD_ACCUMULATION=1
# A_100s=8
# for a 13b model
# This is 4 x 1 x 8 = 32
# For one GPU this would be 8 x 4 x 1 = 32
GRAD_ACCUMULATION = 2
BATCH_SIZE = 8


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # From LLaVA
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    # From AlpacaFarm
    length_column_name: str = field(default="length")
    dataloader_pin_memory: bool = field(default=False)
    bf16: bool = field(default=False)
    max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    query_len: int = field(default=None, metadata={"help": "Length of the query."})
    response_len: int = field(
        default=None, metadata={"help": "Length of the response."}
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj",
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=BATCH_SIZE,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=GRAD_ACCUMULATION,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=2e-5, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False, metadata={"help": "Find unused parameters"}
    )
    ddp_backend: str = field(
        default="ddp", metadata={"help": "Distributed backend to use"}
    )


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(
    train_dataset: Dataset, eval_size: int, seed: int
) -> Tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset


class FinalConversation:
    def __init__(self, output_1, output_2):
        self.output_1 = output_1
        self.output_2 = output_2


class PreprocessDataset(Dataset):
    def __init__(self, data, processor, image_path, caption_map, starting_prompt):
        self.data = data
        self.processor = processor
        self.image_path = image_path
        self.caption_map = caption_map
        self.starting_prompt = starting_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
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
        raw_image = Image.open(os.path.join(self.image_path, image))
        if conversations[0]["from"] == "human":
            starting_index = 0
        else:
            starting_index = 1
        prompt = ""

        assert conversations[-1]["from"] == "gpt"
        conversations = conversations[:-1]
        for conversation in conversations[starting_index:]:
            if conversation["from"] == "human":
                role_string = "USER"
            elif conversation["from"] == "gpt":
                role_string = "ASSISTANT"
            else:
                role_string = "ASSISTANT"
            if prompt == "":
                prompt += f"{self.starting_prompt}\n{role_string}:\n"
            else:
                prompt += f"{role_string}:"
            prompt += f"{conversation['value']}\n"

        prompt_ending = value_prompt(self.caption_map[image])
        prompt_chosen = prompt + f"ASSISTANT: {chosen}\n{prompt_ending}"
        prompt_reject = prompt + f"ASSISTANT: {rejected}\n{prompt_ending}"

        processed_chosen = self.processor(
            prompt_chosen,
            raw_image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(0, torch.bfloat16)
        processed_rejected = self.processor(
            prompt_reject,
            raw_image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(0, torch.bfloat16)

        new_example["input_ids_chosen"] = processed_chosen["input_ids"]
        new_example["attention_mask_chosen"] = processed_chosen["attention_mask"]
        new_example["pixel_values_chosen"] = processed_chosen["pixel_values"]

        new_example["input_ids_rejected"] = processed_rejected["input_ids"]
        new_example["attention_mask_rejected"] = processed_rejected["attention_mask"]
        new_example["pixel_values_rejected"] = processed_rejected["pixel_values"]
        new_example["length"] = processed_chosen["input_ids"].shape

        return new_example

# To use the DataLoader
# dataset = PreprocessDataset(data, processor, image_path, caption_map, starting_prompt)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


def find_all_linear_names(
    bits: int,
    model: torch.nn.Module,
):
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def main():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    bits = 4
    bits_and_bytes_config = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["mm_projector", "lm_head"],
    )

    model_id = "llava-hf/bakLlava-v1-hf"
    processor = AutoProcessor.from_pretrained(
        model_id,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
    )

    image_path = "./data/coco/train2017"
    with open("data/image_to_caption.json") as f:
        caption_map = json.load(f)

    def preprocess_function(example):
        new_example = {}
        # Get the columns of a pandas DataFrame
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
        raw_image = Image.open(os.path.join(image_path, image))
        if conversations[0]["from"] == "human":
            starting_index = 0
        else:
            starting_index = 1
        prompt = ""

        assert conversations[-1]["from"] == "gpt"
        conversations = conversations[:-1]
        for conversation in conversations[starting_index:]:
            if conversation["from"] == "human":
                role_string = "USER"
            elif conversation["from"] == "gpt":
                role_string = "ASSISTANT"
            else:
                role_string = "ASSISTANT"
            if prompt == "":
                prompt += f"{starting_prompt}\n{role_string}:\n"
            else:
                prompt += f"{role_string}:"
            prompt += f"{conversation['value']}\n"

        prompt_ending = value_prompt(caption_map[image])
        prompt_chosen = prompt + f"ASSISTANT: {chosen}\n{prompt_ending}"
        prompt_reject = prompt + f"ASSISTANT: {rejected}\n{prompt_ending}"
        # TODO add the caption map

        processed_chosen = processor(
            prompt_chosen,
            raw_image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )  # .to(0, torch.float16)
        processed_rejected = processor(
            prompt_reject,
            raw_image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )  # .to(0, torch.float16)

        new_example["input_ids_chosen"] = processed_chosen["input_ids"]
        new_example["attention_mask_chosen"] = processed_chosen["attention_mask"]
        new_example["pixel_values_chosen"] = processed_chosen["pixel_values"]

        new_example["input_ids_rejected"] = processed_rejected["input_ids"]
        new_example["attention_mask_rejected"] = processed_rejected["attention_mask"]
        new_example["pixel_values_rejected"] = processed_rejected["pixel_values"]
        new_example["length"] = processed_chosen["input_ids"].shape

        return new_example

    train_dataset = load_dataset("zhiqings/LLaVA-Human-Preference-10K")["train"]
    print("TRAIN LENGTH", len(train_dataset))
    # train_dataset = train_dataset.map(
    #     preprocess_function,
    #     batched=False,
    #     num_proc=4,
    # )
    # To use the DataLoader
    train_dataset = PreprocessDataset(train_dataset, processor, image_path, caption_map, starting_prompt)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("CHECK TRAIN", train_dataset[0].keys())
    # train_dataset = train_dataset.filter(
    #     lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
    #     and len(x["input_ids_rejected"]) <= args.reward_config.max_length
    # )

    current_device = torch.cuda.current_device()
    print("CURRENT DEVICE", current_device)
    modules = training_args.lora_modules or find_all_linear_names(bits, model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=training_args.lora_dropout,
        target_modules=modules,
        # modules_to_save=["scores"],
    )
    vision_config = transformers.CLIPVisionConfig(torch_dtype=torch.bfloat16)
    text_config = transformers.MistralConfig(torch_dtype=torch.bfloat16)
    configuration = transformers.LlavaConfig(vision_config, text_config, torch_dtype=torch.bfloat16)
    model = LlavaForConditionalGeneration(configuration).from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        device_map={"": current_device},
        quantization_config=bits_and_bytes_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    
    adapter_name = "lora_default"
    model = PeftModelForCausalLM(model, peft_config, adapter_name=adapter_name)
    model.multi_modal_projector.to(torch.bfloat16)
    model.vision_tower.to(torch.bfloat16)
    model.vision_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)
    # model.vision_tower = PeftModelForCausalLM(model.vision_tower, peft_config, adapter_name=adapter_name)

    # model.config.torch_dtype = torch.bfloat16
    # for name, module in model.vision_tower.named_modules():
    #     print("ALL", module)

    #     if isinstance(module, LoraLayer):
    #         if training_args.bf16:
    #             print(module)
    #             module = module.to(torch.bfloat16)
    #     if "lm_head" in name or "embed_tokens" in name:
    #         if hasattr(module, "weight"):
    #             if training_args.bf16 and module.weight.dtype == torch.float32:
    #                 print(module)
    #                 module = module.to(torch.bfloat16)
    #     if isinstance(module, torch.nn.Conv2d):
    #         if training_args.bf16 and module.weight.dtype == torch.float32:
    #             print(module)
    #             module = module.to(torch.bfloat16)
    # import pdb; pdb.set_trace()

    train_dataset, eval_dataset = split_train_into_train_and_eval(
        train_dataset=train_dataset,
        eval_size=data_args.eval_size,
        seed=training_args.seed,
    )

    
    data_collator = RewardDataCollatorWithPadding(
        processor.tokenizer, max_length=512
    )
    trainer = MultiModalRewardTrainer(
        model=model,
        tokenizer=processor,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
