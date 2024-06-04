from transformers import (
    BitsAndBytesConfig,
    # LlavaForConditionalGeneration,
    AutoModelForSequenceClassification,
    IdeficsModel,
    AutoProcessor,
)
from dataclasses import dataclass, field
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Literal, Tuple
from datasets import load_dataset, Image
from peft import LoraConfig, PeftModelForCausalLM
import json
import os
import bitsandbytes as bnb
from bakllava_rlhf.multi_modal_reward_trainer import MultiModalRewardTrainer, RewardDataCollatorWithPadding
from bakllava_rlhf.idefics_2_for_sequence_classification import Idefics2ForSequenceClassification
from peft.tuners.lora import LoraLayer
import os
import wandb
from tqdm import tqdm
from bakllava_rlhf.preprocess_dataset import preprocess_data, preprocess_data_batch



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
GRAD_ACCUMULATION = 4
BATCH_SIZE = 4
VALIDATE_MODEL_TRAINED = False


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
    half_precision_backend: str = field(default="amp")
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
    num_train_epochs: int = field(
        default=1
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
        default="wandb",
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
    dataloader_num_workers: int = field(default=4, metadata={"help": "Number of dataloader workers"})


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
        llm_int8_skip_modules=["mm_projector", "lm_head", "classifier"],
    )

    # model_id = "llava-hf/bakLlava-v1-hf"
    model_id = "HuggingFaceM4/idefics2-8b-base"
    processor = AutoProcessor.from_pretrained(
        model_id,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
        size={"longest_edge": 448, "shortest_edge": 378} 
    )
    processor.image_processor.do_image_splitting = False

    image_path = f"{os.getenv('DATA_DIR')}/coco/train2017"
    with open(f"{os.getenv('DATA_DIR')}/image_to_caption.json") as f:
        caption_map = json.load(f)
    
    train_dataset = load_dataset("zhiqings/LLaVA-Human-Preference-10K")["train"]
    def image_path_item(item):
        item["image"] = os.path.join(image_path, item["image"])
        return item
    train_dataset = train_dataset.map(image_path_item).cast_column("image", Image()).map(lambda item: preprocess_data_batch(item, processor, image_path, caption_map), batched=True, batch_size=150, num_proc=4)
    
    print("CHECK TRAIN", train_dataset[0].keys())
    
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
        # use_dora=True,
        modules_to_save=["classifier"],
    )
    
    adapter_name = "lora_default"
    
    train_dataset, eval_dataset = split_train_into_train_and_eval(
        train_dataset=train_dataset,
        eval_size=data_args.eval_size,
        seed=training_args.seed,
    )

    
    data_collator = RewardDataCollatorWithPadding(
        processor.tokenizer, max_length=512
    )
    if VALIDATE_MODEL_TRAINED:
        model = Idefics2ForSequenceClassification.from_pretrained(
            model_id,
            num_labels=1,
            low_cpu_mem_usage=True,
            # load_in_4bit=bits == 4,
            # load_in_8bit=bits == 8,
            device_map="auto",
            # quantization_config=bits_and_bytes_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
        )
        model = PeftModelForCausalLM(model, peft_config, adapter_name=adapter_name)
        training_args.gradient_checkpointing_kwargs = {"use_reentrant":False}
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
        trainer.save_model(f"{training_args.output_dir}/final")
    else:
        trained = True
        results_chosen = []
        results_rejected = []
        print("TRAINED:", trained)
        if not VALIDATE_MODEL_TRAINED:
            model = Idefics2ForSequenceClassification.from_pretrained(
                model_id,
                num_labels=1,
                low_cpu_mem_usage=True,
                # load_in_4bit=bits == 4,
                # load_in_8bit=bits == 8,
                device_map="auto",
                # quantization_config=bits_and_bytes_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",
            )
            model = PeftModelForCausalLM.from_pretrained(model, "./output/final/lora_default") 
        else:
            model = Idefics2ForSequenceClassification.from_pretrained(
                model_id,
                num_labels=1,
                low_cpu_mem_usage=True,
                # load_in_4bit=bits == 4,
                # load_in_8bit=bits == 8,
                device_map="auto",
                # quantization_config=bits_and_bytes_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",
            )
            model = PeftModelForCausalLM(model, peft_config, adapter_name=adapter_name)
        batch_size = 3
        eval_examples = batch_size * 120
        model.eval()
        for i in tqdm(range(0, eval_examples, batch_size)):
            print(i)
            inputs = train_dataset[i:i+batch_size]
            # print(torch.tensor(inputs["input_ids_chosen"]).shape)
            rewards = model(
                input_ids=torch.tensor(inputs["input_ids_chosen"]).to(current_device),
                attention_mask=torch.tensor(inputs["attention_mask_chosen"]).to(current_device),
                pixel_values=torch.tensor(inputs["pixel_values"]).to(current_device),
                pixel_attention_mask=torch.tensor(inputs["pixel_attention_mask"]).to(current_device),
                return_dict=True,
            )["logits"]
            results_chosen.extend(rewards.squeeze().tolist())
            rewards = None
            rewards = model(
                input_ids=torch.tensor(inputs["input_ids_rejected"]).to(current_device),
                attention_mask=torch.tensor(inputs["attention_mask_rejected"]).to(current_device),
                pixel_values=torch.tensor(inputs["pixel_values"]).to(current_device),
                pixel_attention_mask=torch.tensor(inputs["pixel_attention_mask"]).to(current_device),
                return_dict=True,
            )["logits"]
            results_rejected.extend(rewards.squeeze().tolist())
            rewards = None

        results = [1 if chosen > rejected else 0 for chosen, rejected in zip(results_chosen, results_rejected)]

        num_correct = sum(results)
        num_total = len(results)
        print(f"{num_correct}/{num_total} ({num_correct/num_total})")


if __name__ == "__main__":
    main()
