from typing import Callable, Optional, Dict, List, Tuple, Union, Any
import torch
from trl import RewardConfig, RewardTrainer, is_peft_available
from trl.trainer import compute_accuracy
from typing import Callable, Optional, Dict, List, Tuple, Union, Any
from torch.utils.data import Dataset
import warnings
from transformers import (
    DataCollator,
    PreTrainedModel,
    TrainingArguments,
    LlavaProcessor,
    PreTrainedTokenizerBase,
)
import torch.nn as nn
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from dataclasses import dataclass
import inspect
import pdb


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training



@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        pixel_values = []
        # pixel_attention_mask = []
        for feature in features:
            # check if the keys are named as expected
            # print(feature.keys())
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            pixel_values.append(feature["pixel_values"])
            # pixel_attention_mask.append(feature["pixel_attention_mask"])
            if has_margin:
                margin.append(feature["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"].squeeze(),
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"].squeeze(),
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "pixel_values": torch.tensor(pixel_values).requires_grad_(True),
            # "pixel_attention_mask": torch.tensor(pixel_attention_mask),
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin

        return batch


class MultiModalRewardTrainer(RewardTrainer):
    # def __init__(
    #     self,
    #     model: Union[PreTrainedModel, nn.Module] = None,
    #     args: Optional[RewardConfig] = None,
    #     data_collator: Optional[DataCollator] = None,
    #     train_dataset: Optional[Dataset] = None,
    #     eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    #     processor: Optional[LlavaProcessor] = None,
    #     model_init: Optional[Callable[[], PreTrainedModel]] = None,
    #     compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    #     callbacks: Optional[List[TrainerCallback]] = None,
    #     optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
    #         None,
    #         None,
    #     ),
    #     preprocess_logits_for_metrics: Optional[
    #         Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    #     ] = None,
    #     max_length: Optional[int] = None,
    #     peft_config: Optional[Dict] = None,
    # ):
    #     print("TRAINING ARGS", args)
    #     print("DATASET START INIT", train_dataset[0].keys())
    #     if type(args) == TrainingArguments:
    #         warnings.warn(
    #             "Using `transformers.TrainingArguments` for `args` is deprecated and will be removed in a future version. Please use `RewardConfig` instead.",
    #             FutureWarning,
    #         )
    #         if max_length is not None:
    #             warnings.warn(
    #                 "The `max_length` argument is deprecated and will be removed in a future version. Please use the `RewardConfig` to set `max_length` instead.",
    #                 FutureWarning,
    #             )
    #     else:
    #         if max_length is not None and args.max_length is not None:
    #             raise ValueError(
    #                 "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
    #             )
    #         if max_length is not None and args.max_length is None:
    #             warnings.warn(
    #                 "The `max_length` argument is deprecated and will be removed in a future version. Please use the `RewardConfig` to set `max_length` instead.",
    #                 FutureWarning,
    #             )
    #     if not is_peft_available() and peft_config is not None:
    #         raise ValueError(
    #             "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
    #         )
    #     elif is_peft_available() and peft_config is not None:
    #         if not isinstance(model, PeftModel):
    #             if getattr(model, "is_loaded_in_8bit", False) or getattr(
    #                 model, "is_quantized", False
    #             ):
    #                 _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
    #                     inspect.signature(prepare_model_for_kbit_training).parameters
    #                 )

    #                 preprare_model_kwargs = {
    #                     "use_gradient_checkpointing": args.gradient_checkpointing
    #                 }

    #                 if (
    #                     not _supports_gc_kwargs
    #                     and args.gradient_checkpointing_kwargs is not None
    #                 ):
    #                     warnings.warn(
    #                         "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
    #                         "please update to the latest version of peft to use `gradient_checkpointing_kwargs`."
    #                     )
    #                 elif (
    #                     _supports_gc_kwargs
    #                     and args.gradient_checkpointing_kwargs is not None
    #                 ):
    #                     preprare_model_kwargs["gradient_checkpointing_kwargs"] = (
    #                         args.gradient_checkpointing_kwargs
    #                     )

    #                 model = prepare_model_for_kbit_training(
    #                     model, **preprare_model_kwargs
    #                 )

    #             model = get_peft_model(model, peft_config)

    #     if compute_metrics is None:
    #         compute_metrics = compute_accuracy

    #     if data_collator is None:
    #         if processor is None:
    #             raise ValueError(
    #                 "max_length or a tokenizer must be specified when using the default RewardDataCollatorWithPadding"
    #             )
    #         if type(args) == TrainingArguments:
    #             if max_length is None:
    #                 warnings.warn(
    #                     "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
    #                     " It will be set to `512` by default, but you should do it yourself in the future.",
    #                     UserWarning,
    #                 )
    #                 max_length = 512
    #         else:
    #             if max_length is None and args.max_length is None:
    #                 warnings.warn(
    #                     "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
    #                     " It will be set to `512` by default, but you should do it yourself in the future.",
    #                     UserWarning,
    #                 )
    #                 max_length = 512
    #             if max_length is None and args.max_length is not None:
    #                 max_length = args.max_length

    #         data_collator = RewardDataCollatorWithPadding(
    #             processor.tokenizer, max_length=max_length
    #         )

    #         if args.remove_unused_columns:
    #             try:  # for bc before https://github.com/huggingface/transformers/pull/25435
    #                 args.remove_unused_columns = False
    #             except FrozenInstanceError:
    #                 args = replace(args, remove_unused_columns=False)
    #             # warn users
    #             warnings.warn(
    #                 "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
    #                 " we have set it for you, but you should do it yourself in the future.",
    #                 UserWarning,
    #             )

    #         self.use_reward_data_collator = True
    #     else:
    #         self.use_reward_data_collator = False
    #     print("DATASET END INIT", train_dataset[0].keys())
    #     print("***ARGS", args)

    #     super().__init__(
    #         model,
    #         args,
    #         data_collator,
    #         train_dataset,
    #         eval_dataset,
    #         processor,
    #         model_init,
    #         compute_metrics,
    #         callbacks,
    #         optimizers,
    #         preprocess_logits_for_metrics,
    #     )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )


        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            pixel_values=inputs["pixel_values"],
            # pixel_attention_mask=inputs["pixel_attention_mask"],
            return_dict=True,
        )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            pixel_values=inputs["pixel_values"],
            # pixel_attention_mask=inputs["pixel_attention_mask"],
            return_dict=True,
        )["logits"]
        # import pdb; pdb.set_trace()
        # calculate loss, optionally modulate with margin
        # print("REWARDS CHOSEN", rewards_chosen)
        # print("REWARDS REJECTED", rewards_rejected)
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - inputs["margin"]
            ).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss
