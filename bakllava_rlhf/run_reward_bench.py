# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
# from transformers import AutoTokenizer, pipeline
from transformers import (
    BitsAndBytesConfig,
    # LlavaForConditionalGeneration,
    AutoModelForSequenceClassification,
    IdeficsModel,
    AutoProcessor,
)
from peft import LoraConfig, PeftModelForCausalLM
from bakllava_rlhf.llava_for_sequence_classification import Idefics2ForSequenceClassification

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset,
    save_to_hub,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="path to model")
    # parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    quantized = config["quantized"]  # only Starling isn't quantized for now
    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code
    model_id = "HuggingFaceM4/idefics2-8b-base"
    processor = AutoProcessor.from_pretrained(
        model_id,
        model_max_length=512,
        padding_side="left",
        truncation_side="right",
        size={"longest_edge": 448, "shortest_edge": 378} 
    )
    processor.image_processor.do_image_splitting = False

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=processor.tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )
    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {"device_map": {"": current_device}}

    

    current_device = torch.cuda.current_device()
    print("CURRENT DEVICE", current_device)
    modules = "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0,
        target_modules=modules,
        # use_dora=True,
        modules_to_save=["scores"],
    )
    # vision_config = transformers.CLIPVisionConfig(torch_dtype=torch.bfloat16)
    # text_config = transformers.MistralConfig(torch_dtype=torch.bfloat16)
    # configuration = transformers.LlavaConfig(vision_config, text_config, torch_dtype=torch.bfloat16)
    # model = Idefics2ForConditionalGeneration.from_pretrained(
    # with wandb.init(entity="frasermince") as run:
    #     # Pass the name and version of Artifact
    #     my_model_name = "unchart/huggingface/model-gxj6ln6q:v0"
    #     my_model_artifact = run.use_artifact(my_model_name, type='model')

    #     # Download model weights to a folder and return the path
    #     model_dir = my_model_artifact.download()
    #     print("MODEL DIR", model_dir)
    #     model = Idefics2ForSequenceClassification.from_pretrained(
    #         model_dir,
    #         num_labels=1,
    #         low_cpu_mem_usage=True,
    #         # load_in_4bit=bits == 4,
    #         # load_in_8bit=bits == 8,
    #         device_map="auto",
    #         quantization_config=bits_and_bytes_config,
    #         torch_dtype=torch.bfloat16,
    #         trust_remote_code=True,
    #         # attn_implementation="flash_attention_2",
    #     )
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
    model.to(torch.cuda.current_device())
    # model = Idefics2ForSequenceClassification.from_pretrained("./output/final")
    # model.to(torch.bfloat16)
    # model.multi_modal_projector.to(torch.bfloat16)
    # model.to(torch.bfloat16)
    # adapter_name = "lora_default"
    # model = PeftModelForCausalLM("./output/final", peft_config, adapter_name=adapter_name)
    # model.load_adapter("./output/final")
    # model = Idefics2ForSequenceClassification.from_pretrained("./output/final")


    # reward_pipe = pipeline_builder(
    #     "text-classification",
    #     model=model,
    #     tokenizer=tokenizer,
    # )

    # ############################
    # # Tokenization settings & dataset preparation
    # ############################
    # # set pad token to eos token if not set
    # if reward_pipe.tokenizer.pad_token_id is None:
    #     reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
    #     reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # # For models whose config did not contains `pad_token_id`
    # if reward_pipe.model.config.pad_token_id is None:
    #     reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    # if not check_tokenizer_chat_template(tokenizer):
    #     reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    # if pipeline_builder == pipeline:
    logger.info("*** Running forward pass via built in pipeline abstraction ***")
    # this setup can be optimized slightly with one pipeline call
    # prepare for inference
    # reward_pipe = accelerator.prepare(reward_pipe)

    processed = processor(
        dataset["text_rejected"],
        return_tensors="pt",
        padding="max_length",
        truncation=True
    ).to(torch.cuda.current_device())
    batch_size = 8
    results_rej = {'logits': [], 'past_key_values': [], 'hidden_states': []}
    for i in range(0, processed['input_ids'].size(0), batch_size):
        batch_processed = {key: val[i:i+batch_size] for key, val in processed.items()}
        batch_results = model(**batch_processed)
        results_rej['logits'].extend(batch_results.logits.cpu())
        # results_rej['past_key_values'].extend(batch_results.past_key_values)
        # results_rej['hidden_states'].extend(batch_results.hidden_states)
    processed = processor(
        dataset["text_chosen"],
        return_tensors="pt",
        padding="max_length",
        truncation=True
    ).to(torch.cuda.current_device())
    results_cho = {'logits': [], 'past_key_values': [], 'hidden_states': []}
    for i in range(0, processed['input_ids'].size(0), batch_size):
        batch_processed = {key: val[i:i+batch_size] for key, val in processed.items()}
        batch_results = model(**batch_processed)
        results_cho['logits'].extend(batch_results.logits.cpu())
        # results_cho['past_key_values'].extend([[b.cpu()] for b in batch_results.past_key_values])
        # results_cho['hidden_states'].extend(batch_results.hidden_states)

    # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
    scores_chosen = [result.item() for result in results_cho["logits"]]
    scores_rejected = [result.item() for result in results_rej["logits"]]

    # pairwise comparison list comprehension
    results = [1 if chosen > rejected else 0 for chosen, rejected in zip(scores_chosen, scores_rejected)]

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = "custom"
    results_grouped["model_type"] = "idefics2"
    results_grouped["chat_template"] = (
        args.chat_template if not check_tokenizer_chat_template(processor.tokenizer) else "tokenizer"
    )

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)

    ############################
    # Upload results to hub
    ############################
    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    # results_url = save_to_hub(
    #     results_grouped,
    #     args.model,
    #     sub_path,
    #     args.debug,
    #     local_only=args.do_not_save,
    #     save_metrics_for_beaker=not args.disable_beaker_save,
    # )
    if not args.do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    # upload chosen-rejected with scores
    if not model_type == "Custom Classifier":  # custom classifiers do not return scores
        # create new json with scores and upload
        scores_dict = out_dataset.to_dict()
        scores_dict["model"] = args.model
        scores_dict["model_type"] = model_type
        scores_dict["chat_template"] = args.chat_template

        sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

        scores_url = save_to_hub(scores_dict, args.model, sub_path_scores, args.debug, local_only=args.do_not_save)
        logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")
    else:
        logger.info("Not uploading chosen-rejected text with scores due to model compatibility")


if __name__ == "__main__":
    main()
