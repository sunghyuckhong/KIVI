import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")
import argparse
import json, tqdm
import torch
import copy

import math
import time
from lm_eval import evaluator, utils
from lm_eval.tasks import initialize_tasks, include_path
from lm_eval.api.registry import ALL_TASKS

from utils_paper.process_args import process_args
from transformers import LlamaConfig, AutoTokenizer, FalconConfig, MistralConfig
from utils_paper.data import set_seed
from datasets import load_dataset

from accelerate import Accelerator
accelerator = Accelerator()

if __name__ == '__main__':

    set_seed(42)

    model_args, data_args, training_args = process_args()
    dtype = torch.float16
    model_path = model_args.model_name_or_path.lower()
    is_llama = 'llama' in model_path
    is_mistral = 'mistral' in model_path

    if torch.cuda.device_count() > 1:
        parallel = True
        low_cpu_mem_usage=True
    else:
        parallel = False
        low_cpu_mem_usage=True

    if is_llama:
        if model_args.k_bits == 16 and model_args.v_bits == 16:
            from models_paper.modeling_llama import LMEvalLlamaForCausalLM
            model = LMEvalLlamaForCausalLM(
                k_bits=model_args.k_bits,
                v_bits=model_args.v_bits,
                group_size=model_args.group_size,
                residual_length=model_args.residual_length,
                pretrained=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                dtype=dtype,
                batch_size=data_args.batch_size,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        else:
            assert model_args.k_bits in [2, 4] and model_args.v_bits in [2, 4]
            from models_paper.llama_kivi import LMEvalLlamaForCausalLM_KIVI
            model = LMEvalLlamaForCausalLM_KIVI(
                k_bits=model_args.k_bits,
                v_bits=model_args.v_bits,
                group_size=model_args.group_size,
                residual_length=model_args.residual_length,
                pretrained=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                dtype=dtype,
                batch_size=data_args.batch_size,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
    elif is_mistral:
        if model_args.k_bits == 16 and model_args.v_bits == 16:
            from models_paper.modeling_mistral import LMEvalMistralForCausalLM
            model = LMEvalMistralForCausalLM(
                k_bits=model_args.k_bits,
                v_bits=model_args.v_bits,
                group_size=model_args.group_size,
                residual_length=model_args.residual_length,
                pretrained=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                dtype=dtype,
                batch_size=data_args.batch_size,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_fast_tokenizer=False,
            )
        else:
            assert model_args.k_bits in [2, 4] and model_args.v_bits in [2, 4]
            from models_paper.mistral_kivi import LMEvalMistralForCausalLM_KIVI
            model = LMEvalMistralForCausalLM_KIVI(
                k_bits=model_args.k_bits,
                v_bits=model_args.v_bits,
                group_size=model_args.group_size,
                residual_length=model_args.residual_length,
                pretrained=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                dtype=dtype,
                batch_size=data_args.batch_size,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_fast_tokenizer=False,
            )
    else:
        raise NotImplementedError(f"Model {model_args.model_name_or_path} not supported. Use Llama or Mistral.")
    # model = model.eval().cuda()

    if data_args.tasks is not None:
        initialize_tasks()
        tasks_list = data_args.tasks.split(",")
        task_names = utils.pattern_match(tasks_list, ALL_TASKS)
        for task in [task for task in tasks_list if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task
            for task in tasks_list
            if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used

        if task_missing:
            missing = ", ".join(task_missing)
            raise ValueError(
                f"Tasks {missing} were not found. Try `lm-eval --tasks list` for list of available tasks."
            )
        results = evaluator.simple_evaluate(
            model=model,
            tasks=task_names,
            log_samples=False,
        )
        print(evaluator.make_table(results))

        # Save results
        os.makedirs("logs", exist_ok=True)
        model_short = model_args.model_name_or_path.rstrip("/").split("/")[-1].lower()
        bits_tag = f"_kivi{model_args.k_bits}bit_res{model_args.residual_length}"
        out_name = f"{'_'.join(tasks_list)}_{model_short}{bits_tag}_paper_results.json"
        out_path = os.path.join("logs", out_name)
        with open(out_path, "w") as f:
            json.dump(results["results"], f, indent=2)
        print(f"\nSaved: {out_path}")