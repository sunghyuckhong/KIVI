"""GSM8K — full-precision FP16 baseline (no KV quantization)."""
import warnings, json
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate, utils

MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading FP16 Mistral-7B-Instruct-v0.2 (no quantization)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True
).cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)

print("\nRunning GSM8K (5-shot, FP16 baseline)...\n")
results = simple_evaluate(
    model=lm,
    tasks=["gsm8k"],
    num_fewshot=5,
    batch_size=16,
    log_samples=False,
)
print(utils.make_table(results))
with open("logs/gsm8k_fp16_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
print("Saved: logs/gsm8k_fp16_results.json")
