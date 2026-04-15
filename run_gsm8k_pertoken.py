"""
Run full GSM8K with per-token KV cache quantization (both keys and values).

Difference from run_gsm8k.py (KIVI default):
  KIVI default : keys per-channel, values per-token
  This script  : keys per-token,   values per-token
"""
import warnings, json
warnings.filterwarnings("ignore")
import torch
from transformers import MistralConfig, AutoTokenizer
from models.mistral_kivi_pertoken import MistralForCausalLM_KIVI_PerToken
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate, utils

MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

config = MistralConfig.from_pretrained(MODEL_PATH)
config.k_bits = 2
config.v_bits = 2
config.group_size = 32
config.residual_length = 32
config.use_flash = False   # V100 (sm_70) does not support FlashAttention 2

print("Loading KIVI-PerToken Mistral-7B-Instruct-v0.2 (2-bit, per-token KV)...")
model = MistralForCausalLM_KIVI_PerToken.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)

print("\nRunning GSM8K (5-shot, per-token KV quant)...\n")
results = simple_evaluate(
    model=lm,
    tasks=["gsm8k"],
    num_fewshot=5,
    batch_size=16,
    log_samples=False,
)
print(utils.make_table(results))
with open("logs/gsm8k_pertoken_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
print("Saved: logs/gsm8k_pertoken_results.json")
