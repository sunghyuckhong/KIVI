"""
Run full GSM8K evaluation using lm-eval with KIVI 2-bit KV cache quantization.
Uses Mistral-7B-Instruct-v0.2 (locally cached) with KIVI's tuple-based KV cache.
"""
import warnings, json
warnings.filterwarnings("ignore")
import torch
from transformers import MistralConfig, AutoTokenizer
from models.mistral_kivi import MistralForCausalLM_KIVI
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate, utils

MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

# ── KIVI config ────────────────────────────────────────────────────────────────
config = MistralConfig.from_pretrained(MODEL_PATH)
config.k_bits = 2
config.v_bits = 2
config.group_size = 32
config.residual_length = 32
config.use_flash = False   # V100 (sm_70) does not support FlashAttention 2

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading KIVI Mistral-7B-Instruct-v0.2 (2-bit KV cache)...")
model = MistralForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()  # single GPU — 7B fp16 fits in one 32GB V100
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# ── Wrap in HFLM ───────────────────────────────────────────────────────────────
# HFLM accepts a pre-initialized model as `pretrained` (non-string path)
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)

# ── Run GSM8K (5-shot, greedy) ─────────────────────────────────────────────────
print("\nRunning GSM8K (5-shot)...\n")
results = simple_evaluate(
    model=lm,
    tasks=["gsm8k"],
    num_fewshot=5,
    batch_size=16,
    log_samples=False,
)

print(utils.make_table(results))
with open("logs/gsm8k_kivi_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
print("Saved: logs/gsm8k_kivi_results.json")
