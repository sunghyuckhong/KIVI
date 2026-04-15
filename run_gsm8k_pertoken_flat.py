"""GSM8K — per-token keys+values, group_size=128 (flat: 1 scale per token), residual=32."""
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
config.group_size = 128   # = head_dim → 1 scale per token, no grouping
config.residual_length = 32
config.use_flash = False

print("Loading KIVI-PerToken-Flat Mistral-7B-Instruct-v0.2 (group=128, residual=32)...")
model = MistralForCausalLM_KIVI_PerToken.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)

print("\nRunning GSM8K (5-shot, per-token flat, residual=32)...\n")
results = simple_evaluate(
    model=lm,
    tasks=["gsm8k"],
    num_fewshot=5,
    batch_size=16,
    log_samples=False,
)
print(utils.make_table(results))
with open("logs/gsm8k_pertoken_flat_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
print("Saved: logs/gsm8k_pertoken_flat_results.json")
