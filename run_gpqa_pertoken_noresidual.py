"""GPQA Diamond Generative N-Shot — per-token keys+values, residual=0 (no FP16 buffer)."""
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
config.residual_length = 0   # no FP16 residual buffer
config.use_flash = False

print("Loading KIVI-PerToken (no residual) Mistral-7B-Instruct-v0.2 for GPQA...")
model = MistralForCausalLM_KIVI_PerToken.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)

print("\nRunning GPQA Diamond Generative N-Shot (per-token KV, no residual)...\n")
results = simple_evaluate(
    model=lm,
    tasks=["gpqa_diamond_generative_n_shot"],
    batch_size=16,
    log_samples=False,
)
print(utils.make_table(results))
with open("logs/gpqa_pertoken_noresidual_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
print("Saved: logs/gpqa_pertoken_noresidual_results.json")
