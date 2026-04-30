# Reproducing Llama-3-8B-Instruct & Mistral-7B-Instruct-v0.2 vLLM-path results

This is the canonical configuration that produces the headline tables for the
two non-Qwen3 models on the vLLM evaluation path. If you re-pull from the
`logs/` directory and your numbers don't match, the most likely cause is one of
the per-task config decisions documented below.

## Per-task chat-template policy

This is the most-overlooked detail. Both models are instruct-tuned but the
canonical sweep does NOT apply the chat template uniformly.

| Model | Task | Chat template | Filename suffix |
|---|---|---|---|
| Llama-3-8B-Instruct | gsm8k_32k | **off** | `_vllm_results.json` |
| Llama-3-8B-Instruct | math500_32k | **off** | `_vllm_results.json` |
| Llama-3-8B-Instruct | gpqa_main_cot_n_shot_32k | **off** | `_vllm_results.json` |
| Llama-3-8B-Instruct | ifeval | **on** | `_chat_vllm_results.json` |
| Mistral-7B-Instruct-v0.2 | gsm8k_32k | **off** | `_vllm_results.json` |
| Mistral-7B-Instruct-v0.2 | math500_32k | **off** | `_vllm_results.json` |
| Mistral-7B-Instruct-v0.2 | gpqa_main_cot_n_shot_32k | **off** | `_vllm_results.json` |

So Llama is "no chat except ifeval"; Mistral is "no chat anywhere". The earlier
`_chat_vllm_results.json` files in `logs/` for these models are non-canonical
parallel runs and do NOT match the screenshot tables — pulling those gives
different numbers.

The flag controlling this is `--apply_chat_template` to `run_eval_vllm.py`. Omit
it for the no-chat tasks; pass it for ifeval on Llama.

## SmoothKV calibration variant

For these rotate_half-RoPE models, the canonical SmoothKV variant is
`puremax_a1b1_halfpair_slim`:

- α = 1, β = 1 (max-based, no temperature on the K/V scale split)
- `_halfpair`: pair-max `s_K[i] == s_K[i + D/2]` — the rotate_half pairing.
  **Required** for Llama-3 / Mistral / Qwen3. The older `_pair` variant
  (`s_K[2i] == s_K[2i+1]`) is for interleaved RoPE (KIVI's original Llama-2
  era) and produces broken numbers on rotate_half models — see
  `scripts/make_alpha_variants.py` `--pair_max_k` docstring.
- `_slim`: deployment-trimmed payload (just `s_K`, `s_V`, no extra raw stats)
- n_calib = 512 samples
- calibration dataset: `neuralmagic/LLM_compression_calibration` (`perc`)

Calib filenames:
```
logs/calib/smoothkv_meta-llama-3-8b-instruct_bf16_perc_ns512_puremax_a1b1_halfpair_slim.pt
logs/calib/smoothkv_mistral-7b-instruct-v0.2_bf16_perc_ns512_puremax_a1b1_halfpair_slim.pt
```

Two-step build pipeline (one-time, takes ~15-30 min on a single A100):
```bash
# 1. raw stats (max_q, max_k, max_v, has_qk_norm)
python run_smoothkv_calibrate.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --num_samples 512 --seq_length 2048 \
    --output logs/calib/smoothkv_meta-llama-3-8b-instruct_bf16_perc_ns512_base.pt

# 2. produce α=1, half-pair RoPE, slim deployment .pt
python scripts/make_alpha_variants.py \
    --base logs/calib/smoothkv_meta-llama-3-8b-instruct_bf16_perc_ns512_base.pt \
    --alphas 1.0 --betas 1.0 \
    --half_pair_max_k --slim
```

Same recipe with `mistralai/Mistral-7B-Instruct-v0.2` for the Mistral calib.

## Scoring

| Task | Metric reported | Source |
|---|---|---|
| gsm8k_32k | `exact_match,strict-match` (lm_eval `gsm8k_32k.yaml`) | the `_results.json` file |
| gpqa_main_cot_n_shot_32k | `exact_match,flexible-extract` | the `_results.json` file |
| math500_32k | minerva via `math_verify` (boxed extraction + sympy equivalence) | offline rescore |
| ifeval | `inst_level_strict_acc,none` | the `_results.json` file |

For math500 the lm_eval default `exact_match` (Minerva regex requiring
`Final Answer: ... I hope it is correct.`) does work for these two models because
they're not in thinking mode and emit that phrase reliably (their default em
matches minerva to within ~0.5pp). For the screenshot's exact numbers (29.40
vs default 28.80 for Llama bf16 etc.), use the offline rescorer:
```bash
/opt/vllm_qwen3_env/bin/python scripts/rescore_math500_math_verify.py \
    --samples logs/math500_32k_meta-llama-3-8b-instruct_bf16_vllm_samples.json
```

## Generation length

All four tasks run with MG ≤ 32k. For these 8B-class instruct models without
thinking mode, generations rarely exceed 1-2k tokens, so a single-pass run at
the task-yaml MG is sufficient — no adaptive 8k+32k merge needed (unlike the
Qwen3 thinking-mode tasks).

## Headline numbers (from screenshots, for verification)

### Llama-3-8B-Instruct (no chat ex. ifeval)

| Method | gsm8k strict | gpqa flex | math500 minerva | ifeval inst-strict (chat) |
|---|---:|---:|---:|---:|
| BF16 | 75.89 | 18.97 | 29.40 | 77.94 |
| FP8 g=128 | 75.36 | 19.64 | 29.40 | 77.34 |
| pertoken INT4 g=128 | 74.91 | 19.20 | 29.80 | 76.02 |
| SmoothKV (α=β=1, halfpair_slim, n_calib=512) | 76.19 | 20.09 | 30.00 | 77.10 |

### Mistral-7B-Instruct-v0.2 (no chat anywhere)

| Method | gsm8k strict | gpqa flex | math500 minerva |
|---|---:|---:|---:|
| BF16 | 43.97 | 21.88 | 9.60 |
| FP8 g=128 | 43.44 | 22.32 | 10.80 |
| pertoken INT4 g=128 | 43.82 | 22.10 | 10.20 |
| SmoothKV (α=β=1, halfpair_slim, n_calib=512) | 43.52 | 21.65 | 9.80 |

`gsm8k_32k` was run at num_fewshot=5 (per task yaml). The 8-shot variant
scores ~52% on Mistral bf16 (per the original screenshot footnote) but is not
the canonical setting in this repo's sweep.

## Common reasons numbers don't reproduce

1. **Pulled the `_chat_vllm_results.json` file by accident.** Both models also
   have a parallel chat-template-applied sweep in `logs/`; that's a different
   experiment. Stick to `_vllm_results.json` (no `_chat_`) for everything except
   Llama ifeval.
2. **Used `_pair` SmoothKV calib instead of `_halfpair_slim`.** `_pair` enforces
   the wrong RoPE pair-equal constraint on rotate_half models; produces wrong
   KV stats. Easy giveaway: SmoothKV row drops 4-5pp on gsm8k vs bf16.
3. **Used lm_eval default math500 scorer without minerva rescoring.** Default
   ≈ minerva for Llama/Mistral (within 0.5pp), but if you got ~0 across the
   board the model probably ran in chat-template mode and didn't emit the
   "Final Answer:" footer.
4. **Used `inst_level_loose_acc` instead of `inst_level_strict_acc` for ifeval.**
   loose is ~7pp higher than strict; screenshot uses strict.
5. **fewshot mismatch.** gsm8k_32k.yaml uses 5-shot. If you ran the standalone
   `gsm8k.yaml` (8-shot) you'll get the ~52% number for Mistral bf16 instead
   of 43.97.

## File paths cheat sheet

```
# Llama
logs/gsm8k_32k_meta-llama-3-8b-instruct_<quant>_vllm_results.json
logs/math500_32k_meta-llama-3-8b-instruct_<quant>_vllm_results.json
logs/gpqa_main_cot_n_shot_32k_meta-llama-3-8b-instruct_<quant>_vllm_results.json
logs/ifeval_meta-llama-3-8b-instruct_<quant>_chat_vllm_results.json   # chat for ifeval only

# Mistral
logs/<task>_mistral-7b-instruct-v0.2_<quant>_vllm_results.json   # no chat anywhere

# <quant> is one of:
#   bf16
#   fp8_g128
#   pertoken_int4_g128
#   smoothkv_g128_bf16_perc_ns512_puremax_a1b1_halfpair_slim
```
