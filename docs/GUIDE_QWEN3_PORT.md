# Guide: Port KIVI/SmoothKV vLLM stack to Qwen3-8B (on a fresh pod)

Hand this to another Claude (or yourself) to reproduce the max-based SmoothKV /
pure-max / pertoken-INT4 / FP8-KV-cache ablation pipeline on **Qwen3-8B**. The
existing repo is tuned for Llama-3 / Mistral / DSR1-Distill-Llama-8B, all of
which go through vLLM 0.6.6's `LlamaForCausalLM` code path. **Qwen3 is a
different model class** — needs a newer vLLM and a minor patch port.

## What you are porting

We run KV-cache fake-quantization (quant→dequant on K/V post-RoPE) inside
vLLM's attention forward. Two components matter:

1. `vllm_custom/patches.py` — monkey-patches vLLM's Llama attention forward.
   Must be re-pointed at Qwen3's attention class in the newer vLLM version.
2. `run_smoothkv_calibrate.py` — monkey-patches HF `apply_rotary_pos_emb` to
   capture post-RoPE Q, K. Must also match Qwen3's HF structure.

Everything else (task YAMLs, launcher, aggregation scripts) is model-agnostic.

## Prerequisites

```bash
git clone https://github.com/sunghyuckhong/KIVI.git
cd KIVI
git checkout pertoken-experiments   # branch with vLLM scaffold + per-task sizing
export HF_TOKEN=<your_hf_token>     # for gated datasets + Qwen3
```

## Environment — **separate** env so you don't break the current Llama3 runs

Current `/opt/vllm_env` uses vLLM 0.6.6 + transformers 4.45.2 (pinned together).
Qwen3 needs vLLM 0.8+ (which supports `Qwen3ForCausalLM`) and transformers 4.51+.

```bash
python3.10 -m venv /opt/vllm_qwen3_env
/opt/vllm_qwen3_env/bin/pip install --upgrade pip setuptools wheel
# vLLM that supports Qwen3 (pick the current stable compatible with your CUDA)
/opt/vllm_qwen3_env/bin/pip install "vllm>=0.8.0" "transformers>=4.51" \
    "lm_eval==0.4.2" accelerate datasets sentencepiece "numpy<2"

# KIVI's triton kernel for the fake-quant K/V path (same one the Llama path uses)
cd quant && /opt/vllm_qwen3_env/bin/pip install -e . && cd ..
```

Sanity check:
```bash
/opt/vllm_qwen3_env/bin/python -c "
import vllm, transformers
print(f'vllm={vllm.__version__}  transformers={transformers.__version__}')
from vllm.model_executor.models import ModelRegistry
archs = [a for a in ModelRegistry.get_supported_archs() if 'Qwen3' in a]
print('Qwen3 archs:', archs)
"
```

Expected:
```
vllm=0.8.x  transformers=4.51+
Qwen3 archs: ['Qwen3ForCausalLM', ...]
```

If no `Qwen3ForCausalLM`, bump vLLM version.

## Port 1 — `vllm_custom/patches.py` to Qwen3

Our patches.py targets:
```python
import vllm.model_executor.models.llama as _vllm_llama
_vllm_llama.LlamaAttention.forward = patched_forward
```

For Qwen3, open the vLLM installed module to find the attention class:
```bash
ls /opt/vllm_qwen3_env/lib/python3.10/site-packages/vllm/model_executor/models/ | grep -i qwen
```

Expect `qwen3.py` (or similar). Open it, find the class that subclasses
`nn.Module` and has the attention `forward` method — usually called something
like `Qwen3Attention`. Check its forward signature:

```python
# Llama 0.6.6 forward:  (self, positions, hidden_states, kv_cache, attn_metadata)
# Qwen3 0.8+  forward:  likely same, but VERIFY — vLLM 0.7+ changed the API
```

**Action**: in `vllm_custom/patches.py`, replace every reference to
`_vllm_llama.LlamaAttention` with `_vllm_qwen3.Qwen3Attention` (or whatever the
exact class is). Keep the quant hook body (`q, k, v = qkv.split(...)` →
`rotary_emb` → fake-quant → `self.attn(...)`) — the pattern is identical
across Llama-family architectures.

**Signature drift**: vLLM 0.7.x introduced a new attention API where forward
takes only `(positions, hidden_states)` and returns the output directly, with
kv_cache accessed via thread-locals. If so, the patched forward needs to match.
Easiest way to see the exact signature: `inspect.signature(Qwen3Attention.forward)`.

## Port 2 — `run_smoothkv_calibrate.py` for Qwen3

The calibration script monkey-patches HF's `apply_rotary_pos_emb` to capture
post-RoPE Q, K:

```python
# run_smoothkv_calibrate.py around line 176
import transformers.models.llama.modeling_llama as ll
import transformers.models.mistral.modeling_mistral as mm
orig_apply_rope_llama = ll.apply_rotary_pos_emb
orig_apply_rope_mistral = mm.apply_rotary_pos_emb
```

**Action**: add Qwen3:

```python
import transformers.models.qwen3.modeling_qwen3 as qw
orig_apply_rope_qwen3 = qw.apply_rotary_pos_emb

def patched_rope_qwen3(q, k, cos, sin, position_ids=None, *a, **kw):
    q_rot, k_rot = orig_apply_rope_qwen3(q, k, cos, sin, position_ids, *a, **kw)
    # dispatch by layer_idx — see the existing Llama version for the pattern
    collector.update_q(layer_idx, q_rot)
    collector.update_k(layer_idx, k_rot)
    return q_rot, k_rot

qw.apply_rotary_pos_emb = patched_rope_qwen3
```

If Qwen3's attention uses a different signature for `apply_rotary_pos_emb`
(e.g. passes `unsqueeze_dim` explicitly), match it. Double-check by
`inspect.signature(qw.apply_rotary_pos_emb)` before patching.

Also verify Qwen3's model config:
```python
from transformers import AutoConfig
c = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
print(c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.hidden_size)
```

Make sure the `StatCollector` in `run_smoothkv_calibrate.py` gets these
dimensions right.

## Measure max prompt lengths for your tasks

Already done for Llama3 in the repo (gsm8k max 1404, gpqa 2798, math500 1373).
Qwen3's tokenizer may produce different counts.

```bash
/opt/vllm_qwen3_env/bin/python measure_prompt_lens.py \
    --tokenizer Qwen/Qwen3-8B \
    --tasks gsm8k_32k gpqa_diamond_cot_n_shot_32k math500_32k \
    --max_gen_toks 32768
```

Note the `max` column per task — you'll plug these into the launcher.

## Add Qwen3 preset to `scripts/launch_vllm_reasoning.sh`

In the `case "$PRESET"` block, add:

```bash
  qwen3-8b)
    MODEL=Qwen/Qwen3-8B
    MG=32768          # per max_new_tokens rule: Qwen3 ctx=128K > 32K → cap at 32K
    MAX_NS=<TBD>      # compute: 280k / (max_prompt + 32768), round down
    USE_TASK_LEN=1
    ;;
```

In the `task_max_len()` function, if Qwen3's prompt lengths differ meaningfully
from Llama3's, either branch on a `MODEL_FAMILY` var or add a fallback using
the measured values.

## Running — same workflow as the Llama3 pipeline

Once patches + preset are in place:

```bash
# 1. Calibrate
CUDA_VISIBLE_DEVICES=0 /opt/vllm_qwen3_env/bin/python run_smoothkv_calibrate.py \
    --model_path Qwen/Qwen3-8B \
    --num_samples 512 --seq_length 2048 --samples_per_channel 300000 \
    --output logs/calib/smoothkv_qwen3-8b_perc_ns512.pt

# 2. Generate puremax α=β=1 variant (direct python, same as the Llama3 pattern)
/opt/vllm_qwen3_env/bin/python - <<'PY'
import torch
base = torch.load("logs/calib/smoothkv_qwen3-8b_perc_ns512.pt",
                  weights_only=True, map_location="cpu")
eps = 1e-5
s_K = base["max_k"].clamp(min=eps)
L, nh, D = s_K.shape
s_K = s_K.view(L, nh, D // 2, 2).max(dim=-1, keepdim=True).values \
          .expand(L, nh, D // 2, 2).reshape(L, nh, D).clone()
s_V = base["max_v"].clamp(min=eps)
torch.save({**base, "s_K": s_K, "s_V": s_V, "alpha": 1.0, "beta": 1.0,
            "pair_max_k": True},
           "logs/calib/smoothkv_qwen3-8b_perc_ns512_puremax_a1b1_pair.pt")
PY

# 3. Launch 3 downstream streams (1 GPU per task, 16k or 32k max_gen_toks)
CALIB=logs/calib/smoothkv_qwen3-8b_perc_ns512_puremax_a1b1_pair.pt
STEM=qwen3-8b_smoothkv_g128_perc_ns512_puremax_a1b1_pair
TASKS='gsm8k_32k' bash scripts/launch_vllm_reasoning.sh 0 qwen3-8b pX_qwen3_skv_gsm8k   "--model smoothkv --calib_path $CALIB" "$STEM" &
TASKS='gpqa_diamond_cot_n_shot_32k' bash scripts/launch_vllm_reasoning.sh 1 qwen3-8b pX_qwen3_skv_gpqa    "--model smoothkv --calib_path $CALIB" "$STEM" &
TASKS='math500_32k' bash scripts/launch_vllm_reasoning.sh 2 qwen3-8b pX_qwen3_skv_math500 "--model smoothkv --calib_path $CALIB" "$STEM" &
```

## Adaptive max_gen_toks (optional)

To speed up evaluation with post-hoc truncation recovery:

```bash
MG_OVERRIDE=16384 LOG_SAMPLES=1 TASKS='gsm8k_32k' \
    bash scripts/launch_vllm_reasoning.sh ...
```

This sets `max_gen_toks=16384` and saves per-item generations to
`logs/<stem>_samples.json`. Parse those post-hoc for items where
`len(gen_tokens) == max_gen_toks`, rerun them at 32k, merge metrics.

## Key differences vs the Llama3 runs

| Aspect | Llama3 / Mistral / DSR1 | Qwen3-8B |
|---|---|---|
| vLLM version | 0.6.6 | 0.8+ |
| transformers | 4.45.2 | 4.51+ |
| env path | `/opt/vllm_env` | `/opt/vllm_qwen3_env` |
| attention class | `LlamaAttention` | `Qwen3Attention` |
| RoPE patch target | `llama.apply_rotary_pos_emb` | `qwen3.apply_rotary_pos_emb` |
| num_layers / heads / kv_heads | 32/32/8 | check config.json |

## Aggregation

`generate_vllm_report.py` auto-discovers any `logs/*_vllm_results.json` whose
filename matches the `{task}_{model_stem}_{method}` pattern. Add Qwen3's model
stem to `MODEL_DISPLAY` dict in `generate_vllm_report.py`:

```python
MODEL_DISPLAY = {
    ...
    "qwen3-8b": "Qwen3-8B",
}
```

After that, `python generate_vllm_report.py` will pick up Qwen3 runs
automatically alongside Llama3/Mistral/DSR1.

## If you get stuck

- Kernel compile errors on `quant/`: CUDA toolkit version mismatch. Match the
  torch wheel's CUDA to the system's `nvcc --version`.
- `Qwen3Attention.forward` signature doesn't match the patch: print the source
  (`inspect.getsource(Qwen3Attention.forward)`) and copy its exact signature
  before inserting the quant hook.
- Calibration OOM with `samples_per_channel=2097152` (full retention): drop to
  300000 — the reservoir is uniform-random so percentile estimates remain
  unbiased for max-based scales (which use only `max_k`/`max_v`, not the
  reservoir anyway).

## Files touched by this port

- `vllm_custom/patches.py` — re-point at `Qwen3Attention`
- `run_smoothkv_calibrate.py` — add Qwen3 to `monkey_patch_rope`
- `scripts/launch_vllm_reasoning.sh` — add `qwen3-8b` preset
- `measure_prompt_lens.py` — run once with Qwen3 tokenizer
- `generate_vllm_report.py` — add Qwen3 to `MODEL_DISPLAY`

Nothing in the Llama/Mistral path changes.
