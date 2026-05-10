# Notes for Claude (and humans editing the kv_fake_quant kernels)

## Two vllm forks — keep `kv_fake_quant/` in sync

Editing **only one** of the two vllm forks below silently leaves the other on
old code. We hit this 2026-05-10: a kernel "fix" landed only in
`vllm-exaone-fork`, so non-EXAONE models on `.venv` ran the OLD kernel for an
entire sweep before tests caught it.

| venv | Editable install path | Used by |
|---|---|---|
| `.venv`         | `/workspace/sunghyuck/vllm-compression-part/vllm` | Qwen3-8B/30B-A3B/32B, Mistral, Llama-3 (everyone except EXAONE) |
| `.venv-exaone`  | `/workspace/sunghyuck/vllm-exaone-fork/vllm`     | EXAONE-4.5-33B only (needs nuxlear/transformers) |

The exact mappings are in
`./.venv/lib/python3.10/site-packages/__editable___vllm_*finder.py` (and the
analogous file under `.venv-exaone/`). `vllm.__file__` will tell you which
copy is actually loaded.

**Whenever you edit anything under
`vllm/model_executor/layers/quantization/kv_fake_quant/`, apply the SAME
edit to BOTH forks and commit each.** Until they share a base, this is a
manual mirror.

Quick sync check:
```
bash scripts/check_vllm_kernels_in_sync.sh
```
exits non-zero if the two `kv_fake_quant/` trees differ. Run before launching
NVFP4 sweeps if you've recently touched the kernel.

## Verifying NVFP4 changes

`tests/test_nvfp4_bitidentity.py` locks down 32 cases (FP4 round, full quant
+ dequant pipeline, edge cases incl. NaN/Inf, FP8 over/underflow, realistic
K-cache shapes) against the canonical reference
`vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils.ref_nvfp4_quant`.

```
.venv/bin/python tests/test_nvfp4_bitidentity.py        # standalone
.venv-exaone/bin/python tests/test_nvfp4_bitidentity.py  # also against EXAONE venv
```

Run BOTH after a kv_fake_quant edit — they exercise the kernel from each
venv separately and will catch the "only one fork updated" trap.

## EXAONE-4.5 quirks

- Setup order in `.venv-exaone`: vllm cu129 first, then nuxlear/transformers
  + huggingface_hub>=1.3 with `--no-deps`. See `scripts/_setup_venv_exaone.sh`.
- EXAONE-4.5 has **hybrid attention**: sliding-window layers go through
  `apply_rotary_pos_emb`, but global layers use NoPE (no RoPE applied — see
  `transformers.models.exaone4_5.modeling_exaone4_5:428`). The K-capture hook
  in `calib/hooks.py` only fires for layers that pass through
  `apply_rotary_pos_emb`, so 16/64 EXAONE layers end up with `max_k=0` in the
  calib. `derive_nvfp4_global_scales.py` then produces `gs_K_smooth ≈ 1e33`
  for those layers, which produces 0.0/garbage outputs from NVFP4 SmoothKV.
  TODO: add a `k_proj` forward hook so global-NoPE K is captured too.

  Workaround: avoid running EXAONE NVFP4 SmoothKV sweeps until the hook is
  extended. EXAONE int4 SmoothKV is unaffected (s_K floor of 1e-5 means raw K
  flows through INT4 quant cleanly).
