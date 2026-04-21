# Batch-size ceilings for paper_env_fast on A100-SXM4-80GB

Empirically measured `--batch_size` ceiling that (a) does not OOM and (b) produces
numbers within 0.5 pp of bs=1 in `paper_env_fast` (transformers 4.36.2 + lm-eval 0.4.2).
Higher bs = faster wallclock. MATH500/GPQA-n-shot use long generation (1024 tokens)
and may be tighter than TQA (256 tokens).

| date | model | task | hardware | bs tested | bs=1 score | largest-safe-bs | score at max bs |
|---|---|---|---|---|---|---|---|
| 2026-04-20 | Llama-2-7B-hf FP16 | truthfulqa_gen | A100-SXM4-80GB | 32,64,128 | 30.74 | 128 | 30.77 |
