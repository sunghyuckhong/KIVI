"""
Generate self-calib data: model's own assistant responses to user prompts.

For each sample in source dataset, take the user-side messages, generate an
assistant response via vLLM, and save (user + generated assistant) as a JSONL
record with HF-datasets-compatible 'messages' format. The output JSONL can
then be passed to run_smoothkv_calibrate.py via --dataset to calibrate on
on-distribution generated text.
"""
import argparse, json, os, sys
from datasets import load_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--source_dataset", default="neuralmagic/LLM_compression_calibration")
    p.add_argument("--source_split", default="train")
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--max_gen_toks", type=int, default=1024,
                   help="Max generation per sample. Bigger = better calib match but slower.")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--output", required=True, help="Output JSONL")
    p.add_argument("--max_user_chars", type=int, default=4000,
                   help="Skip prompts longer than this many chars (avoids OOM).")
    return p.parse_args()


def extract_user_turns(messages):
    """Take only the user-side messages, drop any pre-existing assistant turns."""
    out = []
    for m in messages:
        if m.get("role") == "user":
            out.append({"role": "user", "content": m.get("content", "")})
            break  # use just the first user turn — typical instruction-style
    return out


def main():
    args = parse_args()
    print(f"Loading source dataset: {args.source_dataset}")
    ds = load_dataset(args.source_dataset, split=args.source_split)
    print(f"  total samples: {len(ds)}")

    # Build user-only prompts up to num_samples
    print(f"Extracting up to {args.num_samples} user prompts...")
    prompts = []
    for i in range(len(ds)):
        if len(prompts) >= args.num_samples:
            break
        msgs = ds[i].get("messages")
        if not msgs:
            continue
        user = extract_user_turns(msgs)
        if not user:
            continue
        if len(user[0]["content"]) > args.max_user_chars:
            continue
        prompts.append(user)
    print(f"  collected: {len(prompts)} user prompts")

    # vLLM batched generation
    from vllm import LLM, SamplingParams
    print(f"Loading vLLM with model={args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=64,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_gen_toks,
    )

    # Use chat-template'd prompts
    print(f"Generating {len(prompts)} responses, max_gen={args.max_gen_toks}...")
    outputs = llm.chat(prompts, sp)

    # Save (user + generated assistant) JSONL
    print(f"Writing to {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_written = 0
    with open(args.output, "w") as fh:
        for prompt, out in zip(prompts, outputs):
            text = out.outputs[0].text if out.outputs else ""
            if not text.strip():
                continue
            messages = list(prompt) + [{"role": "assistant", "content": text}]
            fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"  wrote {n_written}/{len(prompts)} samples")


if __name__ == "__main__":
    main()
