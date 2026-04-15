#!/bin/bash
cd /workspace/sunghyuck/KIVI

EXPECTED_JSON=(
    "logs/gsm8k_kivi_results.json"
    "logs/gpqa_kivi_results.json"
    "logs/gsm8k_pertoken_results.json"
    "logs/gpqa_pertoken_results.json"
    "logs/gsm8k_pertoken_noresidual_results.json"
    "logs/gpqa_pertoken_noresidual_results.json"
    "logs/gsm8k_fp16_results.json"
    "logs/gpqa_fp16_results.json"
    "logs/gsm8k_pertoken_flat_results.json"
    "logs/gpqa_pertoken_flat_results.json"
    "logs/gsm8k_pertoken_flat_noresidual_results.json"
    "logs/gpqa_pertoken_flat_noresidual_results.json"
)

echo "Monitoring all 12 experiments (batch_size=32)..."

while true; do
    all_done=true
    done_count=0
    echo "$(date '+%H:%M:%S') status:"
    for f in "${EXPECTED_JSON[@]}"; do
        name=$(basename "$f" _results.json)
        if [[ -f "$f" ]]; then
            echo "  [DONE]    $name"
            ((done_count++))
        else
            echo "  [pending] $name"
            all_done=false
        fi
    done
    echo "  ($done_count / ${#EXPECTED_JSON[@]} complete)"
    echo ""

    if $all_done; then
        echo "All experiments complete — generating final report..."
        /anaconda/envs/kivi/bin/python generate_report.py | tee logs/final_report.txt
        break
    fi
    sleep 300
done
