import argparse
import subprocess
from pathlib import Path

# Map (task, mode) → eval script filename
FILENAME_MAP = {
    ("boolq", "zero_shot"): "eval_boolq_zero_shot_vllm.py",
    ("boolq", "three_shot"): "eval_boolq_three_shot_vllm.py",
    ("boolq", "cot"): "eval_boolq_cot_vllm.py",

    ("csqa", "zero_shot"): "eval_csqa_zero_shot_vllm.py",
    ("csqa", "three_shot"): "eval_csqa_three_shot_vllm.py",
    ("csqa", "cot"): "eval_csqa_cot_vllm.py",

    ("gsm8k", "zero_shot"): "eval_gsm8k_zero_shot_vllm.py",
    ("gsm8k", "three_shot"): "eval_gsm8k_three_shot_vllm.py",
    ("gsm8k", "cot"): "eval_gsm8k_cot_vllm.py",

    ("piqa", "zero_shot"): "eval_piqa_zero_shot_vllm.py",
    ("piqa", "three_shot"): "eval_piqa_three_shot_vllm.py",
    ("piqa", "cot"): "eval_piqa_cot_vllm.py",

    ("strategyqa", "zero_shot"): "eval_strategyqa_zero_shot_vllm.py",
    ("strategyqa", "three_shot"): "eval_strategyqa_three_shot_vllm.py",
    ("strategyqa", "cot"): "eval_strategyqa_cot_vllm.py",
}

def main():
    parser = argparse.ArgumentParser(description="URBench evaluation runner")
    parser.add_argument(
        "--task",
        choices=["boolq", "csqa", "gsm8k", "piqa", "strategyqa"],
        required=True,
        help="Which URBench task to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "three_shot", "cot"],
        required=True,
        help="Prompting mode to use.",
    )
    args, extra = parser.parse_known_args()

    key = (args.task, args.mode)
    if key not in FILENAME_MAP:
        raise ValueError(f"No eval script mapped for task={args.task}, mode={args.mode}")

    script_name = FILENAME_MAP[key]
    script_path = Path(__file__).resolve().parents[1] / "eval" / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Eval script not found: {script_path}")

    print(f"[URBench] Running: {script_path.name} (task={args.task}, mode={args.mode})")
    # Pass through any extra CLI args to the underlying script
    cmd = ["python", str(script_path), *extra]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
