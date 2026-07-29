#!/usr/bin/env python3
"""
efbpt_train_qlora.py

QLoRA supervised fine-tuning for EFBPT Plan A'. One condition, one seed, per run
(GPU quota is gres/gpu=1, so all runs are sequential).

Contract: docs/EFBPT_PLAN_A_FREEZE.md
  - LoRA: rank 16, alpha 32, dropout 0.05, lr 2e-4, 3 epochs, seeds 13/42/2026
  - AMENDMENT 3: serialization; AMENDMENT 4: fixed prompt

Loss is computed ONLY on target tokens. The prompt is masked with -100.
This is done explicitly here rather than via a library flag, so it is auditable.

TEST mode (--test) runs a handful of steps and then RELOADS the saved adapter
to prove the artifact is usable. TEST results are never banked.

Usage (inside a SLURM job, from ~/URBench):
  python eval/error_analysis_tests/efbpt/efbpt_train_qlora.py \
      --condition C3 --seed 13 --test
  python eval/error_analysis_tests/efbpt/efbpt_train_qlora.py \
      --condition C3 --seed 13
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------------------------------------------------------
# Frozen configuration. Any change here needs a dated amendment.
# ----------------------------------------------------------------------------

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"

TRAIN_FILES = {
    "C1": "data/strategyqa_official/efbpt/train/plan_a_train_c1_100.jsonl",
    "C2": "data/strategyqa_official/efbpt/train/plan_a_train_c2_100.jsonl",
    "C3": "data/strategyqa_official/efbpt/train/plan_a_train_c3_100.jsonl",
}

TRAIN_MD5 = {
    "C1": "7fa4472b4c98489c8d888c0abc9119b1",
    "C2": "fac4178ccc6b1a76ef14897bd5b718fc",
    "C3": "adb0515fa08742529faf345c1a43791c",
}

EXPECTED_ROWS = 100
VALID_SEEDS = {13, 42, 2026}

# LoRA (frozen by the freeze doc)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Optimization (frozen by the freeze doc)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# NOT in the freeze doc. Chosen once here and identical for all 9 runs.
# Recorded in experiments.md so it cannot drift between conditions.
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
MAX_SEQ_LEN = 1024

OUT_ROOT = "outputs/efbpt/plan_a/adapters"


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Dataset: prompt masked, loss on target only
# ----------------------------------------------------------------------------

class EFBPTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.examples = []
        self.n_truncated = 0
        self.lengths = []

        for row in rows:
            messages = [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]},
            ]
            # Thinking OFF, to match the frozen evaluation regime.
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            target_text = row["target"] + "<|im_end|>"
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + list(target_ids)

            self.lengths.append(len(input_ids))
            if len(input_ids) > max_len:
                self.n_truncated += 1
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

            if all(x == -100 for x in labels):
                die("qid %s: every label masked — prompt already fills max_len"
                    % row["qid"])

            self.examples.append(
                {"input_ids": input_ids, "labels": labels, "qid": row["qid"]}
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return {"input_ids": ex["input_ids"], "labels": ex["labels"]}


def make_collator(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
    return collate


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["C1", "C2", "C3"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--test", action="store_true",
                    help="short run: few steps, then reload the adapter to prove it loads")
    ap.add_argument("--test-steps", type=int, default=6)
    args = ap.parse_args()

    if not args.test and args.seed not in VALID_SEEDS:
        die("seed %d is not one of the frozen seeds %s" % (args.seed, sorted(VALID_SEEDS)))

    cond = args.condition
    train_path = TRAIN_FILES[cond]

    if not os.path.exists(train_path):
        die("training file missing: " + train_path)

    actual_md5 = md5_of_file(train_path)
    if actual_md5 != TRAIN_MD5[cond]:
        die("training file MD5 mismatch for %s\n  expected %s\n  actual   %s"
            % (cond, TRAIN_MD5[cond], actual_md5))
    print("[ok] %s training file verified, MD5 %s" % (cond, actual_md5))

    rows = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != EXPECTED_ROWS:
        die("%s has %d rows, expected %d" % (train_path, len(rows), EXPECTED_ROWS))

    for r in rows:
        if r["condition"] != cond:
            die("row %s has condition %s inside %s" % (r["qid"], r["condition"], train_path))

    set_seed(args.seed)

    tag = "%s_seed%d%s" % (cond, args.seed, "_TEST" if args.test else "")
    out_dir = os.path.join(OUT_ROOT, tag)
    if os.path.exists(out_dir) and not args.test:
        die("output already exists, refusing to overwrite: " + out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = EFBPTDataset(rows, tok, MAX_SEQ_LEN)
    lens = sorted(ds.lengths)
    print("[len] tokens per example: min=%d median=%d max=%d  truncated=%d/%d (max_seq_len=%d)"
          % (lens[0], lens[len(lens) // 2], lens[-1], ds.n_truncated, len(lens), MAX_SEQ_LEN))
    if ds.n_truncated > 0:
        die("%d example(s) exceed MAX_SEQ_LEN=%d. Truncated targets would corrupt "
            "training. Raise MAX_SEQ_LEN and record the change." % (ds.n_truncated, MAX_SEQ_LEN))

    # Sanity: show that masking really happened on example 0.
    ex0 = ds[0]
    n_sup = sum(1 for x in ex0["labels"] if x != -100)
    print("[mask] example 0: %d total tokens, %d supervised (prompt masked)"
          % (len(ex0["input_ids"]), n_sup))

    # ---- model ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",   # flash_attn is not installed
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- training args ----
    common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
    )
    if args.test:
        targs = TrainingArguments(max_steps=args.test_steps, **common)
    else:
        targs = TrainingArguments(num_train_epochs=NUM_EPOCHS, **common)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=make_collator(tok.pad_token_id),
    )

    result = trainer.train()
    loss = result.training_loss
    print("[train] final training loss: %r" % loss)
    if loss is None or loss != loss or loss in (float("inf"), float("-inf")):
        die("training loss is not a finite number — pipeline is broken")

    # ---- save adapter ----
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("[save] adapter written to %s" % out_dir)
    for fn in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p):
            print("   %-40s %d bytes" % (fn, os.path.getsize(p)))

    # ---- TEST only: prove the adapter reloads ----
    if args.test:
        print("\n[reload] freeing model and reloading the saved adapter ...")
        del trainer, model
        torch.cuda.empty_cache()

        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
            trust_remote_code=True,
        )
        reloaded = PeftModel.from_pretrained(base, out_dir)
        reloaded.eval()

        n_lora = sum(1 for n, _ in reloaded.named_parameters() if "lora_" in n)
        print("[reload] OK — %d LoRA tensors present in the reloaded model" % n_lora)
        if n_lora == 0:
            die("reloaded model has no LoRA tensors — the adapter is empty")

        # one short greedy generation, purely to prove the thing runs
        messages = [
            {"role": "system", "content": rows[0]["system"]},
            {"role": "user", "content": rows[0]["user"]},
        ]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = tok(text, return_tensors="pt").to(reloaded.device)
        with torch.no_grad():
            out = reloaded.generate(**enc, max_new_tokens=128, do_sample=False)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print("[reload] sample generation (TEST only, never banked):")
        print(repr(gen))

        print("\nTEST COMPLETE. Nothing here is a result.")


if __name__ == "__main__":
    main()
