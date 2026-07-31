#!/usr/bin/env python3
"""
d1_translate_facts.py

DIAGNOSTIC D1, Section H: translate the DEV200 English facts into Urdu so that
arm D can test knowledge with language held constant.

Design decisions, and why:

1. ONE fact per call. Batching several facts into a single prompt lets the model
   merge, drop, or reorder them. Each fact gets its own prompt.

2. Unique strings only. Identical English facts must receive identical Urdu, or
   the same knowledge would be presented differently across rows. Facts are
   deduplicated, translated once, then mapped back.

3. Named entities are written in Urdu FOLLOWED BY the English in parentheses,
   e.g.  سنیپ کیپ (SnapCap).
   Reason: this project's core disease is entity corruption during
   transliteration. Pure transliteration risks poisoning arm D with brand-new
   corruptions, which would look like evidence for a language bottleneck when
   it is really a translation artifact. Keeping the English alongside preserves
   identity exactly, while the Urdu form gives a lexical bridge to the Urdu
   question. No knowledge is lost either way.

4. Numbers stay as Western digits, and a check verifies the digit set survives.

5. Every Urdu string that this script sends into a prompt is validated to
   contain only Arabic-block, ASCII, and whitespace characters. A Cyrillic or
   Greek lookalike (already found once in this project) aborts the run.

Nothing is written unless every translation passes its checks. Failures are
reported for manual inspection instead of being silently accepted.

Usage (inside a SLURM job, from ~/URBench):
  python eval/error_analysis_tests/efbpt/d1_translate_facts.py --test
  python eval/error_analysis_tests/efbpt/d1_translate_facts.py
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import torch

MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
DEV_PATH = "data/strategyqa_official/dev200_seed4242.jsonl"

OUT_DIR = "data/strategyqa_official/efbpt"
OUT_JSON = os.path.join(OUT_DIR, "d1_facts_ur.json")
REVIEW_DIR = "outputs/efbpt/d1"
REVIEW_FILE = os.path.join(REVIEW_DIR, "d1_translation_review.txt")
FLAGGED_FILE = os.path.join(REVIEW_DIR, "d1_translation_flagged.txt")

MAX_NEW_TOKENS = 512
REVIEW_SAMPLE = 40          # how many pairs to lay out for human checking

# ---------------------------------------------------------------------------
# Prompt. Urdu examples are short and simple so they can be verified by eye.
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a professional English-to-Urdu translator. "
    "You translate factual sentences accurately and completely."
)

# Example pair 1 has named entities; example pair 2 has none. The contrast
# teaches the model when to use parentheses and when not to.
EX1_EN = "Paris is the capital of France."
EX1_UR = "پیرس (Paris) فرانس (France) کا دارالحکومت ہے۔"

EX2_EN = "Bees produce honey."
EX2_UR = "شہد کی مکھیاں شہد پیدا کرتی ہیں۔"

INSTRUCTION = (
    "Translate the English sentence into natural, fluent Urdu.\n"
    "\n"
    "Rules:\n"
    "1. Translate the meaning completely. Do not add or remove any information.\n"
    "2. For a named entity (person, place, company, brand, title, organisation), "
    "write the Urdu form and then the original English in round brackets. "
    "Example: SnapCap becomes سنیپ کیپ (SnapCap).\n"
    "3. If the sentence has no named entity, use no brackets.\n"
    "4. Keep all numbers as ordinary digits 0-9.\n"
    "5. Output ONLY the Urdu translation, on one line. "
    "No explanation, no English sentence, no quotation marks, no labels.\n"
)

ARABIC_LO, ARABIC_HI = 0x0600, 0x06FF
ARABIC_SUPP_LO, ARABIC_SUPP_HI = 0x0750, 0x077F   # Arabic Supplement
PRES_A_LO, PRES_A_HI = 0xFB50, 0xFDFF             # Presentation Forms-A
PRES_B_LO, PRES_B_HI = 0xFE70, 0xFEFF             # Presentation Forms-B


def die(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(1)


def is_arabic(ch):
    c = ord(ch)
    return (ARABIC_LO <= c <= ARABIC_HI
            or ARABIC_SUPP_LO <= c <= ARABIC_SUPP_HI)


def arabic_count(s):
    return sum(1 for ch in s if is_arabic(ch))


def validate_prompt_urdu(name, s):
    """Abort if a prompt-side Urdu string holds a non-Arabic lookalike."""
    bad = []
    for i, ch in enumerate(s):
        c = ord(ch)
        ok = (c < 128) or is_arabic(ch) or ch.isspace()
        if not ok:
            bad.append((i, ch, "U+%04X" % c))
    if bad:
        die("prompt string %s contains %d suspicious character(s): %s\n"
            "A Cyrillic or Greek lookalike would silently corrupt every "
            "translation. Fix the source before running."
            % (name, len(bad), bad[:10]))
    print("[ok] prompt Urdu %s: %d chars, %d Arabic, no lookalikes"
          % (name, len(s), arabic_count(s)))


def build_prompt(tok, english):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTION + "\nEnglish: " + EX1_EN},
        {"role": "assistant", "content": EX1_UR},
        {"role": "user", "content": "English: " + EX2_EN},
        {"role": "assistant", "content": EX2_UR},
        {"role": "user", "content": "English: " + english},
    ]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)


# ---------------------------------------------------------------------------
# Quality checks on each translation
# ---------------------------------------------------------------------------

SOFT_PREFIX = "SOFT:"


def find_entities(english):
    """Capitalised words likely to be named entities.

    Non-first-word capitals are taken as entities. The FIRST word is included
    only when it has an internal capital (SnapCap, LendingTree, iPhone),
    because every sentence starts with a capital and treating that as an
    entity would flag almost everything.
    """
    ents = set(re.findall(r"\b[A-Z][A-Za-z]{2,}\b", english[1:]))
    first = re.match(r"\s*([A-Za-z][A-Za-z]{2,})", english)
    if first:
        w = first.group(1)
        if re.search(r"[a-z][A-Z]", w) or w.isupper():
            ents.add(w)
    return ents


def check_translation(english, urdu):
    """Return a list of problem strings. Empty list means the translation passed.

    Problems prefixed with SOFT: are advisory. They are written to the review
    file but do not block the run, because capitalisation heuristics produce
    false positives and blocking on them would bury the real failures.
    """
    problems = []

    u = urdu.strip()
    if u == "":
        problems.append("empty")
        return problems

    if "\n" in u:
        problems.append("multi_line")

    n_ar = arabic_count(u)
    if n_ar == 0:
        problems.append("no_urdu_characters")
    elif n_ar < 5:
        problems.append("almost_no_urdu (%d arabic chars)" % n_ar)

    # presentation forms indicate a broken/normalised encoding
    for ch in u:
        c = ord(ch)
        if PRES_A_LO <= c <= PRES_A_HI or PRES_B_LO <= c <= PRES_B_HI:
            problems.append("arabic_presentation_form U+%04X" % c)
            break

    # the model must not simply echo the English sentence back
    if english.strip().lower() in u.lower():
        problems.append("english_echoed_verbatim")

    # digits must survive
    src_digits = sorted(re.findall(r"\d+", english))
    tgt_digits = sorted(re.findall(r"\d+", u))
    if src_digits != tgt_digits:
        problems.append("digits_changed src=%s tgt=%s" % (src_digits, tgt_digits))

    # eastern-arabic numerals were forbidden
    if re.search(r"[\u0660-\u0669\u06F0-\u06F9]", u):
        problems.append("eastern_arabic_numerals")

    # length sanity: a real translation is not a fragment
    if len(u) < 0.4 * len(english):
        problems.append("suspiciously_short (%d vs %d)" % (len(u), len(english)))
    if len(u) > 4.0 * len(english):
        problems.append("suspiciously_long (%d vs %d)" % (len(u), len(english)))

    # named-entity preservation. Advisory only: the heuristic misfires on
    # ordinary capitalised words, and Section H's human review is the real
    # safeguard for entity quality.
    caps = find_entities(english)
    missing = [w for w in caps if w not in u]
    if missing:
        problems.append(SOFT_PREFIX + "entity_english_missing:%s"
                        % ",".join(sorted(missing)))

    return problems


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_all(model, tok, prompts, batch_size):
    n = len(prompts)
    order = sorted(range(n), key=lambda i: len(prompts[i]))
    out_text = [None] * n
    t0 = time.time()
    done = 0
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        batch = [prompts[i] for i in idxs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idxs):
            new_ids = gen[j][in_len:]
            out_text[i] = tok.decode(new_ids, skip_special_tokens=True).strip()
        done += len(idxs)
        el = time.time() - t0
        print("    %d/%d  (%.1fs, %.2fs/item)" % (done, n, el, el / done), flush=True)
    return out_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="translate only the first 12 unique facts; writes nothing")
    ap.add_argument("--test-items", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # ---- validate the Urdu we are about to put INTO the prompt ----
    validate_prompt_urdu("EX1_UR", EX1_UR)
    validate_prompt_urdu("EX2_UR", EX2_UR)
    print("[ok] EX1_UR codepoints: %s"
          % " ".join("%04X" % ord(c) for c in EX1_UR))
    print("[ok] EX2_UR codepoints: %s"
          % " ".join("%04X" % ord(c) for c in EX2_UR))

    if not os.path.exists(DEV_PATH):
        die("DEV200 missing: " + DEV_PATH)

    rows = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("[data] %d DEV200 rows" % len(rows))

    # ---- collect unique facts ----
    all_facts = []
    for r in rows:
        facts = r.get("urbench_facts") or []
        if not facts:
            die("row %s has no urbench_facts" % r.get("urbench_qid"))
        for fct in facts:
            if not isinstance(fct, str) or fct.strip() == "":
                die("row %s has an empty fact" % r.get("urbench_qid"))
            all_facts.append(fct)

    uniq = sorted(set(all_facts))
    print("[data] %d fact instances, %d unique strings" % (len(all_facts), len(uniq)))

    # facts must be English; a stray Urdu fact would break the premise of D1
    n_ar_facts = sum(1 for fct in uniq if arabic_count(fct) > 0)
    print("[data] unique facts already containing Urdu characters: %d (expected 0)"
          % n_ar_facts)

    targets = uniq[:args.test_items] if args.test else uniq
    print("[plan] translating %d strings%s"
          % (len(targets), " (TEST)" if args.test else ""))

    # ---- model ----
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("[load] base model, 4-bit nf4 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True
    model.eval()

    prompts = [build_prompt(tok, e) for e in targets]
    print("[prompt] example 0 (repr, first 600 chars):")
    print(repr(prompts[0][:600]))

    urdu = generate_all(model, tok, prompts, args.batch_size)

    # ---- check every translation ----
    mapping = {}
    flagged = []      # hard failures: block the write
    advisory = []     # soft failures: recorded, do not block
    for en, ur in zip(targets, urdu):
        probs = check_translation(en, ur)
        hard = [p for p in probs if not p.startswith(SOFT_PREFIX)]
        soft = [p for p in probs if p.startswith(SOFT_PREFIX)]
        if hard:
            flagged.append((en, ur, probs))
        else:
            mapping[en] = ur
            if soft:
                advisory.append((en, ur, soft))

    print("\n" + "=" * 70)
    print("TRANSLATION CHECK")
    print("=" * 70)
    print("passed (written)      : %d" % len(mapping))
    print("hard failures (blocked): %d" % len(flagged))
    print("advisory notes         : %d  (entity heuristic; review, not blocking)"
          % len(advisory))
    if flagged:
        from collections import Counter
        kinds = Counter(p.split(" ")[0].split(":")[0]
                        for _, _, ps in flagged for p in ps)
        print("flag reasons:")
        for k, v in kinds.most_common():
            print("   %-38s %d" % (k, v))

    os.makedirs(REVIEW_DIR, exist_ok=True)

    # flagged items go to a file for inspection; Urdu is never printed to stdout
    fl_path = FLAGGED_FILE + ("_TEST" if args.test else "")
    with open(fl_path, "w", encoding="utf-8") as f:
        f.write("FLAGGED TRANSLATIONS — open this file in the VS Code editor.\n")
        f.write("The terminal cannot render Urdu correctly; the editor can.\n\n")
        f.write("=== HARD FAILURES (these blocked the write) ===\n\n")
        for en, ur, probs in flagged:
            f.write("PROBLEMS: %s\n" % "; ".join(probs))
            f.write("EN: %s\n" % en)
            f.write("UR: %s\n\n" % ur)
        f.write("\n=== ADVISORY (accepted, but the English entity form was "
                "not found in the Urdu) ===\n\n")
        for en, ur, probs in advisory:
            f.write("NOTE: %s\n" % "; ".join(probs))
            f.write("EN: %s\n" % en)
            f.write("UR: %s\n\n" % ur)
        f.flush()
        os.fsync(f.fileno())
    print("\nflagged items -> %s" % fl_path)

    # human-review sample, per DIAGNOSTIC D1 Section H
    rv_path = REVIEW_FILE + ("_TEST" if args.test else "")
    sample = list(mapping.items())[:REVIEW_SAMPLE]
    with open(rv_path, "w", encoding="utf-8") as f:
        f.write("HUMAN REVIEW SAMPLE — open this file in the VS Code editor.\n")
        f.write("Check: is the Urdu accurate, natural, and complete?\n")
        f.write("Are named entities correct, with the English in brackets?\n")
        f.write("Section H of DIAGNOSTIC D1 requires this check BEFORE arm D\n")
        f.write("is interpreted. Record the sample size and outcome.\n\n")
        for i, (en, ur) in enumerate(sample, 1):
            f.write("--- %d ---\nEN: %s\nUR: %s\n\n" % (i, en, ur))
        f.flush()
        os.fsync(f.fileno())
    print("review sample (%d items) -> %s" % (len(sample), rv_path))

    if args.test:
        print("\nTEST COMPLETE. Nothing was written to data/. "
              "Open the two files above in the editor and read the Urdu.")
        return

    if flagged:
        die("%d translation(s) failed their checks. Nothing written to %s.\n"
            "Inspect %s, then decide: fix the prompt and re-run, or accept "
            "specific items explicitly. Do NOT let a bad translation into "
            "arm D — Section H warns it would masquerade as evidence that the "
            "bottleneck is language." % (len(flagged), OUT_JSON, fl_path))

    if len(mapping) != len(uniq):
        die("mapping has %d entries but there are %d unique facts"
            % (len(mapping), len(uniq)))

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_JSON):
        die("output already exists, refusing to overwrite: " + OUT_JSON)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    with open(OUT_JSON, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    reread = json.load(open(OUT_JSON, "r", encoding="utf-8"))
    if len(reread) != len(mapping):
        die("verification failed: wrote %d entries, disk has %d"
            % (len(mapping), len(reread)))

    print("\nWROTE %s" % OUT_JSON)
    print("  entries : %d" % len(reread))
    print("  MD5     : %s" % md5)
    print("\nRecord this MD5 in experiments.md and in DIAGNOSTIC D1 Section H.")
    print("Next: read the review sample in the editor BEFORE running arm D.")


if __name__ == "__main__":
    main()
