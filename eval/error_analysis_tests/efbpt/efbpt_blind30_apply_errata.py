"""
efbpt_blind30_apply_errata.py — CPU-only. Applies the manual gold errata
to blind30_gold.jsonl EXACTLY as specified. Backs up first, logs errata,
validates, prints diff. Modifies ONLY the gold file. Reruns nothing.
"""
import json, shutil, sys
from pathlib import Path
from datetime import date

BASE  = Path("/mnt/home/user41/URBench")
EF    = BASE / "data/strategyqa_official/efbpt"
GOLD  = EF / "blind30_gold.jsonl"
CANDS = EF / "blind30_candidates.jsonl"
BACKUP = EF / f"blind30_gold_backup_{date.today().isoformat()}.jsonl"
ERRATA = EF / "blind30_gold_errata_log.jsonl"

# ---- manual verdicts, transcribed EXACTLY from the audit packet ----
# actions: ("ADD", title, span) | ("REMOVE", title) | ("FIXSPAN", title, span)
VERDICTS = {
    "038d2f23ebc149069a74": [("REMOVE", "Apple")],
    "16338eaba71f146a0c40": [("ADD", "Computer fan", "پنکھا")],
    "29ee7da0020eb03888fb": [("ADD", "Computer programming", "کوڈنگ")],
    "2bfe7f37f939ee456600": [("ADD", "Pig", "سوروں")],
    "59542ff1d7782e4cbd89": [("ADD", "Smartphone", "اسمارٹ فونز"),
                             ("FIXSPAN", "Android (operating system)", "اینڈرائیڈ")],
    "6042b48035952d0e1a61": [("ADD", "Astrology", "ماہر نجوم")],
    "6b86b4445e7fa97f52c6": [("ADD", "Academic degree", "کالج کی ڈگری")],
    "6f0b33d71d2e65d3b376": [("ADD", "Poet", "شاعر"),
                             ("ADD", "Islam", "اسلامی مذہب")],
    "722dc38bd849d8b6ec0f": [("ADD", "Reproduction", "تولید مثل"),
                             ("ADD", "Parent", "والدین بننے")],
    "af2d3c137bd3f5230012": [("REMOVE", "Office of Migrant Education"),
                             ("ADD", "Illegal immigration", "غیر دستاویزی تارکین وطن")],
    "a44c3d9161f5e5c3f41d": [("ADD", "Human", "انسانوں")],
    # all other rows = OK (no change), including Row 14 override to OK
}
NOTES = {
    "6042b48035952d0e1a61": ("source dataset factually questionable — Earth (5.51 g/cm3) is "
        "denser than Venus (5.24 g/cm3), so 'densest terrestrial planet'->Venus premise is "
        "wrong; entity audit unaffected"),
}

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm(s):
    import re
    return re.sub(r"\s+", " ", (str(s) if s is not None else "").replace("_"," ")).strip().casefold()

def main():
    gold  = load_jsonl(GOLD)
    cands = {r["urbench_qid"]: r for r in load_jsonl(CANDS)}
    assert len(gold) == 30, f"gold has {len(gold)} rows"
    gmap = {r["urbench_qid"]: r for r in gold}

    # backup FIRST
    if not BACKUP.exists():
        shutil.copy2(GOLD, BACKUP)
    else:
        print(f"backup already exists: {BACKUP.name} (not overwritten)")

    errata, diff_lines = [], []
    add_n = rem_n = fix_n = 0

    for qid in [r["urbench_qid"] for r in gold]:
        row = gmap[qid]
        before = [dict(e) for e in row.get("question_entities", [])]
        actions = VERDICTS.get(qid, [])
        titles = {norm(e["canonical_title"]): e for e in row["question_entities"]}

        for act in actions:
            if act[0] == "ADD":
                _, title, span = act
                if norm(title) in titles:
                    sys.exit(f"ERROR {qid}: ADD '{title}' already present")
                titles[norm(title)] = {"canonical_title": title, "urdu_span": span}
                add_n += 1
            elif act[0] == "REMOVE":
                _, title = act
                if norm(title) not in titles:
                    sys.exit(f"ERROR {qid}: REMOVE '{title}' not present")
                del titles[norm(title)]
                rem_n += 1
            elif act[0] == "FIXSPAN":
                _, title, span = act
                if norm(title) not in titles:
                    sys.exit(f"ERROR {qid}: FIXSPAN '{title}' not present")
                titles[norm(title)]["urdu_span"] = span
                fix_n += 1

        row["question_entities"] = list(titles.values())
        if qid in NOTES:
            row["note"] = (row.get("note","") + " | " if row.get("note") else "") + NOTES[qid]

        after = row["question_entities"]
        if actions or qid in NOTES:
            errata.append({"qid": qid,
                           "before": [e["canonical_title"] for e in before],
                           "actions": [list(a) for a in actions],
                           "after": [e["canonical_title"] for e in after]})
            diff_lines.append(f"\n{qid}")
            diff_lines.append(f"  before: {[e['canonical_title'] for e in before]}")
            diff_lines.append(f"  action: {actions}")
            diff_lines.append(f"  after : {[e['canonical_title'] for e in after]}")

    # ---- validation ----
    qids = [r["urbench_qid"] for r in gold]
    assert len(qids) == len(set(qids)) == 30, "not 30 unique qids"
    span_errors = []
    for r in gold:
        ur = norm(cands[r["urbench_qid"]]["question_ur"])
        for e in r["question_entities"]:
            sp = norm(e.get("urdu_span"))
            if not sp:
                span_errors.append((r["urbench_qid"][:8], e["canonical_title"], "EMPTY span"))
            elif sp not in ur:
                span_errors.append((r["urbench_qid"][:8], e["canonical_title"],
                                    f"NOT verbatim: {e.get('urdu_span')!r}"))

    if span_errors:
        print("!! SPAN VALIDATION FAILED — gold NOT written:")
        for e in span_errors: print("   ", e)
        print("\nFix the span(s) in VERDICTS and rerun. Original gold untouched (only backup made).")
        sys.exit(1)

    # write gold + errata (only if validation passed)
    with open(GOLD, "w", encoding="utf-8") as f:
        for r in gold: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ERRATA, "w", encoding="utf-8") as f:
        for e in errata: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    # validate JSONL re-reads
    _ = load_jsonl(GOLD)

    print("BLIND30 GOLD ERRATA APPLIED")
    print("=" * 50)
    print(f"backup      : {BACKUP.name}")
    print(f"errata log  : {ERRATA.name}")
    print(f"ADD={add_n}  REMOVE={rem_n}  FIXSPAN={fix_n}")
    print(f"rows changed: {len(errata)}")
    print(f"validation  : 30 unique qids OK | JSONL valid | all spans verbatim OK")
    print("\nDIFF:")
    print("\n".join(diff_lines))

if __name__ == "__main__":
    main()