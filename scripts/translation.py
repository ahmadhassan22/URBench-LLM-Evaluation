import os
import json
import time
import argparse
from typing import List
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI

class URBenchItem(BaseModel):
    qid: str
    term: str
    description: str
    question: str
    answer: bool
    facts: List[str] = Field(default_factory=list)
    decomposition: List[str] = Field(default_factory=list)

def load_json_array(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError("Expected the input JSON to be a top-level array (list of items).")

def to_jsonl_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=r"C:\Users\Administrator\Documents\URBench\data\strategyqa_raw\strategyQA_train.json",
        help="Input .json (array of items)",
    )
    ap.add_argument(
        "--output",
        default=r"C:\Users\Administrator\Documents\URBench\data\strategyqa_raw\strategyQA_train_ur.jsonl",
        help="Output .jsonl (one item per line)",
    )
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--resume", default=True,action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=900)
    args = ap.parse_args()

    # Requires OPENAI_API_KEY to be set in env
    client = OpenAI()

    items = load_json_array(args.input)

    done_qids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        qid = obj.get("qid")
                        if qid:
                            done_qids.add(qid)
                except Exception:
                    pass

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as out_f:
        for item in tqdm(items, desc="Translating to Urdu"):

            # ---- HARD GUARD (fixes list.get crash) ----
            if not isinstance(item, dict):
                continue

            qid = item["qid"]
            if args.resume and qid in done_qids:
                continue

            instructions = (
                "Translate values from English to Urdu.\n"
                "Rules:\n"
                "- Do NOT change keys or structure.\n"
                "- Keep id and answer unchanged.\n"
                "- Translate term, question and facts with same meaning and similar length.\n"
                "- Proper nouns (people, places, books, religions, titles) MUST be transliterated or written in their established Urdu names, not literally translated.\n"
                "- answer must be exactly ہاں or نہیں.\n"
                "- facts must be a single-string array, merged in original order.\n"
                "- Use pure Urdu only (no English or Roman words).\n"
                "- Output valid JSON only, exact same schema.\n"
            )

            for attempt in range(6):
                try:
                    resp = client.responses.create(
                        model=args.model,
                        instructions=instructions,
                        input=json.dumps(item, ensure_ascii=False),
                        temperature=0,
                        max_output_tokens=args.max_output_tokens,
                    )

                    out_obj = json.loads(resp.output_text)

                    # extra guard (unchanged intent)
                    out_obj["qid"] = item["qid"]
                    out_obj["answer"] = item["answer"]

                    out_f.write(to_jsonl_line(out_obj))
                    out_f.flush()
                    break

                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    if attempt == 5:
                        fail = dict(item)
                        fail["_urbench_error"] = str(e)[:500]
                        out_f.write(to_jsonl_line(fail))
                        out_f.flush()
                        break
                    time.sleep(wait)

            if args.sleep > 0:
                time.sleep(args.sleep)

if __name__ == "__main__":
    main()
