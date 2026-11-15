# D14 preparation: inventory for the next experiment (read-only)

Written 2026-11-15. This file is an inventory only. Nothing was run on a model, no file other than this document was written, and nothing is decided here. Every count below is in the quoted output of a command; the command is shown above its output. The inventory script is not committed (it lives under the untracked `logs/`), so its full source is quoted in the appendix.

Scope: CSQA, PIQA, BoolQ and GSM8K in the fair regime (README, *Fair-Regime Protocol*), the StrategyQA rows not yet used by any set, and the model weights on disk.

## 1. Evaluation files: path, rows, MD5, Urdu fields, English original

```
$ python3 -I logs/unattended/task5_inventory.py data
== csqa
eval file: data/sdfr_splits/csqa_eval.jsonl  rows 300  MD5 c10cefaf28e1816c0b7d1d22ddbc2225
fields: ['qid', 'id', 'question', 'question_concept', 'choices', 'answerKey']
  field qid               rows with Arabic-script letters 0, rows with Latin letters 300, rows with neither 0
  field id                rows with Arabic-script letters 0, rows with Latin letters 300, rows with neither 0
  field question          rows with Arabic-script letters 300, rows with Latin letters 0, rows with neither 0
  field question_concept  rows with Arabic-script letters 300, rows with Latin letters 0, rows with neither 0
  field choices           rows with Arabic-script letters 300, rows with Latin letters 300, rows with neither 0
  field answerKey         rows with Arabic-script letters 0, rows with Latin letters 300, rows with neither 0
Urdu source file: data/csqa_raw/csqa_train_1500_ur.jsonl  rows 1500  MD5 396601c0f6c1b26aaa4c29c3fc6cf018
English source file: data/csqa_raw/csqa_train_1500_en.jsonl  rows 1500  MD5 4a47619e60c4be2478757b8c8e6581bc
eval rows found in the Urdu source file by question: 300 of 300; distinct source positions 300; positions 1200 to 1499
Urdu and English source files, same 'id' at the same position: 1499 of 1500
fields of the eval file that are Latin-only on every row: ['qid', 'id', 'answerKey']

== piqa
eval file: data/sdfr_splits/piqa_eval.jsonl  rows 150  MD5 416fb9dabf496f2416b39985fbf7fdf5
fields: ['goal', 'sol1', 'sol2', 'label']
  field goal              rows with Arabic-script letters 150, rows with Latin letters 0, rows with neither 0
  field sol1              rows with Arabic-script letters 150, rows with Latin letters 0, rows with neither 0
  field sol2              rows with Arabic-script letters 150, rows with Latin letters 0, rows with neither 0
  field label             rows with Arabic-script letters 0, rows with Latin letters 0, rows with neither 150
Urdu source file: data/piqa_raw/piqa_train_750_ur_fixed.jsonl  rows 750  MD5 01586d82da54b781aefb1d5427389515
English source file: data/piqa_raw/piqa_train_750.jsonl  rows 750  MD5 9176d3b16cddebde5401ffe54dec49c2
eval rows found in the Urdu source file by goal+sol1+sol2: 150 of 150; distinct source positions 149; positions 600 to 749
Urdu and English source files have no id field; they have the same row count: True
fields of the eval file that are Latin-only on every row: []

== boolq
eval file: data/sdfr_splits/boolq_eval.jsonl  rows 310  MD5 45fb7cf674474b4e722097c3414d4910
fields: ['qid', 'question', 'passage', 'answer']
  field qid               rows with Arabic-script letters 0, rows with Latin letters 310, rows with neither 0
  field question          rows with Arabic-script letters 310, rows with Latin letters 0, rows with neither 0
  field passage           rows with Arabic-script letters 310, rows with Latin letters 0, rows with neither 0
  field answer            rows with Arabic-script letters 310, rows with Latin letters 0, rows with neither 0
Urdu source file: data/boolq_raw/boolq_train_1550_ur_fixed.jsonl  rows 1550  MD5 5da4e4276bdcd6af950696c32b6b314c
English source file: data/boolq_raw/boolq_train_1550_en.jsonl  rows 1550  MD5 12b4afce9c3a208b02706513f7631d8f
eval rows found in the Urdu source file by question: 310 of 310; distinct source positions 310; positions 1240 to 1549
Urdu and English source files, same 'qid' at the same position: 1550 of 1550
fields of the eval file that are Latin-only on every row: ['qid']

== gsm8k
eval file: data/sdfr_splits/gsm8k_eval.jsonl  rows 700  MD5 cb701f18965757d563391a9cda2242aa
fields: ['question', 'answer']
  field question          rows with Arabic-script letters 699, rows with Latin letters 12, rows with neither 0
  field answer            rows with Arabic-script letters 0, rows with Latin letters 699, rows with neither 1
Urdu source file: data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl  rows 700  MD5 ffb57c3df417799ad1d03f79a5aafc12
English source file: data/gsm8k_raw/gsm8k_main_train_700.jsonl  rows 700  MD5 651a81016197dbaba7bc4b8a64d55c82
eval rows found in the Urdu source file by question: 700 of 700; distinct source positions 700; positions 0 to 699
Urdu and English source files have no id field; they have the same row count: True
fields of the eval file that are Latin-only on every row: []

```

Reading notes (no new counts):
- The four evaluation files are the files under `data/sdfr_splits/`. Their text fields are Urdu. No row carries the original English text; the English original of a row is the row at the same position of the English source file under `data/<set>_raw/` (the position is given by the match against the Urdu source file above; ids agree where an id field exists, with the one CSQA exception shown).
- CSQA, PIQA and BoolQ evaluation rows are the tail of their Urdu source file; GSM8K uses the whole source file.
- PIQA: the number of distinct source positions is one less than the number of evaluation rows, so one evaluation row occurs twice.
- GSM8K: one evaluation question has no Arabic-script letter.

## 2. Baselines in the fair regime: script, prompt, scorer, max tokens, sealed output, accuracy recomputed

```
$ python3 -I logs/unattended/task5_inventory.py baselines
== csqa
script: eval/error_analysis_tests/cot_csqa_baseline_fair.py  MD5 70491653824525aeaa58a3282d5d177f
  line 6: MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
  line 13: def extract_answer(text):
  line 23: llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
  line 25: sp = SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|im_end|>"])
  line 41: add_generation_prompt=True, enable_thinking=True))
sealed output: outputs/sdfr/cot_csqa_baseline_fair_qwen3_14b.jsonl  rows 300  MD5 40e09c2208cce4217f86fb54c1cc034c
accuracy from the stored 'correct' field: 190 / 300 = 63.33%
accuracy recomputed (the script's extract_answer on the text after </think>, against the stored gold): 190 / 300 = 63.33%; recomputed pred equals stored pred on 300 rows
rows without </think> in the generation: 20; rows with an empty pred: 3; output rows 300 = eval rows 300: True

== piqa
script: eval/error_analysis_tests/cot_piqa_baseline_fair.py  MD5 cd6f074c4ea96820226b48768767ad37
  line 6: MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
  line 13: def extract_answer(text):
  line 23: llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
  line 25: sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])
  line 39: add_generation_prompt=True, enable_thinking=True))
sealed output: outputs/sdfr/cot_piqa_baseline_fair_qwen3_14b.jsonl  rows 150  MD5 d969451aa8563c4524e89aef959553d5
accuracy from the stored 'correct' field: 108 / 150 = 72.00%
accuracy recomputed (the script's extract_answer on the text after </think>, against the stored gold): 108 / 150 = 72.00%; recomputed pred equals stored pred on 150 rows
rows without </think> in the generation: 1; rows with an empty pred: 0; output rows 150 = eval rows 150: True

== boolq
no file named cot_boolq_baseline_fair.py: True
  outputs/boolq/qwen3_14b/boolq_cot_qwen3_14b.jsonl  rows 1550  MD5 b40dc06955f8c801ca23bd1110a93b68  fields ['qid', 'question', 'passage', 'gold_answer', 'pred_answer', 'correct', 'prompt', 'raw_output', 'model_name']  stored correct 1300 (83.87%)
  outputs/boolq/qwen3_14b/boolq_three_shot_qwen3_14b.jsonl  rows 1550  MD5 4caf48af97ee60a57b77c5cb39ad0948  fields ['qid', 'question', 'passage', 'gold_answer', 'pred_answer', 'correct', 'prompt', 'raw_output', 'model_name']  stored correct 1315 (84.84%)
  outputs/boolq/qwen3_14b/boolq_zero_shot_qwen3_14b.jsonl  rows 1550  MD5 90adc07cce562b8e27ab4ae6283f0c28  fields ['qid', 'question', 'passage', 'gold_answer', 'pred_answer', 'correct', 'prompt', 'raw_output', 'model_name']  stored correct 1309 (84.45%)
  outputs/boolq/qwen3_14b/boolq_three_shot_qwen3_14b.jsonl: rows 1550, correct 1315 (84.84%); rows whose qid is in the 310-row eval file: 310, correct 263 (84.84%)
  outputs/sdfr/sdfr_boolq_large_passage_qwen3_14b.jsonl: rows 310, correct 265 (85.48%); rows whose qid is in the 310-row eval file: 310, correct 265 (85.48%)
  outputs/sdfr/sdfr_boolq_qwen3_14b.jsonl: rows 310, correct 262 (84.52%); rows whose qid is in the 310-row eval file: 310, correct 262 (84.52%)

== gsm8k
script: eval/error_analysis_tests/cot_gsm8k_baseline_fair.py  MD5 0ac419f37b59b5695434736f9abe9a4a
  line 6: MODEL_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
  line 13: def extract_gold(answer_str):
  line 18: def extract_answer(text):
  line 30: llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
  line 32: sp = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|im_end|>"])
  line 44: add_generation_prompt=True, enable_thinking=True))
sealed output: outputs/sdfr/cot_gsm8k_baseline_fair_qwen3_14b.jsonl  rows 700  MD5 d94ea3127817cf7357ca085f7a06e273
accuracy from the stored 'correct' field: 621 / 700 = 88.71%
accuracy recomputed (the script's extract_answer on the text after </think>, against the stored gold): 621 / 700 = 88.71%; recomputed pred equals stored pred on 700 rows
rows without </think> in the generation: 37; rows with an empty pred: 16; output rows 700 = eval rows 700: True

```

### CSQA: scorer and prompt text, as in the script

```
$ sed -n '13,19p' eval/error_analysis_tests/cot_csqa_baseline_fair.py
def extract_answer(text):
    m = re.search(r"[Aa]nswer\s*[:：]\s*([A-E])", text)
    if m: return m.group(1).upper()
    m2 = re.findall(r"\b([A-E])\b", text.upper())
    return m2[-1] if m2 else ""

if __name__ == "__main__":
```

```
$ sed -n '31,38p' eval/error_analysis_tests/cot_csqa_baseline_fair.py
        instr = ("آپ کو ایک سوال اور پانچ اختیارات (A، B، C، D، E) دیے گئے ہیں۔\n"
                 "پہلے مختصر طور پر اردو میں سوچیں کہ کون سا جواب درست ہے۔\n"
                 "آخر میں صرف یہ شکل استعمال کریں: Answer: X\n"
                 "جہاں X صرف A، B، C، D یا E ہو۔")
        q_block = (f"سوال: {item['question']}\n"
                   f"A: {A}\nB: {B}\nC: {C}\nD: {D}\nE: {E}\n\n"
                   f"Step-by-step reasoning (اردو میں):\n\nAnswer:")
        raw = f"{instr}\n\n{q_block}"
```

### PIQA: scorer and prompt text, as in the script

```
$ sed -n '13,19p' eval/error_analysis_tests/cot_piqa_baseline_fair.py
def extract_answer(text):
    m = re.search(r"حتمی\s*جواب\s*[:：]\s*([01])", text)
    if m: return m.group(1)
    m2 = re.findall(r"\b([01])\b", text)
    return m2[-1] if m2 else ""

if __name__ == "__main__":
```

```
$ sed -n '29,36p' eval/error_analysis_tests/cot_piqa_baseline_fair.py
        instr = ('مقصد اور دو ممکنہ حل دیے گئے ہیں۔\n'
                 'پہلے مرحلہ وار اور تفصیل سے سوچیں کہ کون سا حل زیادہ منطقی ہے اور عام فہم کے مطابق ہے۔\n'
                 'پھر آخر میں صرف ایک ہندسہ لکھیں: 0 یا 1۔\n'
                 '0 کا مطلب ہے حل 1 درست ہے، 1 کا مطلب ہے حل 2 درست ہے۔')
        q_block = (f"مقصد: {item['goal']}\n"
                   f"حل 1: {item['sol1']}\nحل 2: {item['sol2']}\n\n"
                   f"سوچنے کے مراحل:\nحتمی جواب:")
        raw = f"{instr}\n\n{q_block}"
```

### GSM8K: scorer and prompt text, as in the script

```
$ sed -n '13,26p' eval/error_analysis_tests/cot_gsm8k_baseline_fair.py
def extract_gold(answer_str):
    m = re.search(r"####\s*(-?[\d,]+)", answer_str)
    if m: return m.group(1).replace(",", "").strip()
    return answer_str.strip()

def extract_answer(text):
    m = re.search(r"حتمی\s*جواب\s*[:：]\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "").strip()
    m2 = re.search(r"####\s*(-?[\d,]+)", text)
    if m2: return m2.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""

if __name__ == "__main__":
```

```
$ sed -n '36,41p' eval/error_analysis_tests/cot_gsm8k_baseline_fair.py
        instr = ('آپ کو ایک ریاضی کا لفظی مسئلہ دیا گیا ہے۔\n'
                 'پہلے مرحلہ وار سوچیں اور مسئلہ حل کریں۔\n'
                 'اس کے بعد "حتمی جواب:" کے بعد صرف ایک عددی جواب لکھیں۔\n'
                 'حتمی جواب میں کوئی یونٹ، الفاظ یا اضافی متن شامل نہ کریں۔')
        q_block = f"مسئلہ: {item['question']}\n\nسوچنے کے مراحل:\nحتمی جواب:"
        raw = f"{instr}\n\n{q_block}"
```

### BoolQ: no dedicated fair baseline

There is no `cot_boolq_baseline_fair.py` and no baseline output on the evaluation split alone. The README compares the SDFR run on the evaluation split with the three-shot run on the whole Urdu source file (both with thinking off). The blocks below show the three-shot script as it stands, the prompt files on disk, and the SDFR script used for the method run.

```
$ sed -n '9,21p' eval/eval_boolq_three_shot_vllm.py
BOOLQ_PATH = "../data/boolq_raw/boolq_train_1550_ur_fixed.jsonl"
PROMPT_PATH = "../prompts/boolq/three_shot.txt"

MODEL_NAME = "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
OUTPUT_DIR = "/mnt/home/user41/URBench/outputs/boolq/llama3.1_70b_awq"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = "boolq_three_shot_llama3.1_70b_awq.jsonl"

MAX_EXAMPLES = None

MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 8
```

```
$ sed -n '44,64p' eval/eval_boolq_three_shot_vllm.py
def normalize_boolq_output(text):
    if text is None:
        return None

    t = text.strip().replace('"', "").replace("'", "").strip()

    if "ہاں" in t:
        return "ہاں"
    if "نہیں" in t or "نھیں" in t or "نہيں" in t:
        return "نہیں"

    low = t.lower()
    if "yes" in low:
        return "ہاں"
    if "no" in low:
        return "نہیں"

    return None


def build_prompt(template: str, passage: str, question: str) -> str:
```

```
$ ls prompts/boolq; md5sum prompts/boolq/three_shot_p1.txt; ls prompts/boolq/three_shot.txt
cot_p1.txt
cot_p2.txt
cot_p3.txt
three_shot_p1.txt
three_shot_p2.txt
three_shot_p3.txt
zero_shot_p1.txt
zero_shot_p2.txt
zero_shot_p3.txt
1f08b11a1b8d475ee9ec87b7ea9e7c20  prompts/boolq/three_shot_p1.txt
ls: cannot access 'prompts/boolq/three_shot.txt': No such file or directory
```

```
$ cat prompts/boolq/three_shot_p1.txt
مثال 1:
عبارت: {"یہ معقول معلوم ہوتا ہے کہ مچھلیوں کے جھنڈ میں باقاعدہ فاصلہ اور سائز کی یکسانیت ہائیڈروڈائنامک کارکردگی کا باعث بنتی ہے۔ تاہم، تجربات میں یہ ثابت نہیں ہوا کہ جھنڈ میں موجود مچھلی کے پڑوسیوں کی وجہ سے ہائیڈروڈائنامک لفٹ سے کوئی فائدہ ہوتا ہے، اگرچہ اب بھی خیال کیا جاتا ہے کہ جنگل میں کارکردگی میں بہتری آتی ہے۔ لینڈا (1998) کا کہنا ہے کہ جھنڈ کا رہنما مسلسل بدلتا رہتا ہے، کیونکہ اگرچہ جھنڈ کے اندر رہنے سے ہائیڈروڈائنامک فائدہ ملتا ہے، لیکن رہنما ہونے کا مطلب ہے کہ آپ سب سے پہلے خوراک تک پہنچتے ہیں۔"}
سوال: {"کیا مچھلیوں کے جھنڈ میں کوئی رہنما ہوتا ہے"}
جواب: {"ہاں"}

مثال 2:
عبارت: {"ریاست نہ تو کسی شخص کو ہتھیار لے جانے کے لیے اجازت نامہ جاری کرتی ہے اور نہ ہی اس کی ضرورت ہوتی ہے، چاہے وہ کھلے عام ہو یا چھپ کر۔ امریکہ میں اسے آئینی اجازت نامہ کہا جاتا ہے، کیونکہ یہاں کا 'اجازت نامہ' ریاستہائے متحدہ امریکہ کا آئین ہے۔ ورمونٹ واحد ریاست ہے جہاں ہمیشہ سے یہی صورت حال رہی ہے (اسی لیے اسے ورمونٹ کیری بھی کہا جاتا ہے)۔ ورمونٹ کے قانون میں ریاست کے رہائشیوں اور غیر رہائشیوں کے درمیان کوئی فرق نہیں کیا گیا؛ دونوں کو ورمونٹ میں بغیر اجازت نامہ کے ہتھیار لے جانے کا یکساں حق حاصل ہے۔"}
سوال: {"کیا ورمونٹ میں بندوق رکھنے کے لیے اجازت نامہ درکار ہے"}
جواب: {"نہیں"}

مثال 3:
عبارت: {"پیلس، جو اپنی بندش کے وقت 29 سال پرانا تھا، اب بھی کھیلوں اور کنسرٹ کے مقام کے طور پر بہترین حالت میں ہے۔ تاہم، یہ ایک شمالی مضافاتی علاقے میں واقع ہے، جو شہر کے مرکز سے نسبتاً دور ہے، خاص طور پر اس بڑھتے ہوئے رجحان کے پیش نظر جسے 'واک ایبل اربن ازم' کہا جاتا ہے، جہاں پِسٹنز اپنے مداحوں کی تعداد بڑھانا چاہتے ہیں۔ اگرچہ ابھی تک کوئی سرکاری تصدیق نہیں ہوئی، لیکن توقع ہے کہ پیلس کو منہدم کر دیا جائے گا، جہاں اسے نئے آٹو سپلائر ہیڈکوارٹرز اور تحقیق و ترقی کے پارکس کے لیے دوبارہ تیار کیا جا سکتا ہے۔"}
سوال: {"کیا پیلس آف آبرن ہلز اب بھی قائم ہے"}
جواب: {"ہاں"}

اب درج ذیل سوال کا جواب دیں۔
صرف ایک لفظ میں: ہاں یا نہیں۔
کوئی وضاحت شامل نہ کریں۔

عبارت: {passage}
سوال: {question}
جواب:
```

```
$ grep -n 'max_tokens\|enable_thinking\|TOP_K *=\|POOL\|INDEX' eval/sdfr_boolq_large_passage.py
12:INDEXES     = f"{BASE}/sdfr_indexes"
17:TOP_K       = 3
53:    index = faiss.read_index(f"{INDEXES}/boolq_large_passage_faiss.index")
63:    sampling_params = SamplingParams(temperature=0.0, max_tokens=16,
86:            enable_thinking=False
```

```
$ sed -n '23,31p' eval/sdfr_boolq_large_passage.py
def normalize_answer(text):
    t = text.strip().lower()
    if t in ["true", "yes", "ہاں", "ہاں۔"]: return "ہاں"
    if t in ["false", "no", "نہیں", "نہیں۔"]: return "نہیں"
    if "ہاں" in text: return "ہاں"
    if "نہیں" in text: return "نہیں"
    return text.strip()

def format_example(item):
```

Reading notes (no new counts):
- CSQA, PIQA and GSM8K baselines: Qwen3-14B through vLLM, thinking on, the answer is extracted from the text after `</think>`; token limits as on the quoted lines. The stored and the recomputed accuracy agree for all three.
- BoolQ: the three-shot script currently names the 70B model and a prompt path (`../prompts/boolq/three_shot.txt`) that does not exist on disk; the prompt files are `three_shot_p1.txt` to `three_shot_p3.txt`. The stored `correct` field of the Qwen3-14B three-shot output gives the percentage shown in the block, which is not the figure printed in the README for that run (see QUESTIONS in the progress file of 2026-11-15).

## 3. Retrieval / demo pools and their overlap with the evaluation files

```
$ python3 -I logs/unattended/task5_inventory.py pools
== csqa
pool data/sdfr_splits/csqa_pool.jsonl  rows 1200  MD5 88dea82c755e71849976edad3fcc1063
  eval rows whose English source text (question) is in the pool: 0 of 300; eval rows whose Urdu text is in the pool: 0; eval 'qid' values in the pool: 0; eval 'id' values in the pool: 0
  pool rows with Arabic-script letters in question: 0
pool data/sdfr_splits/csqa_pool_large_clean.jsonl  rows 9441  MD5 3e393d1d913fa45ed6f6fbdeef33203f
  eval rows whose English source text (question) is in the pool: 0 of 300; eval rows whose Urdu text is in the pool: 0; eval 'id' values in the pool: 0
  pool rows with Arabic-script letters in question: 0

== piqa
pool data/sdfr_splits/piqa_pool.jsonl  rows 600  MD5 c80e2194d038fab1e1d1a521e2295d7a
  eval rows whose English source text (goal+sol1+sol2) is in the pool: 0 of 150; eval rows whose Urdu text is in the pool: 0
  pool rows with Arabic-script letters in goal: 0

== boolq
pool data/sdfr_splits/boolq_pool.jsonl  rows 1240  MD5 eca31a80d4cb42bb3f03feb6b8a3d2c2
  eval rows whose English source text (question) is in the pool: 0 of 310; eval rows whose Urdu text is in the pool: 0; eval 'qid' values in the pool: 0
  pool rows with Arabic-script letters in question: 0
pool data/sdfr_splits/boolq_pool_large_clean.jsonl  rows 7877  MD5 cb86a02397c5b02bd8e437dfe4376a8b
  eval rows whose English source text (question) is in the pool: 0 of 310; eval rows whose Urdu text is in the pool: 0
  pool rows with Arabic-script letters in question: 0

== gsm8k
pool data/sdfr_splits/gsm8k_pool.jsonl  rows 7473  MD5 577e9ccd23c5a0bd889360ebee47b097
  eval rows whose English source text (question) is in the pool: 700 of 700; eval rows whose Urdu text is in the pool: 1
  pool rows with Arabic-script letters in question: 0
  gsm8k: eval rows whose 'answer' string is in the pool answers: 700 of 700
  g1_sdfr_gsm8k_clean.py: TOP_K        = 3        # unchanged from sdfr_gsm8k_fair.py
  g1_sdfr_gsm8k_clean.py: POOL_EXPECT  = 7473
  g1_sdfr_gsm8k_clean.py: POOL_AFTER_EXPECT = 6773   # freeze C.2

```

Reading notes (no new counts):
- CSQA, PIQA and BoolQ: no evaluation row is in a pool, by English source text, by Urdu text or by id.
- GSM8K: every evaluation row's English source question is in the pool (the leakage recorded as EXPERIMENT G1). No decontaminated pool file exists on disk: `g1_sdfr_gsm8k_clean.py` drops the leaked rows at retrieval time and expects the pool sizes on the quoted lines.

## 4. StrategyQA: rows not used by any set

```
$ python3 -I logs/unattended/task5_inventory.py strategyqa
mapped file rows 2290, distinct urbench_qid 2290, MD5 bb982798279f5d519ddef01f5cb7ba8c
  eval458 (is_eval in the mapped file)                  458 qids,  458 of them in the mapped file
  eval458 (data/sdfr_splits/strategyqa_eval.jsonl)      458 qids,  458 of them in the mapped file
  DEV200 (dev200_seed4242_qids.txt)                     200 qids,  200 of them in the mapped file
  RDEV300 (efbpt/rdev300_seed2026.jsonl)                300 qids,  300 of them in the mapped file
  DEV50 (dev50_seed42_qids.txt)                          50 qids,   50 of them in the mapped file
  train manifest 100 (efbpt/plan_a_qids_100.txt)        100 qids,  100 of them in the mapped file
  train manifest 250 (efbpt/plan_a_qids_250.txt)        250 qids,  250 of them in the mapped file
  train manifest 500 (efbpt/plan_a_qids_500.txt)        500 qids,  500 of them in the mapped file
  AUDIT30 (efbpt/audit30_qids.txt)                       30 qids,   30 of them in the mapped file
  BLIND30 (efbpt/blind30_qids.txt)                       30 qids,   30 of them in the mapped file
the two eval458 lists are the same set: True
pairwise overlaps (qids in both):
  eval458 (data/sdfr_splits/strategyqa_eval.jsonl) x DEV200 (dev200_seed4242_qids.txt)                0
  eval458 (data/sdfr_splits/strategyqa_eval.jsonl) x RDEV300 (efbpt/rdev300_seed2026.jsonl)           0
  eval458 (data/sdfr_splits/strategyqa_eval.jsonl) x DEV50 (dev50_seed42_qids.txt)                    0
  eval458 (data/sdfr_splits/strategyqa_eval.jsonl) x train manifest 100 (efbpt/plan_a_qids_100.txt)   0
  eval458 (data/sdfr_splits/strategyqa_eval.jsonl) x train manifest 250 (efbpt/plan_a_qids_250.txt)   0
  eval458 (data/sdfr_splits/strategyqa_eval.jsonl) x train manifest 500 (efbpt/plan_a_qids_500.txt)   0
  DEV200 (dev200_seed4242_qids.txt)                x RDEV300 (efbpt/rdev300_seed2026.jsonl)           0
  DEV200 (dev200_seed4242_qids.txt)                x DEV50 (dev50_seed42_qids.txt)                    50
  DEV200 (dev200_seed4242_qids.txt)                x train manifest 100 (efbpt/plan_a_qids_100.txt)   0
  DEV200 (dev200_seed4242_qids.txt)                x train manifest 250 (efbpt/plan_a_qids_250.txt)   0
  DEV200 (dev200_seed4242_qids.txt)                x train manifest 500 (efbpt/plan_a_qids_500.txt)   0
  RDEV300 (efbpt/rdev300_seed2026.jsonl)           x DEV50 (dev50_seed42_qids.txt)                    0
  RDEV300 (efbpt/rdev300_seed2026.jsonl)           x train manifest 100 (efbpt/plan_a_qids_100.txt)   24
  RDEV300 (efbpt/rdev300_seed2026.jsonl)           x train manifest 250 (efbpt/plan_a_qids_250.txt)   45
  RDEV300 (efbpt/rdev300_seed2026.jsonl)           x train manifest 500 (efbpt/plan_a_qids_500.txt)   84
  DEV50 (dev50_seed42_qids.txt)                    x train manifest 100 (efbpt/plan_a_qids_100.txt)   0
  DEV50 (dev50_seed42_qids.txt)                    x train manifest 250 (efbpt/plan_a_qids_250.txt)   0
  DEV50 (dev50_seed42_qids.txt)                    x train manifest 500 (efbpt/plan_a_qids_500.txt)   0
  train manifest 100 (efbpt/plan_a_qids_100.txt)   x train manifest 250 (efbpt/plan_a_qids_250.txt)   100
  train manifest 100 (efbpt/plan_a_qids_100.txt)   x train manifest 500 (efbpt/plan_a_qids_500.txt)   100
  train manifest 250 (efbpt/plan_a_qids_250.txt)   x train manifest 500 (efbpt/plan_a_qids_500.txt)   250
union of eval458, DEV200, RDEV300, DEV50 and the three train manifests: 1374 qids
rows of the 2290 in none of them: 916
  of these, answer true 430 / false 486
also outside AUDIT30 and BLIND30: 867
stage1_report.jsonl rows 1782, fields ['qid', 'status', 'reason', 'n_steps']
data/sdfr_splits/strategyqa_pool.jsonl (the SDFR demo pool): 1832 qids; of the 916 free rows, 916 are in it
```

Reading notes (no new counts):
- DEV50 lies inside DEV200. RDEV300 shares rows with the training manifests. The three training manifests are nested.
- Every free row is in the SDFR demo pool file `data/sdfr_splits/strategyqa_pool.jsonl`, so those rows have been shown to a model as demonstrations in the SDFR runs, but have not been used as evaluation rows of any set listed here.

## 5. Model weights on disk

```
$ python3 -I logs/unattended/task5_inventory.py models
AI-ModelScope/gemma-2-27b-it                                              51G  shards 12 of 12 present
AI-ModelScope/gemma-2-9b-it                                               18G  shards 4 of 4 present
Alif/Alif-1.0-8B-Instruct                                                6.2G  no safetensors index; weight files 1
Alif/Alif-1.0-8B-Merged                                                   15G  shards 4 of 4 present
BAAI/bge-reranker-v2-m3                                                  2.2G  no safetensors index; weight files 1
LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4                         38G  shards 9 of 9 present -> /mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4
LLM-Research/Meta-Llama-3.1-8B-Instruct                                   30G  shards 4 of 4 present -> /mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-8B-Instruct
LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4                       38G  shards 9 of 9 present
LLM-Research/Meta-Llama-3___1-8B-Instruct                                 30G  shards 4 of 4 present
Qalb/Qalb-1.0-8B-Instruct                                                 15G  shards 4 of 4 present
Qwen/Qwen2.5-1.5B-Instruct                                               2.9G  no safetensors index; weight files 1 -> /mnt/home/user41/downloaded_models/Qwen/Qwen2___5-1___5B-Instruct
Qwen/Qwen2.5-14B-Instruct-AWQ                                            9.4G  shards 3 of 3 present
Qwen/Qwen2.5-7B-Instruct                                                  15G  shards 4 of 4 present -> /mnt/home/user41/downloaded_models/Qwen/Qwen2___5-7B-Instruct
Qwen/Qwen2___5-1___5B-Instruct                                           2.9G  no safetensors index; weight files 1
Qwen/Qwen2___5-7B-Instruct                                                15G  shards 4 of 4 present
Qwen/Qwen3-14B                                                            28G  shards 8 of 8 present
Qwen/Qwen3-8B                                                             16G  shards 5 of 5 present
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B                                   15G  shards 2 of 2 present
nllb-200-3.3B                                                             17G  no safetensors index; weight files 3
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2              4.4G  no safetensors index; weight files 2
directories without config.json: ['AI-ModelScope/Llama-3___1-70B-Instruct-AWQ', 'AI-ModelScope/Llama-3___3-70B-Instruct-AWQ', 'AI-ModelScope/Meta-Llama-3___1-70B-Instruct-AWQ', 'LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', 'LLM-Research/Meta-Llama-3___3-70B-Instruct-AWQ', 'large-traversaal', 'large-traversaal/Alif-1___0-8B-Instruct', 'models--large-traversaal--Alif-1.0-8B-Instruct', 'models--large-traversaal--Alif-1.0-8B-Instruct/refs', 'modelscope', 'modelscope/Meta-Llama-3___3-70B-Instruct-AWQ']
```

## Appendix: the inventory script

```python
#!/usr/bin/env python3
# Read-only inventory for docs/handoff/D14_PREP.md. Run from the repo root: python3 -I logs/unattended/task5_inventory.py <section>
# Sections: data, baselines, pools, strategyqa, models. It opens files for reading only and writes nothing.
import ast, glob, hashlib, json, os, re, subprocess, sys

S = "data/sdfr_splits/"
O = "outputs/sdfr/"
E = "eval/error_analysis_tests/"
DS = {
    "csqa": dict(ev=S + "csqa_eval.jsonl", ur="data/csqa_raw/csqa_train_1500_ur.jsonl", en="data/csqa_raw/csqa_train_1500_en.jsonl", key=["question"], idf="id",
                 pools=[S + "csqa_pool.jsonl", S + "csqa_pool_large_clean.jsonl"], script=E + "cot_csqa_baseline_fair.py", out=O + "cot_csqa_baseline_fair_qwen3_14b.jsonl"),
    "piqa": dict(ev=S + "piqa_eval.jsonl", ur="data/piqa_raw/piqa_train_750_ur_fixed.jsonl", en="data/piqa_raw/piqa_train_750.jsonl", key=["goal", "sol1", "sol2"], idf=None,
                 pools=[S + "piqa_pool.jsonl"], script=E + "cot_piqa_baseline_fair.py", out=O + "cot_piqa_baseline_fair_qwen3_14b.jsonl"),
    "boolq": dict(ev=S + "boolq_eval.jsonl", ur="data/boolq_raw/boolq_train_1550_ur_fixed.jsonl", en="data/boolq_raw/boolq_train_1550_en.jsonl", key=["question"], idf="qid",
                  pools=[S + "boolq_pool.jsonl", S + "boolq_pool_large_clean.jsonl"], script=None, out=None),
    "gsm8k": dict(ev=S + "gsm8k_eval.jsonl", ur="data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl", en="data/gsm8k_raw/gsm8k_main_train_700.jsonl", key=["question"], idf=None,
                  pools=[S + "gsm8k_pool.jsonl"], script=E + "cot_gsm8k_baseline_fair.py", out=O + "cot_gsm8k_baseline_fair_qwen3_14b.jsonl"),
}


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def arabic(s):
    return any("؀" <= ch <= "ۿ" for ch in s)


def latin(s):
    return any("a" <= ch.lower() <= "z" for ch in s)


def flat(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def norm(s):
    return " ".join(str(s).lower().split())


def kk(r, key):
    return tuple(norm(flat(r[k])) for k in key)


def to_raw_index(d):
    ev, ur = rd(d["ev"]), rd(d["ur"])
    pos = {}
    for i, r in enumerate(ur):
        pos.setdefault(kk(r, d["key"]), []).append(i)
    return ev, ur, [pos.get(kk(r, d["key"]), [None])[0] for r in ev]


def data():
    for name, d in DS.items():
        ev, ur, idx = to_raw_index(d)
        en = rd(d["en"])
        print("== %s" % name)
        print("eval file: %s  rows %d  MD5 %s" % (d["ev"], len(ev), md5(d["ev"])))
        print("fields: %s" % list(ev[0].keys()))
        for k in ev[0]:
            vals = [flat(r.get(k, "")) for r in ev]
            print("  field %-17s rows with Arabic-script letters %d, rows with Latin letters %d, rows with neither %d"
                  % (k, sum(arabic(v) for v in vals), sum(latin(v) for v in vals), sum((not arabic(v)) and (not latin(v)) for v in vals)))
        print("Urdu source file: %s  rows %d  MD5 %s" % (d["ur"], len(ur), md5(d["ur"])))
        print("English source file: %s  rows %d  MD5 %s" % (d["en"], len(en), md5(d["en"])))
        hit = [i for i in idx if i is not None]
        print("eval rows found in the Urdu source file by %s: %d of %d; distinct source positions %d; positions %s to %s"
              % ("+".join(d["key"]), len(hit), len(ev), len(set(hit)), min(hit) if hit else None, max(hit) if hit else None))
        if d["idf"]:
            same = sum(1 for a, b in zip(ur, en) if a.get(d["idf"]) == b.get(d["idf"]))
            print("Urdu and English source files, same '%s' at the same position: %d of %d" % (d["idf"], same, len(ur)))
        else:
            print("Urdu and English source files have no id field; they have the same row count: %s" % (len(ur) == len(en)))
        eng_in_eval = [k for k in ev[0] if sum(latin(flat(r.get(k, ""))) and not arabic(flat(r.get(k, ""))) for r in ev) == len(ev)]
        print("fields of the eval file that are Latin-only on every row: %s" % eng_in_eval)
        print()


def funcs(script, names):
    tree = ast.parse(open(script, encoding="utf-8").read())
    ns = {"re": re}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), script, "exec"), ns)
    return ns


def baselines():
    for name, d in DS.items():
        print("== %s" % name)
        if not d["script"]:
            print("no file named cot_boolq_baseline_fair.py: %s" % (not os.path.exists(E + "cot_boolq_baseline_fair.py")))
            for p in sorted(glob.glob("outputs/boolq/qwen3_14b/*.jsonl")):
                r = rd(p)
                acc = sum(1 for x in r if x.get("correct")) if "correct" in r[0] else None
                print("  %s  rows %d  MD5 %s  fields %s  stored correct %s" % (p, len(r), md5(p), list(r[0].keys())[:9], "%d (%.2f%%)" % (acc, 100.0 * acc / len(r)) if acc is not None else "n/a"))
            evq = {r["qid"] for r in rd(d["ev"])}
            for p in sorted(glob.glob("outputs/boolq/qwen3_14b/*three_shot*.jsonl")) + [O + "sdfr_boolq_large_passage_qwen3_14b.jsonl", O + "sdfr_boolq_qwen3_14b.jsonl"]:
                r = rd(p)
                idk = "qid" if "qid" in r[0] else "id" if "id" in r[0] else None
                sub = [x for x in r if idk and x.get(idk) in evq]
                if "correct" in r[0]:
                    print("  %s: rows %d, correct %d (%.2f%%); rows whose %s is in the %d-row eval file: %d, correct %d%s"
                          % (p, len(r), sum(bool(x["correct"]) for x in r), 100.0 * sum(bool(x["correct"]) for x in r) / len(r), idk, len(evq), len(sub),
                             sum(bool(x["correct"]) for x in sub), " (%.2f%%)" % (100.0 * sum(bool(x["correct"]) for x in sub) / len(sub)) if sub else ""))
            print()
            continue
        print("script: %s  MD5 %s" % (d["script"], md5(d["script"])))
        src = open(d["script"], encoding="utf-8").read().split("\n")
        for i, l in enumerate(src, 1):
            if re.search(r"max_tokens|enable_thinking|MODEL_PATH *=|def extract|max_model_len", l):
                print("  line %d: %s" % (i, l.strip()))
        out = rd(d["out"])
        ns = funcs(d["script"], ("extract_answer", "extract_gold"))
        stored = sum(bool(r["correct"]) for r in out)
        re_pred = [ns["extract_answer"](r["generated"].split("</think>")[-1]) for r in out]
        recomputed = sum(p == r["gold"] for p, r in zip(re_pred, out))
        print("sealed output: %s  rows %d  MD5 %s" % (d["out"], len(out), md5(d["out"])))
        print("accuracy from the stored 'correct' field: %d / %d = %.2f%%" % (stored, len(out), 100.0 * stored / len(out)))
        print("accuracy recomputed (the script's extract_answer on the text after </think>, against the stored gold): %d / %d = %.2f%%; recomputed pred equals stored pred on %d rows"
              % (recomputed, len(out), 100.0 * recomputed / len(out), sum(p == r["pred"] for p, r in zip(re_pred, out))))
        ev = rd(d["ev"])
        print("rows without </think> in the generation: %d; rows with an empty pred: %d; output rows %d = eval rows %d: %s"
              % (sum("</think>" not in r["generated"] for r in out), sum(r["pred"] == "" for r in out), len(out), len(ev), len(out) == len(ev)))
        print()


def pools():
    for name, d in DS.items():
        ev, ur, idx = to_raw_index(d)
        en = rd(d["en"])
        ev_en = [kk(en[i], d["key"]) if i is not None else None for i in idx]
        ev_ur = [kk(r, d["key"]) for r in ev]
        print("== %s" % name)
        for p in d["pools"]:
            pool = rd(p)
            pk = {kk(r, d["key"]) for r in pool}
            line = ("pool %s  rows %d  MD5 %s\n  eval rows whose English source text (%s) is in the pool: %d of %d; eval rows whose Urdu text is in the pool: %d"
                    % (p, len(pool), md5(p), "+".join(d["key"]), sum(1 for x in ev_en if x is not None and x in pk), len(ev), sum(1 for x in ev_ur if x in pk)))
            for idf in ("qid", "id"):
                if idf in ev[0] and idf in pool[0]:
                    line += "; eval '%s' values in the pool: %d" % (idf, len({r[idf] for r in ev} & {r[idf] for r in pool}))
            print(line)
            print("  pool rows with Arabic-script letters in %s: %d" % (d["key"][0], sum(arabic(flat(r[d["key"][0]])) for r in pool)))
        if name == "gsm8k":
            pool = rd(d["pools"][0])
            pa = {norm(r["answer"]) for r in pool}
            print("  gsm8k: eval rows whose 'answer' string is in the pool answers: %d of %d" % (sum(norm(r["answer"]) in pa for r in ev), len(ev)))
            for l in open(E + "g1_sdfr_gsm8k_clean.py", encoding="utf-8").read().split("\n"):
                if re.match(r"^(POOL_EXPECT|POOL_AFTER_EXPECT|TOP_K)\s*=", l):
                    print("  g1_sdfr_gsm8k_clean.py: %s" % l.strip())
        print()


def strategyqa():
    mp = rd("data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl")
    allq = [r["urbench_qid"] for r in mp]
    print("mapped file rows %d, distinct urbench_qid %d, MD5 %s" % (len(allq), len(set(allq)), md5("data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl")))

    def txt(p):
        return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    sets = {
        "eval458 (is_eval in the mapped file)": {r["urbench_qid"] for r in mp if str(r.get("is_eval")).lower() in ("true", "1")},
        "eval458 (data/sdfr_splits/strategyqa_eval.jsonl)": {r["qid"] for r in rd(S + "strategyqa_eval.jsonl")},
        "DEV200 (dev200_seed4242_qids.txt)": set(txt("data/strategyqa_official/dev200_seed4242_qids.txt")),
        "RDEV300 (efbpt/rdev300_seed2026.jsonl)": {r["urbench_qid"] for r in rd("data/strategyqa_official/efbpt/rdev300_seed2026.jsonl")},
        "DEV50 (dev50_seed42_qids.txt)": set(txt("data/strategyqa_official/dev50_seed42_qids.txt")),
        "train manifest 100 (efbpt/plan_a_qids_100.txt)": set(txt("data/strategyqa_official/efbpt/plan_a_qids_100.txt")),
        "train manifest 250 (efbpt/plan_a_qids_250.txt)": set(txt("data/strategyqa_official/efbpt/plan_a_qids_250.txt")),
        "train manifest 500 (efbpt/plan_a_qids_500.txt)": set(txt("data/strategyqa_official/efbpt/plan_a_qids_500.txt")),
    }
    extra = {"AUDIT30 (efbpt/audit30_qids.txt)": set(txt("data/strategyqa_official/efbpt/audit30_qids.txt")), "BLIND30 (efbpt/blind30_qids.txt)": set(txt("data/strategyqa_official/efbpt/blind30_qids.txt"))}
    A = set(allq)
    for k, v in list(sets.items()) + list(extra.items()):
        print("  %-52s %4d qids, %4d of them in the mapped file" % (k, len(v), len(v & A)))
    names = list(sets)
    print("the two eval458 lists are the same set: %s" % (sets[names[0]] == sets[names[1]]))
    print("pairwise overlaps (qids in both):")
    for i in range(1, len(names)):
        for j in range(i + 1, len(names)):
            print("  %-48s x %-48s %d" % (names[i], names[j], len(sets[names[i]] & sets[names[j]])))
    union = set().union(*sets.values())
    print("union of eval458, DEV200, RDEV300, DEV50 and the three train manifests: %d qids" % len(union & A))
    free = A - union
    print("rows of the %d in none of them: %d" % (len(A), len(free)))
    gold = {r["urbench_qid"]: r["answer"] for r in mp}
    print("  of these, answer true %d / false %d" % (sum(gold[q] is True for q in free), sum(gold[q] is False for q in free)))
    free2 = free - extra["AUDIT30 (efbpt/audit30_qids.txt)"] - extra["BLIND30 (efbpt/blind30_qids.txt)"]
    print("also outside AUDIT30 and BLIND30: %d" % len(free2))
    st1 = rd("data/strategyqa_official/efbpt/stage1_report.jsonl")
    print("stage1_report.jsonl rows %d, fields %s" % (len(st1), list(st1[0].keys())[:10]))
    pool = {r["qid"] for r in rd(S + "strategyqa_pool.jsonl")}
    print("data/sdfr_splits/strategyqa_pool.jsonl (the SDFR demo pool): %d qids; of the %d free rows, %d are in it" % (len(pool), len(free), len(free & pool)))


def models():
    base = os.path.expanduser("~/downloaded_models")
    for cfg in sorted(glob.glob(base + "/*/config.json") + glob.glob(base + "/*/*/config.json")):
        d = os.path.dirname(cfg)
        idx = os.path.join(d, "model.safetensors.index.json")
        shards = ""
        if os.path.exists(idx):
            sh = sorted(set(json.load(open(idx))["weight_map"].values()))
            shards = "shards %d of %d present" % (sum(os.path.exists(os.path.join(d, s)) for s in sh), len(sh))
        else:
            w = [f for f in os.listdir(d) if f.endswith((".safetensors", ".bin"))]
            shards = "no safetensors index; weight files %d" % len(w)
        size = subprocess.check_output(["du", "-shL", d]).decode().split()[0]
        link = " -> " + os.path.realpath(d) if os.path.islink(d) else ""
        print("%-70s %6s  %s%s" % (os.path.relpath(d, base), size, shards, link))
    print("directories without config.json:", sorted(os.path.relpath(p, base) for p in glob.glob(base + "/*/*") + glob.glob(base + "/*") if os.path.isdir(p) and not os.path.exists(os.path.join(p, "config.json"))
                                                     and not glob.glob(os.path.join(p, "*", "config.json"))))


if __name__ == "__main__":
    {"data": data, "baselines": baselines, "pools": pools, "strategyqa": strategyqa, "models": models}[sys.argv[1]]()
```
