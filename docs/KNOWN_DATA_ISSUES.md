# Known data issues

Written 2026-11-15. This file lists four oddities in the data files and one script issue. They were found by the read-only inventory `docs/handoff/D14_PREP.md`. The data and the script are left as they are; nothing is fixed here. Each item has the command that shows it and the output of that command. Every command is run from the repo root and only reads files.

The data files under `data/` are not tracked by git, so the MD5 of each file read is given.

## 1. PIQA: one evaluation row occurs twice

`data/sdfr_splits/piqa_eval.jsonl` (MD5 416fb9dabf496f2416b39985fbf7fdf5) has 150 rows and 149 distinct rows. Lines 104 and 105 have the same goal, sol1, sol2 and label.

```bash
python3 -I -c "
import json, collections
r = [json.loads(l) for l in open('data/sdfr_splits/piqa_eval.jsonl', encoding='utf-8') if l.strip()]
c = collections.defaultdict(list)
for i, x in enumerate(r, 1): c[(x['goal'], x['sol1'], x['sol2'])].append(i)
rep = [v for v in c.values() if len(v) > 1]
print('rows', len(r), 'distinct', len(c), 'repeated at lines', rep, 'labels', [[r[i - 1]['label'] for i in v] for v in rep])
"
```

```
rows 150 distinct 149 repeated at lines [[104, 105]] labels [[1, 1]]
```

## 2. GSM8K: one evaluation question is not in Urdu

`data/sdfr_splits/gsm8k_eval.jsonl` (MD5 cb701f18965757d563391a9cda2242aa) has 700 rows. The question on line 222 has no Arabic-script letter; it is the English text.

```bash
python3 -I -c "
import json
r = [json.loads(l) for l in open('data/sdfr_splits/gsm8k_eval.jsonl', encoding='utf-8') if l.strip()]
hit = [(i, x['question']) for i, x in enumerate(r, 1) if not any('؀' <= ch <= 'ۿ' for ch in x['question'])]
print('rows', len(r), 'questions with no Arabic-script letter', len(hit))
for i, q in hit: print('line', i, repr(q[:80]))
"
```

```
rows 700 questions with no Arabic-script letter 1
line 222 'Tim spent $50.00 on Chinese take-out.  80% of the cost went to 4 entrees.  The r'
```

## 3. CSQA: the Urdu and English source files disagree on one id

`data/csqa_raw/csqa_train_1500_ur.jsonl` (MD5 396601c0f6c1b26aaa4c29c3fc6cf018) and `data/csqa_raw/csqa_train_1500_en.jsonl` (MD5 4a47619e60c4be2478757b8c8e6581bc) have 1500 rows each. The `id` field is the same at 1499 positions. On line 5 the Urdu file has a `z` where the English file has a `1`.

```bash
python3 -I -c "
import json
ur = [json.loads(l) for l in open('data/csqa_raw/csqa_train_1500_ur.jsonl', encoding='utf-8') if l.strip()]
en = [json.loads(l) for l in open('data/csqa_raw/csqa_train_1500_en.jsonl', encoding='utf-8') if l.strip()]
bad = [(i, a.get('id'), b.get('id')) for i, (a, b) in enumerate(zip(ur, en), 1) if a.get('id') != b.get('id')]
print('rows', len(ur), len(en), 'positions with a different id', len(bad))
for i, a, b in bad: print('line', i, 'ur id', a, 'en id', b)
"
```

```
rows 1500 1500 positions with a different id 1
line 5 ur id 23505889b94e880c3e89cff4ba1z19860 en id 23505889b94e880c3e89cff4ba119860
```

Line 5 is outside the evaluation split (the evaluation rows are source positions 1200 to 1499, see `docs/handoff/D14_PREP.md` section 1).

## 4. StrategyQA: every row not used by a set is in the SDFR demo pool

Of the 2290 rows of `data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl` (MD5 bb982798279f5d519ddef01f5cb7ba8c), 916 are in none of eval458, DEV200, DEV50, RDEV300 and the three training manifests. All 916 are in `data/sdfr_splits/strategyqa_pool.jsonl` (MD5 55b122852cdcd9df2c932497f9611eb9), the demo pool of the SDFR runs. So these rows have been shown to a model as demonstrations, but have not been evaluation rows of any of these sets. 867 of the 916 are also outside AUDIT30 and BLIND30.

```bash
python3 -I -c "
import json
D = 'data/strategyqa_official/'
def rd(p): return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
def txt(p): return {l.strip() for l in open(p, encoding='utf-8') if l.strip()}
allq = {r['urbench_qid'] for r in rd(D + 'strategyqa_official_mapped_urbench_qid.jsonl')}
used = {r['qid'] for r in rd('data/sdfr_splits/strategyqa_eval.jsonl')} | txt(D + 'dev200_seed4242_qids.txt') | txt(D + 'dev50_seed42_qids.txt') | {r['urbench_qid'] for r in rd(D + 'efbpt/rdev300_seed2026.jsonl')}
for n in ('100', '250', '500'): used |= txt(D + 'efbpt/plan_a_qids_%s.txt' % n)
free = allq - used
pool = {r['qid'] for r in rd('data/sdfr_splits/strategyqa_pool.jsonl')}
print('rows', len(allq), 'in no set', len(free), 'of these in the SDFR demo pool', len(free & pool), 'of these also outside AUDIT30 and BLIND30', len(free - txt(D + 'efbpt/audit30_qids.txt') - txt(D + 'efbpt/blind30_qids.txt')))
"
```

```
rows 2290 in no set 916 of these in the SDFR demo pool 916 of these also outside AUDIT30 and BLIND30 867
```

## 5. Script issue: `eval/eval_boolq_three_shot_vllm.py` names a prompt file that does not exist

The script (MD5 7fa1ae0f8890c95dfd2c4763de162857) sets `PROMPT_PATH` to `../prompts/boolq/three_shot.txt`. There is no such file; the three-shot prompt files are `three_shot_p1.txt`, `three_shot_p2.txt` and `three_shot_p3.txt`. The script also names the Llama-3.1-70B model at present, not Qwen3-14B. As it stands the script cannot reproduce the Qwen3-14B three-shot run in `experiments.md`.

```bash
grep -n 'PROMPT_PATH = \|MODEL_NAME = ' eval/eval_boolq_three_shot_vllm.py; ls prompts/boolq/three_shot.txt; ls prompts/boolq/ | grep three_shot
```

```
10:PROMPT_PATH = "../prompts/boolq/three_shot.txt"
12:MODEL_NAME = "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
ls: cannot access 'prompts/boolq/three_shot.txt': No such file or directory
three_shot_p1.txt
three_shot_p2.txt
three_shot_p3.txt
```
