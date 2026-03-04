## BoolQ – Urdu – Qwen2.5-1.5B-Instruct

Model: Qwen2.5-1.5B-Instruct  
Dataset: BoolQ (Urdu)  
Total examples: 1550  

### BoolQ Zero-shot
- Used items: 1550
- Scored on: 1281
- Accuracy: 62.69%
- Output file: outputs/boolq/qwen2.5_1.5b/boolq_zero_shot_qwen2.5_1.5b.jsonl

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1450
- Accuracy: 55.24%
- Output file: outputs/boolq/qwen2.5_1.5b/boolq_three_shot_qwen2.5_1.5b.jsonl

### BoolQ Chain-of-Thought (CoT)
- Used items: 1550
- Scored on: 1500
- Accuracy: 47.00%
- Output file: outputs/boolq/qwen2.5_1.5b/boolq_cot_qwen2.5_1.5b.jsonl

## BoolQ – Urdu – Qwen2.5-7B-Instruct

Model: Qwen2.5-7B-Instruct  
Dataset: BoolQ (Urdu)  
Total examples: 1550  

### BoolQ Zero-shot
- Used items: 1550
- Scored on: 1548
- Accuracy: 66.73%
- Output file: outputs/boolq/qwen2.5_7b/boolq_zero_shot_qwen2.5_7b.jsonl

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1546
- Accuracy: 71.41%
- Output file: outputs/boolq/qwen2.5_7b/boolq_three_shot_qwen2.5_7b.jsonl
> Note: For the BoolQ 3-shot experiment with Qwen2.5-7B, passages were truncated
> to fit within the 4096-token context limit after encountering max_model_len
> errors when running on the full dataset. Zero-shot experiments were run
> without truncation.

### BoolQ CoT
- Used items: 1550
- Scored on: 1550
- Accuracy: 62.26%
- Output file: outputs/boolq/qwen2.5_7b/boolq_cot_qwen2.5_7b.jsonl
> Note: The BoolQ CoT experiment with Qwen2.5-7B uses the same passage/question
> truncation settings as the 3-shot run (passage ≤ 3000 chars, question ≤ 512 chars) to respect the 4096-token context limit.

## BoolQ – Urdu – Qwen2.5-14B-Instruct (AWQ)

Model: Qwen2.5-14B-Instruct-AWQ  
Dataset: BoolQ (Urdu)  
Total examples: 1550  

### BoolQ Zero-shot
- Used items:  1550
- Scored on:   1550
- Accuracy:    76.26%
- Output file: outputs/boolq/qwen2.5_14b_awq/boolq_zero_shot_qwen2.5_14b_awq.jsonl

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1549
- Accuracy: 72.24%
- Output file: outputs/boolq/qwen2.5_14b_awq/boolq_three_shot_qwen2.5_14b_awq.jsonl  

> Note: For the BoolQ 3-shot experiment with Qwen2.5-14B-Instruct-AWQ, passages were
> truncated to fit within the 3072-token context limit after encountering
> `max_model_len` errors when running on the full dataset. Zero-shot experiments
> were run without truncation.

### BoolQ CoT
- Used items: 1550
- Scored on: 1550
- Accuracy: 76.13%
- Output file: outputs/boolq/qwen2.5_14b_awq/boolq_cot_qwen2.5_14b_awq.jsonl  

> Note: For the BoolQ CoT experiment with Qwen2.5-14B-Instruct-AWQ, passages were truncated to fit within the 3072-token context limit.

## BoolQ – Urdu – Meta-LLaMA-3.1-8B-Instruct

Model: Meta-LLaMA-3.1-8B-Instruct  
Dataset: BoolQ (Urdu)  
Total examples: 1550  

### BoolQ Zero-shot
- Used items: 1550  
- Scored on: 1548  
- Accuracy: 78.49%  
- Output file: outputs/boolq/meta_llama3.1_8b/boolq_zero_shot_meta_llama3.1_8b.jsonl  

> **Note:** Meta-LLaMA-3.1-8B-Instruct improves BoolQ-Urdu zero-shot accuracy  
> over Qwen2.5-14B-Instruct-AWQ (76.26% → 78.49%), suggesting stronger  
> direct binary QA behavior in this setting.

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1546
- Accuracy: 66.82%
- Output file: outputs/boolq/meta_llama3.1_8b/boolq_three_shot_meta_llama3.1_8b.jsonl  

> **Note:** Compared to Qwen2.5-14B-Instruct-AWQ (72.24%), Meta-LLaMA-3.1-8B-Instruct shows a noticeable drop in 3-shot performance, suggesting weaker few-shot adaptation on BoolQ-Urdu despite strong zero-shot results.

### BoolQ Chain-of-Thought (CoT)
- Used items: 1550
- Scored on: 1549
- Accuracy: 71.40%
- Output file: outputs/boolq/meta_llama3.1_8b/boolq_cot_meta_llama3.1_8b.jsonl  

> **Note:** Passages were truncated to fit the model context window; even with truncation,
> Meta-LLaMA-3.1-8B-Instruct remains below Qwen2.5-14B-Instruct-AWQ on BoolQ-Urdu CoT.

## BoolQ – Urdu – DeepSeek-R1-Distill-Qwen-7B

Model: DeepSeek-R1-Distill-Qwen-7B  
Dataset: BoolQ (Urdu)  
Total examples: 1550  

### BoolQ Zero-shot (Fixed)
- Used items: 1550
- Scored on: 1538 (99.23%)
- Accuracy: 56.76%
- Output file: outputs/boolq/deepseek_r1_distill_qwen_7b/boolq_zero_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied (no stop, increased max_tokens, answer extraction). Accuracy (56.76%) slightly higher than original (56.56%) with improved coverage (99.23% vs 98.77%).

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1529 (98.65%)
- Accuracy: 62.07%
- Output file: outputs/boolq/deepseek_r1_distill_qwen_7b/boolq_three_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied. 3-shot accuracy (62.07%) shows improvement over zero-shot (56.76%) with high coverage (98.65%).

### BoolQ Chain-of-Thought (CoT)
- Used items: 1550
- Scored on: 1543 (99.55%)
- Accuracy: 61.31%
- Output file: outputs/boolq/deepseek_r1_distill_qwen_7b/boolq_cot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied. CoT accuracy (61.31%) falls between zero-shot (56.76%) and 3-shot (62.07%) with excellent coverage (99.55%).

---

## CSQA – Urdu – Qwen2.5-1.5B-Instruct

Model: Qwen2.5-1.5B-Instruct  
Dataset: CSQA (Urdu)  
Total examples: 1500  

### CSQA Zero-shot
- Used items:       1500
- Answered (A–E):   367 (24.47%)
- Accuracy overall: 6.07%   (correct / 1500)
- Accuracy answered: 24.80% (correct / 367)
- Output file:      outputs/csqa/qwen2.5_1.5b/csqa_zero_shot_qwen2.5_1.5b.jsonl

### CSQA 3-shot
- Used items:       1500
- Answered (A–E):   659 (43.93%)
- Accuracy overall: 9.53%   (correct / 1500)
- Accuracy answered: 21.70% (correct / 659)
- Output file:      outputs/csqa/qwen2.5_1.5b/csqa_three_shot_qwen2.5_1.5b.jsonl

### CSQA Chain-of-Thought (CoT)
- Used items:       1500
- Answered (A–E):   1500 (100.00%)
- Accuracy overall: 25.20%  (correct / 1500)
- Accuracy answered: 25.20% (correct / 1500)
- Output file:      outputs/csqa/qwen2.5_1.5b/csqa_cot_qwen2.5_1.5b.jsonl

> **Note:**  
> In CSQA zero-shot and 3-shot, the model often returned Urdu fillers like “________.” instead of a letter.  
> These were treated as `pred = None`, which lowers “answered” coverage.  
> With the CoT prompt (`حتمی جواب: X`), the model consistently returns a clean A–E choice, so coverage becomes 100%.

## CSQA – Urdu – Qwen2.5-7B-Instruct

Model: Qwen2.5-7B-Instruct  
Dataset: CSQA (Urdu)  
Total examples: 1500  

### CSQA Zero-shot
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 45.40%   <!-- correct / 1500 -->
- Accuracy answered: 45.40%  <!-- correct / 1500 -->
- Output file: outputs/csqa/qwen2.5_7b/csqa_zero_shot_qwen2.5_7b.jsonl

> Note: Predictions are normalized to one of A–E using Latin letters, Urdu
> option letters (الف، ب، ج، د، ہ), digits (1–5 / ۱–۵ / ١–٥), and matching
> against option texts where possible.

### CSQA 3-shot
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 44.40%  <!-- correct / 1500 -->
- Accuracy answered: 44.40%  <!-- correct / 1500 -->
- Output file: outputs/csqa/qwen2.5_7b/csqa_three_shot_qwen2.5_7b.jsonl

### CSQA Chain-of-Thought (CoT)
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 45.13%  <!-- correct / 1500 -->
- Accuracy answered: 45.13%  <!-- correct / 1500 -->
- Output file: outputs/csqa/qwen2.5_7b/csqa_cot_qwen2.5_7b.jsonl

> Note: For CSQA-Urdu with Qwen2.5-7B, CoT yields accuracy very close to
> zero-shot (45.40% → 45.13%) and slightly above the 3-shot setting (44.40%),
> suggesting limited benefit from few-shot examples or explicit reasoning here.

## CSQA – Urdu – Qwen2.5-14B-Instruct (AWQ)

Model: Qwen2.5-14B-Instruct-AWQ  
Dataset: CSQA (Urdu)  
Total examples: 1500  

### CSQA Zero-shot
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 50.93%
- Accuracy answered: 50.93%
- Output file: outputs/csqa/qwen2.5_14b_awq/csqa_zero_shot_qwen2.5_14b_awq.jsonl  
> Note: As we move from 1.5B → 7B → 14B models, CSQA accuracy steadily improves,
> showing a clear gain in Urdu commonsense reasoning with larger Qwen variants.

### CSQA 3-shot
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 46.53%
- Accuracy answered: 46.53%
- Output file: outputs/csqa/qwen2.5_14b_awq/csqa_three_shot_qwen2.5_14b_awq.jsonl  
> Note: Accuracy is slightly higher than with Qwen2.5-7B, indicating a modest gain
> from scaling to the 14B model in the 3-shot setting.

### CSQA Chain-of-Thought (CoT)
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 49.73%
- Accuracy answered: 49.73%
- Output file: outputs/csqa/qwen2.5_14b_awq/csqa_cot_qwen2.5_14b_awq.jsonl  
> Note: CoT with Qwen2.5-14B performs better than the 7B CoT and 3-shot results,
> though it still slightly trails the 14B zero-shot setting on CSQA-Urdu.

## CSQA – Urdu – Meta-LLaMA-3.1-8B-Instruct

Model: Meta-LLaMA-3.1-8B-Instruct  
Dataset: CSQA (Urdu)  
Total examples: 1500  

### CSQA Zero-shot
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 49.20%  <!-- correct / 1500 -->
- Accuracy answered: 49.20%  <!-- correct / 1500 -->
- Output file: outputs/csqa/meta_llama3.1_8b/csqa_zero_shot_meta_llama3.1_8b.jsonl  

> **Note:** Zero-shot accuracy is slightly below Qwen2.5-14B-Instruct-AWQ on CSQA-Urdu,
> indicating a small drop in commonsense MCQ performance despite full coverage.

### CSQA 3-shot
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 48.13%  <!-- correct / 1500 -->
- Accuracy answered: 48.13%  <!-- correct / 1500 -->
- Output file: outputs/csqa/meta_llama3.1_8b/csqa_three_shot_meta_llama3.1_8b.jsonl  

> **Note:** Compared to its own zero-shot (49.20%), 3-shot slightly underperforms here;
> the benefit from few-shot examples is not consistent for Meta-LLaMA-3.1-8B on CSQA-Urdu. And still slighlty better than Qwen 14B.

### CSQA Chain-of-Thought (CoT)
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 46.47%  <!-- correct / 1500 -->
- Accuracy answered: 46.47%  <!-- correct / 1500 -->
- Output file: outputs/csqa/meta_llama3.1_8b/csqa_cot_meta_llama3.1_8b.jsonl  

> **Note:** CoT performance remains behind Qwen2.5-14B-Instruct-AWQ on CSQA-Urdu
> (49.73% vs 46.47%), suggesting Qwen-14B benefits more from explicit reasoning here.

## CSQA – Urdu – DeepSeek-R1-Distill-Qwen-7B

Model: DeepSeek-R1-Distill-Qwen-7B  
Dataset: CSQA (Urdu)  
Total examples: 1500  

### CSQA Zero-shot
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 26.33%
- Accuracy answered: 26.33%
- Output file: outputs/csqa/deepseek_r1_distill_qwen_7b/csqa_zero_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied (no stop, increased max_tokens, answer extraction). Accuracy (26.33%) slightly higher than original (26.07%) with full coverage achieved. Comparable to Qwen-1.5B (24.80%) on CSQA zero-shot.

### CSQA 3-shot
- Used items: 1500
- Answered (A–E): 1491 (99.40%)
- Accuracy overall: 22.87%
- Accuracy answered: 23.00%
- Output file: outputs/csqa/deepseek_r1_distill_qwen_7b/csqa_three_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note (Fixes Applied):**  
> Initially, DeepSeek answered rate was low because generation stopped at first newline (`stop=["\n"]`), cutting off its answer after reasoning.  
>  
> **Changes (evaluation logic unchanged):**  
> - Removed `stop=["\n"]` and increased `max_tokens` to 512  
> - Added answer extraction that searches output from bottom for final letter  
> - Auto-detects DeepSeek so other models keep original parameters  
>  
> **Result:** Answered rate improved from ~24% to 99.40%, accuracy at 22.87% (comparable to Qwen-1.5B's 21.70%). Prompt and normalization unchanged.

### CSQA Chain-of-Thought (CoT)
- Used items: 1500
- Answered (A–E): 1500 (100.00%)
- Accuracy overall: 23.33%
- Accuracy answered: 23.33%
- Output file: outputs/csqa/deepseek_r1_distill_qwen_7b/csqa_cot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** Same DeepSeek-specific fixes applied as in 3-shot (no stop, increased max_tokens, answer extraction). Accuracy comparable to 3-shot (22.87%), with full coverage achieved.

---

## PIQA – Urdu – Qwen2.5-1.5B-Instruct

Model: Qwen2.5-1.5B-Instruct  
Dataset: PIQA (Urdu)  
Total examples: 750  

### PIQA Zero-shot
- Used items: 750
- Answered (0/1): 521 (69.47%)
- Accuracy overall: 33.73%   <!-- correct / 750 -->
- Accuracy answered: 48.56%  <!-- correct / 521 -->
- Output file: outputs/piqa/qwen2.5_1.5b/piqa_zero_shot_qwen2.5_1.5b.jsonl

> Note: Only predictions that explicitly contain `0` or `1` are scored.  
> Items where the model does not produce a clear digit are counted as `pred = None` and excluded from “Accuracy answered”.

### PIQA 3-shot
- Used items: 750
- Answered (0/1): 750 (100.00%)
- Accuracy overall: 51.07%
- Accuracy answered: 51.07%
- Output file: outputs/piqa/qwen2.5_1.5b/piqa_three_shot_qwen2.5_1.5b.jsonl

### PIQA Chain-of-Thought (CoT)
- Used items: 750
- Answered (0/1): 749 (99.87%)
- Accuracy overall: 51.07%   <!-- correct / 750 -->
- Accuracy answered: 51.13%  <!-- correct / 749 -->
- Output file: outputs/piqa/qwen2.5_1.5b/piqa_cot_qwen2.5_1.5b.jsonl

> Note: For PIQA, 3-shot and CoT give very similar accuracy (≈51%), only slightly above the random 0/1 baseline (50%).  
> CoT makes the model almost always output a digit (coverage ~100%), but does not significantly improve accuracy for this small model.

## PIQA – Urdu – Qwen2.5-7B-Instruct

Model: Qwen2.5-7B-Instruct  
Dataset: PIQA (Urdu)  
Total examples: 750  

### PIQA Zero-shot
- Used items: 750
- Answered (0/1): 747 (99.60%)
- Accuracy overall: 59.20%
- Accuracy answered: 59.44%
- Output file: outputs/piqa/qwen2.5_7b/piqa_zero_shot_qwen2.5_7b.jsonl

> Note: As with other PIQA runs, only predictions that clearly contain `0` or `1`
> are scored. Items where the model does not produce a usable digit are treated
> as `pred = None` and excluded from “Accuracy answered”.

### PIQA 3-shot
- Used items: 750
- Answered (0/1): 746 (99.47%)
- Accuracy overall: 62.00%
- Accuracy answered: 62.33%
- Output file: outputs/piqa/qwen2.5_7b/piqa_three_shot_qwen2.5_7b.jsonl

> Note: 3-shot improves both coverage and accuracy compared to the zero-shot
> setting for Qwen2.5-7B on PIQA-Urdu.

### PIQA Chain-of-Thought (CoT)
- Used items: 750
- Answered (0/1): 673 (89.73%)
- Accuracy overall: 44.93%
- Accuracy answered: 50.07%
- Output file: outputs/piqa/qwen2.5_7b/piqa_cot_qwen2.5_7b.jsonl

> Note: Compared to the 3-shot setting, CoT reduces both coverage and accuracy
> for Qwen2.5-7B on PIQA-Urdu, with accuracy only slightly above random among
> the answered items.

## PIQA – Urdu – Qwen2.5-14B-Instruct (AWQ)

Model: Qwen2.5-14B-Instruct-AWQ  
Dataset: PIQA (Urdu)  
Total examples: 750  

### PIQA Zero-shot
- Used items: 750
- Answered (0/1): 747 (99.60%)
- Accuracy overall: 60.67%
- Accuracy answered: 60.91%
- Output file: outputs/piqa/qwen2.5_14b_awq/piqa_zero_shot_qwen2.5_14b_awq.jsonl  

> Note: As with other PIQA runs, only predictions that clearly contain `0` or `1`
> are scored. Items with `pred = None` are excluded from “Accuracy answered”; with Qwen2.5-14B the accuracy improves slightly over the 7B variant.

### PIQA 3-shot
- Used items: 750
- Answered (0/1): 750 (100.00%)
- Accuracy overall: 65.33%
- Accuracy answered: 65.33%
- Output file: outputs/piqa/qwen2.5_14b_awq/piqa_three_shot_qwen2.5_14b_awq.jsonl  

> Note: 3-shot with Qwen2.5-14B not only improves accuracy over the 7B and zero-shot runs, but also reaches full (100%) coverage on PIQA-Urdu.

### PIQA Chain-of-Thought (CoT)
- Used items: 750
- Answered (0/1): 750 (100.00%)
- Accuracy overall: 58.93%
- Accuracy answered: 58.93%
- Output file: outputs/piqa/qwen2.5_14b_awq/piqa_cot_qwen2.5_14b_awq.jsonl  

> Note: Compared to the Qwen2.5-7B CoT run, the 14B model achieves both full
> coverage and a clear accuracy gain, making CoT genuinely useful for PIQA-Urdu.

## PIQA – Urdu – Meta-LLaMA-3.1-8B-Instruct

Model: Meta-LLaMA-3.1-8B-Instruct  
Dataset: PIQA (Urdu)  
Total examples: 750  

### PIQA Zero-shot
- Used items: 750
- Answered (0/1): 501 (66.80%)
- Accuracy overall: 35.20%
- Accuracy answered: 52.69%
- Output file: outputs/piqa/meta_llama3.1_8b/piqa_zero_shot_meta_llama3.1_8b.jsonl  

> Note: Only predictions that clearly contain `0` or `1` are scored. Items with
> `pred = None` are excluded from “Accuracy answered”.
> Meta-LLaMA-3.1-8B performs better than Qwen2.5-1.5B on PIQA-Urdu, but remains
> behind the Qwen2.5-7B and Qwen2.5-14B runs in both coverage and overall accuracy.

### PIQA 3-shot
- Used items: 750
- Answered (0/1): 749 (99.87%)
- Accuracy overall: 55.87%
- Accuracy answered: 55.94%
- Output file: outputs/piqa/meta_llama3.1_8b/piqa_three_shot_meta_llama3.1_8b.jsonl  

> **Note:** Coverage rises to ~100%, but accuracy stays in the same range as the stronger PIQA runs:
> better than Qwen2.5-1.5B, yet still behind Qwen2.5-7B and Qwen2.5-14B on PIQA-Urdu.

### PIQA Chain-of-Thought (CoT)
- Used items: 750
- Answered (0/1): 749 (99.87%)
- Accuracy overall: 49.47%
- Accuracy answered: 49.53%
- Output file: outputs/piqa/meta_llama3.1_8b/piqa_cot_meta_llama3.1_8b.jsonl  

> **Note:** Despite high coverage, CoT underperforms all Qwen baselines (1.5B, 7B, and 14B),
> indicating that Meta-LLaMA-3.1-8B does not benefit from explicit reasoning on PIQA-Urdu.

## PIQA – Urdu – DeepSeek-R1-Distill-Qwen-7B

Model: DeepSeek-R1-Distill-Qwen-7B  
Dataset: PIQA (Urdu)  
Total examples: 750  

### PIQA Zero-shot
- Used items: 750
- Answered (0/1): 747 (99.60%)
- Accuracy overall: 50.67%
- Accuracy answered: 50.87%
- Output file: outputs/piqa/deepseek_r1_distill_qwen_7b/piqa_zero_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied (no stop, increased max_tokens, answer extraction). Accuracy near random baseline (50%), comparable to Qwen-1.5B (48.56% answered accuracy).

### PIQA 3-shot
- Used items: 750
- Answered (0/1): 749 (99.87%)
- Accuracy overall: 53.47%
- Accuracy answered: 53.54%
- Output file: outputs/piqa/deepseek_r1_distill_qwen_7b/piqa_three_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** Same DeepSeek-specific fixes applied. 3-shot shows modest improvement over zero-shot (50.67% → 53.47%).

### PIQA Chain-of-Thought (CoT)
- Used items: 750
- Answered (0/1): 748 (99.73%)
- Accuracy overall: 52.93%
- Accuracy answered: 53.07%
- Output file: outputs/piqa/deepseek_r1_distill_qwen_7b/piqa_cot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** Same DeepSeek-specific fixes applied. CoT performance (52.93%) similar to zero-shot (50.67%) and 3-shot (53.47%), showing minimal gain from reasoning for PIQA with this model.

---

## StrategyQA – Urdu – Qwen2.5-1.5B-Instruct

Model: Qwen2.5-1.5B-Instruct  
Dataset: StrategyQA (Urdu)  
Total examples: 2290  

### StrategyQA Zero-shot
- Used items:       2290
- Answered (ہاں/نہیں): 2288 (99.91%)
- Accuracy overall: 53.28%
- Accuracy answered: 53.32%
- Output file: outputs/strategyqa/qwen2.5_1.5b/strategyqa_zero_shot_qwen2.5_1.5b.jsonl

> Note: The model produces a valid binary answer for nearly all questions under a constrained zero-shot prompt.  
> Despite high coverage, accuracy remains only slightly above the random baseline (50%), indicating limited multi-hop reasoning ability in zero-shot for StrategyQA-Urdu.

### StrategyQA 3-shot
- Used items:       2290
- Answered (ہاں/نہیں): 2289 (99.96%)
- Accuracy overall: 48.38%
- Accuracy answered: 48.41%
- Output file: outputs/strategyqa/qwen2.5_1.5b/strategyqa_three_shot_qwen2.5_1.5b.jsonl

> Note: The 3-shot prompt achieves almost perfect coverage (2289/2290 items) but *reduces* accuracy compared to the zero-shot setting (53.28% → 48.38%).  
> This suggests that, for Qwen2.5-1.5B on StrategyQA-Urdu, few-shot examples mainly help format the answer but do not improve (and may even hurt) multi-hop factual reasoning.

### StrategyQA Chain-of-Thought (CoT)
- Used items:       2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 49.91%
- Accuracy answered: 49.91%
- Output file: outputs/strategyqa/qwen2.5_1.5b/strategyqa_cot_qwen2.5_1.5b.jsonl

> Note: While CoT prompting achieves perfect answer coverage, it does not improve accuracy over the zero-shot setting and performs only slightly better than 3-shot prompting. This suggests that explicit step-by-step reasoning in Urdu does not reliably benefit multi-hop factual reasoning for a 1.5B model on StrategyQA.

## StrategyQA – Urdu – Qwen2.5-7B-Instruct

Model: Qwen2.5-7B-Instruct  
Dataset: StrategyQA (Urdu)  
Total examples: 2290  

### StrategyQA Zero-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2273 (99.26%)
- Accuracy overall: 55.63%
- Accuracy answered: 56.05%
- Output file: outputs/strategyqa/qwen2.5_7b/strategyqa_zero_shot_qwen2.5_7b.jsonl

### StrategyQA 3-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2288 (99.91%)
- Accuracy overall: 70.17%
- Accuracy answered: 70.24%
- Output file: outputs/strategyqa/qwen2.5_7b/strategyqa_three_shot_qwen2.5_7b.jsonl
> Note: Unlike BoolQ and CSQA, StrategyQA benefits substantially from few-shot
> prompting. The 3-shot setting provides concrete reasoning patterns that help
> the model combine multiple facts, leading to a large accuracy jump
> (55.63% → 70.17%) over zero-shot.

### StrategyQA Chain-of-Thought (CoT)
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 56.11%
- Accuracy answered: 56.11%
- Output file: outputs/strategyqa/qwen2.5_7b/strategyqa_cot_qwen2.5_7b.jsonl

> Note: CoT slightly improves over the zero-shot setting (55.63% → 56.11%) but
> is still far below the 3-shot accuracy (70.17%), indicating that for
> StrategyQA-Urdu, concrete few-shot demonstrations are more beneficial than
> unconstrained reasoning-only prompts.

## StrategyQA – Urdu – Qwen2.5-14B-Instruct (AWQ)

Model: Qwen2.5-14B-Instruct-AWQ  
Dataset: StrategyQA (Urdu)  
Total examples: 2290  

### StrategyQA Zero-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 73.19%
- Accuracy answered: 73.19%
- Output file: outputs/strategyqa/qwen2.5_14b_awq/strategyqa_zero_shot_qwen2.5_14b_awq.jsonl

### StrategyQA 3-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 75.11%
- Accuracy answered: 75.11%
- Output file: outputs/strategyqa/qwen2.5_14b_awq/strategyqa_three_shot_qwen2.5_14b_awq.jsonl 
> Note: Accuracy is slightly higher than zero-shot, indicating that the model benefits from in-context examples for Urdu reasoning.

### StrategyQA CoT
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 76.72%
- Accuracy answered: 76.72%
- Output file: outputs/strategyqa/qwen2.5_14b_awq/strategyqa_cot_qwen2.5_14b_awq.jsonl  
> Note: CoT yields a modest but consistent improvement over zero-shot and 3-shot,
> indicating that explicit reasoning steps help the model better structure
> Urdu commonsense inference.

## StrategyQA – Urdu – Meta-Llama-3.1-8B-Instruct

Model: Meta-Llama-3.1-8B-Instruct  
Dataset: StrategyQA (Urdu)  
Total examples: 2290  

### StrategyQA Zero-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 57.55%
- Accuracy answered: 57.55%
- Output file: outputs/strategyqa/meta_llama3.1_8b/strategyqa_zero_shot_meta_llama3.1_8b.jsonl

> **Note (code changes):** For Meta-Llama-3.1-8B, AWQ quantization settings were removed and the vLLM configuration was updated. We also increased `max_tokens` and improved Urdu normalization to handle truncated outputs like `ہا → ہاں` and `نہ → نہیں`.

### StrategyQA 3-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 78.43%
- Accuracy answered: 78.43%
- Output file: outputs/strategyqa/meta_llama3.1_8b/strategyqa_three_shot_meta_llama3.1_8b.jsonl

> **Note:** For Meta-Llama-3.1-8B, we removed AWQ quantization settings and updated the vLLM config. We also increased `max_tokens` (4 → 16) and improved Urdu normalization to handle truncated outputs like `ہا → ہاں` and `نہ → نہیں`, ensuring fair “Answered” scoring. With these fixes, Meta-Llama-3.1-8B performs better than the Qwen baseline models on StrategyQA 3-shot in our current runs. Next, we should run Qwen with the same updated code/settings to confirm the comparison is not affected by evaluation differences.

### StrategyQA CoT
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 78.17%
- Accuracy answered: 78.17%
- Output file: outputs/strategyqa/meta_llama3.1_8b/strategyqa_cot_meta_llama3.1_8b.jsonl

> **Note (findings):** Using Meta-Llama-3.1-8B-Instruct on StrategyQA (Urdu) with CoT, the model achieved **78.17%** accuracy with **100% answered**. We removed AWQ-specific settings, updated the vLLM configuration for this model, increased `max_tokens`, and improved Urdu normalization to correctly handle truncated outputs like `ہا → ہاں` and `نہ → نہیں`. With these fixes, Meta-Llama-3.1-8B currently performs **better than all Qwen baseline models** on StrategyQA CoT in our runs. Next, we should run the **same updated code/settings on Qwen** to confirm that the comparison is not influenced by evaluation differences.

## StrategyQA – Urdu – DeepSeek-R1-Distill-Qwen-7B

Model: DeepSeek-R1-Distill-Qwen-7B  
Dataset: StrategyQA (Urdu)  
Total examples: 2290  

### StrategyQA Zero-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 55.24%
- Accuracy answered: 55.24%
- Output file: outputs/strategyqa/deepseek_r1_distill_qwen_7b/strategyqa_zero_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied (no stop, increased max_tokens, answer extraction). Accuracy (55.24%) comparable to Qwen-1.5B (53.28%) but below Qwen-7B (55.63%) and Qwen-14B (73.19%).

### StrategyQA 3-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2272 (99.21%)
- Accuracy overall: 51.66%
- Accuracy answered: 52.07%
- Output file: outputs/strategyqa/deepseek_r1_distill_qwen_7b/strategyqa_three_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** Same DeepSeek-specific fixes applied. 3-shot accuracy (51.66%) slightly lower than zero-shot (55.24%), unlike Qwen-7B which saw large gains (55.63% → 70.17%) from few-shot examples.

### StrategyQA Chain-of-Thought (CoT)
- Used items: 2290
- Answered (ہاں/نہیں): 2288 (99.91%)
- Accuracy overall: 47.51%
- Accuracy answered: 47.55%
- Output file: outputs/strategyqa/deepseek_r1_distill_qwen_7b/strategyqa_cot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** Same DeepSeek-specific fixes applied. CoT accuracy (47.51%) lower than zero-shot (55.24%) and 3-shot (51.66%), suggesting reasoning prompts may hurt performance for this model on StrategyQA.

---

## GSM8K – Urdu – Qwen2.5-1.5B-Instruct

Model: Qwen2.5-1.5B-Instruct  
Dataset: GSM8K (Urdu)  
Total examples: 700  

### GSM8K Zero-shot
- Used items:       700
- Answered (numeric): 700 (100.00%)
- Accuracy overall: 11.29%
- Accuracy answered: 11.29%
- Output file: outputs/gsm8k/qwen2.5_1.5b/gsm8k_zero_shot_qwen2.5_1.5b.jsonl

### GSM8K 3-shot
- Used items:       700
- Answered (numeric): 700 (100.00%)
- Accuracy overall: 5.43%
- Accuracy answered: 5.43%
- Output file: outputs/gsm8k/qwen2.5_1.5b/gsm8k_three_shot_qwen2.5_1.5b.jsonl

### GSM8K Chain-of-Thought (CoT)
- Used items:       700
- Answered (numeric): 685 (97.86%)
- Accuracy overall: 3.71%
- Accuracy answered: 3.80%
- Output file: outputs/gsm8k/qwen2.5_1.5b/gsm8k_cot_qwen2.5_1.5b.jsonl

> Note: In GSM8K-Urdu, zero-shot prompting achieves modest accuracy (~11%), 
> while 3-shot and CoT prompting both *reduce* performance (5.4% and 3.7%, respectively), 
> despite near-perfect numeric coverage. This suggests that, for a 1.5B model, 
> adding Urdu demonstrations or chain-of-thought instructions introduces additional complexity 
> without improving the underlying arithmetic reasoning.

## GSM8K – Urdu – Qwen2.5-7B-Instruct

Model: Qwen2.5-7B-Instruct  
Dataset: GSM8K (Urdu)  
Total examples: 700  

### GSM8K Zero-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   17.14%
- Accuracy answered:  17.14%
- Output file:        outputs/gsm8k/qwen2.5_7b/gsm8k_zero_shot_qwen2.5_7b.jsonl  

> **Prompt:** Urdu problem statement, instruction to solve the question,  
> and a final answer requested in the format `#### N` (numeric only).  
> **Evaluation:** We parse the gold and predicted answers using the same
> numeric extractor, prioritizing the `#### N` pattern and falling back to
> the last numeric token in the text.

### GSM8K 3-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   5.00%
- Accuracy answered:  5.00%
- Output file:        outputs/gsm8k/qwen2.5_7b/gsm8k_three_shot_qwen2.5_7b.jsonl  

> **Prompt:** Three Urdu example problems with numeric-only answers, followed by
> the test problem, again asking for a single numeric answer.  
> **Notes:** Despite perfect numeric coverage, accuracy is very low (5%), suggesting
> that few-shot direct-answer prompting without explicit reasoning does not help
> 7B Qwen on GSM8K-style multi-step arithmetic in Urdu.

### GSM8K Chain-of-Thought (CoT)
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   28.29%
- Accuracy answered:  28.29%
- Output file:        outputs/gsm8k/qwen2.5_7b/gsm8k_cot_qwen2.5_7b.jsonl  

> **Prompt:** Urdu instruction to “solve step by step” and *then* produce a final
> numeric answer on a new line in the format `#### N`. The model is free to reason
> in Urdu before the final answer line.  
> **Notes:** CoT substantially improves performance over both zero-shot (17.1%) and
> 3-shot (5.0%) for the 7B model. This contrasts with the 1.5B model, where CoT
> reduced accuracy (from 11.3% → 3.8%). This suggests an emergent benefit of
> chain-of-thought prompting only at larger model scale, even under Urdu translation.

## GSM8K – Urdu – Qwen2.5-14B-Instruct (AWQ)

Model: Qwen2.5-14B-Instruct-AWQ  
Dataset: GSM8K (Urdu)  
Total examples: 700  

### GSM8K Zero-shot
- Used items:         700
- Answered (numeric): 699 (99.86%)
- Accuracy overall:   25.14%
- Accuracy answered:  25.18%
- Output file:        outputs/gsm8k/qwen2.5_14b_awq/gsm8k_zero_shot_qwen2.5_14b_awq.jsonl

### GSM8K 3-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   8.86%
- Accuracy answered:  8.86%
- Output file:        outputs/gsm8k/qwen2.5_14b_awq/gsm8k_three_shot_qwen2.5_14b_awq.jsonl  

> **Note:** Accuracy improves over the 7B model (5.00% → 8.86%), indicating that larger model capacity helps few-shot arithmetic reasoning, though performance remains far below CoT-based prompting for GSM8K-style problems.

### GSM8K Chain-of-Thought (CoT)
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   39.43%
- Accuracy answered:  39.43%
- Output file:        outputs/gsm8k/qwen2.5_14b_awq/gsm8k_cot_qwen2.5_14b_awq.jsonl  

> **Note:** Compared to earlier GSM8K-Urdu runs with smaller Qwen models and non-CoT
> prompting, this 39.43% CoT accuracy is a substantial jump and quite strong for a low-resource language setting.

## GSM8K – Urdu – Meta-Llama-3.1-8B-Instruct

Model: Meta-Llama-3.1-8B-Instruct  
Dataset: GSM8K (Urdu)  
Total examples: 700  

### GSM8K Zero-shot
- Used items: 700
- Answered (numeric): 697 (99.57%)
- Accuracy overall: 17.29%
- Accuracy answered: 17.36%
- Output file: outputs/gsm8k/meta_llama3.1_8b/gsm8k_zero_shot_meta_llama3.1_8b.jsonl

> **Note:** For Meta-Llama-3.1-8B, we removed AWQ quantization settings and used the updated vLLM configuration for this model. The model performs better than the Qwen 1.5B and 7B baselines on GSM8K Urdu zero-shot, but remains below Qwen 14B. Later, we should run Qwen with the same updated code/settings to double-check that the comparison is not affected by evaluation differences.

### GSM8K 3-shot
- Used items: 700
- Answered (numeric): 700 (100.00%)
- Accuracy overall: 12.57%
- Accuracy answered: 12.57%
- Output file: outputs/gsm8k/meta_llama3.1_8b/gsm8k_three_shot_meta_llama3.1_8b.jsonl

> **Note:** We updated the **3-shot prompt format** to force the final answer as `#### N` (and updated the demo examples to the same format) because Meta-Llama-3.1-8B was continuing the “مسئلہ/جواب” pattern and generating extra problems after answering, which broke evaluation when the parser picked unrelated trailing numbers (e.g., frequent `60`). In the code, we changed numeric extraction to **accept only the `#### N` pattern** (removed the “last number” fallback), and used the Meta-Llama-3.1-8B vLLM settings (no AWQ quantization; safe config like `max_model_len=3072`, `gpu_memory_utilization=0.75`, `max_num_seqs=1`) to avoid OOM/Slurm kills.

### GSM8K CoT
- Used items: 700
- Answered (numeric): 700 (100.00%)
- Accuracy overall: 11.00%
- Accuracy answered: 11.00%
- Output file: outputs/gsm8k/meta_llama3.1_8b/gsm8k_cot_meta_llama3.1_8b.jsonl

> **Note:** Meta-Llama-3.1-8B achieves full numeric answer rate on GSM8K-CoT, but accuracy remains low (11.00%), indicating the model often produces a numeric output yet fails the final computation. We used the same evaluation/parsing logic as before and only updated the run for this model (no AWQ quantization; vLLM set to safe values like `max_model_len=3072`, `gpu_memory_utilization=0.75`, `max_num_seqs=1`). For a fair comparison, we can run the exact same CoT prompt + code on Qwen models and compare directly.

## GSM8K – Urdu – DeepSeek-R1-Distill-Qwen-7B

Model: DeepSeek-R1-Distill-Qwen-7B  
Dataset: GSM8K (Urdu)  
Total examples: 700  

### GSM8K Zero-shot
- Used items: 700
- Answered (numeric): 677 (96.71%)
- Accuracy overall: 28.71%
- Accuracy answered: 29.69%
- Output file: outputs/gsm8k/deepseek_r1_distill_qwen_7b/gsm8k_zero_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied (no stop, increased max_tokens=1024, specialized number extraction). Performance (28.71%) better than Qwen-7B (17.14%) and approaches Qwen-14B (25.14%) on GSM8K Urdu zero-shot.

### GSM8K 3-shot
- Used items: 700
- Answered (numeric): 379 (54.14%)
- Accuracy overall: 9.14%
- Accuracy answered: 16.89%
- Output file: outputs/gsm8k/deepseek_r1_distill_qwen_7b/gsm8k_three_shot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied. 3-shot performance (9.14%) lower than zero-shot (28.71%), with only 54% answer rate, suggesting few-shot examples don't help and may hinder math reasoning for this model.

### GSM8K Chain-of-Thought (CoT)
- Used items: 700
- Answered (numeric): 681 (97.29%)
- Accuracy overall: 33.43%
- Accuracy answered: 34.36%
- Output file: outputs/gsm8k/deepseek_r1_distill_qwen_7b/gsm8k_cot_deepseek_r1_distill_qwen_7b.jsonl

> **Note:** DeepSeek-specific fixes applied. CoT achieves best performance (33.43%), significantly outperforming zero-shot (28.71%) and 3-shot (9.14%), with high answer rate (97.29%).
```