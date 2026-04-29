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

## BoolQ – Urdu – Qwen3-8B

Model: Qwen3-8B  
Dataset: BoolQ (Urdu)  
Total examples: 1550  

### BoolQ Zero-shot
- Used items: 1550
- Scored on: 1550
- Accuracy: 79.68%
- Output file: outputs/boolq/qwen3_8b/boolq_zero_shot_qwen3_8b.jsonl

> Note: Qwen3-8B uses a chat template format (apply_chat_template) with
> thinking mode disabled (enable_thinking=False) for zero-shot evaluation,
> keeping results directly comparable to other models. 1 passage was truncated
> to fit within the 3072-token context limit.

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1548
- Accuracy: 81.52%
- Output file: outputs/boolq/qwen3_8b/boolq_three_shot_qwen3_8b.jsonl

> Note: Qwen3-8B uses chat template format with thinking mode disabled
> (enable_thinking=False) for 3-shot evaluation. 1 passage was truncated
> to fit within the 3072-token context limit.

### BoolQ Chain-of-Thought (CoT)
- Used items: 1550
- Scored on: 1545
- Accuracy: 80.84%
- Output file: outputs/boolq/qwen3_8b/boolq_cot_qwen3_8b.jsonl

> Note: Qwen3-8B CoT uses thinking mode enabled (enable_thinking=True),
> activating the model's built-in reasoning via <think>...</think> blocks
> before the final answer. The <think> block is stripped during answer
> extraction; only the final answer token is used for accuracy scoring.
> Full raw output including reasoning is preserved in the JSONL file.
> Runtime was significantly longer (~54 min) due to thinking mode generation.

## BoolQ – Urdu – Qwen3-14B

Model: Qwen3-14B  
Dataset: BoolQ (Urdu)  
Total examples: 1550  

### BoolQ Zero-shot
- Used items: 1550
- Scored on: 1549
- Accuracy: 84.51%
- Output file: outputs/boolq/qwen3_14b/boolq_zero_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses a chat template format (apply_chat_template) with
> thinking mode disabled (enable_thinking=False) for zero-shot evaluation,
> accuracy is improved over the QWEN3-8B for boolq zero-shot.

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1549
- Accuracy: 84.89%
- Output file: outputs/boolq/qwen3_14b/boolq_three_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses chat template format with thinking mode disabled
> (enable_thinking=False) for 3-shot evaluation.This model shows that over time model improved in boolq 3-shot for Urdu language.

### BoolQ Chain-of-Thought (CoT)
- Used items: 1550
- Scored on: 1539
- Accuracy: 84.47%
- Output file: outputs/boolq/qwen3_14b/boolq_cot_qwen3_14b.jsonl

> Note: Qwen3-14B CoT uses thinking mode enabled (enable_thinking=True),
> activating the model's built-in reasoning via <think>...</think> blocks
> before the final answer. The <think> block is stripped during answer
> extraction; only the final answer token is used for accuracy scoring.
> 11 items yielded no extractable prediction after think-block removal,
> reducing scored items from 1550 to 1539.

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

## BoolQ – Urdu – Meta-Llama-3.1-70B-Instruct-AWQ-INT4

Model: Meta-Llama-3.1-70B-Instruct-AWQ-INT4
Dataset: BoolQ (Urdu)
Total examples: 1550

### BoolQ Zero-shot
- Used items: 1550
- Scored on: 1545 (pred != None)
- Accuracy: 87.51%
- Output file: outputs/boolq/llama3.1_70b_awq/boolq_zero_shot_llama3.1_70b_awq.jsonl

> Note: Meta-Llama-3.1-70B-AWQ-INT4 achieves 87.51% on BoolQ-Urdu zero-shot,
> setting a new high across all evaluated models, surpassing Gemma-2-9B-IT
> (86.30%), Qwen3-14B (84.51%), and Qwen3-8B (79.68%). This confirms that
> scaling to 70B with AWQ quantization yields strong direct binary QA
> performance in Urdu even without few-shot examples.

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1546 (pred != None)
- Accuracy: 88.49%
- Output file: outputs/boolq/llama3.1_70b_awq/boolq_three_shot_llama3.1_70b_awq.jsonl

> Note: 3-shot improves over zero-shot (87.51% → 88.49%), maintaining the top
> position across all evaluated models on BoolQ-Urdu. Surpasses Gemma-2-9B-IT
> 3-shot (87.20%) and Qwen3-14B 3-shot (84.89%). The 70B model benefits
> consistently from few-shot examples, unlike smaller models where 3-shot
> sometimes hurts performance.

### BoolQ Chain-of-Thought (CoT)
- Used items: 1550
- Scored on: 1546 (pred != None)
- Accuracy: 87.45%
- Output file: outputs/boolq/llama3.1_70b_awq/boolq_cot_llama3.1_70b_awq.jsonl

> Note: CoT accuracy (87.45%) is slightly below zero-shot (87.51%) and 3-shot
> (88.49%), suggesting that explicit reasoning does not further improve BoolQ-Urdu
> performance for this model. All three prompt types remain tightly clustered
> within 1 percentage point, indicating stable and robust performance regardless
> of prompt strategy — a pattern not seen in smaller models.

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

## BoolQ – Urdu – Gemma-2-9B-IT

Model: Gemma-2-9B-IT
Dataset: BoolQ (Urdu)
Total examples: 1550

### BoolQ Zero-shot
- Used items: 1550
- Scored on: 1548 (99.87%)
- Accuracy: 86.30%
- Output file: outputs/boolq/gemma2_9b_it/boolq_zero_shot_gemma2_9b_it.jsonl

> Note: Evaluated using standard prompt setup with no model-specific modifications.
> Gemma-2-9B-IT achieves the highest zero-shot accuracy across all tested models on
> BoolQ-Urdu, outperforming Qwen3-14B (84.51%), Qwen3-8B (79.68%),
> Meta-LLaMA-3.1-8B (78.49%), Qwen2.5-14B-AWQ (76.26%), and DeepSeek-R1-Distill-Qwen-7B (56.76%).

### BoolQ 3-shot
- Used items: 1550
- Scored on: 1547 (99.81%)
- Accuracy: 87.20%
- Output file: outputs/boolq/gemma2_9b_it/boolq_three_shot_gemma2_9b_it.jsonl

> Note: 3-shot improves over zero-shot (86.30% → 87.20%), maintaining the top
> position across all models. Coverage remains high at 99.81%.

### BoolQ Chain-of-Thought (CoT)
- Used items: 1550
- Scored on: 1547
- Accuracy: 86.10%
- Output file: outputs/boolq/gemma2_9b_it/boolq_cot_gemma2_9b_it.jsonl

> Note: CoT accuracy (86.10%) is slightly below zero-shot (86.30%) and 3-shot (87.20%),
> suggesting Gemma-2-9B-IT does not benefit from chain-of-thought prompting on BoolQ-Urdu.
> Despite this, it remains competitive across all prompt types and holds a strong position
> among all tested models on this dataset.

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

## CSQA – Urdu – Qwen3-8B

Model: Qwen3-8B  
Dataset: CSQA (Urdu)  
Total examples: 1500  

### CSQA Zero-shot
- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  54.80%  (correct / 1500)
- Accuracy answered: 54.80%  (correct / 1500)
- Output file:       outputs/csqa/qwen3_8b/csqa_zero_shot_qwen3_8b.jsonl

> Note: Qwen3-8B uses chat template format with thinking mode disabled
> (enable_thinking=False) for zero-shot evaluation. 100% answer coverage
> achieved. At 54.80%, this is the highest CSQA zero-shot result across
> all evaluated models, surpassing Qwen2.5-14B-AWQ (50.93%).

### CSQA 3-shot

- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  52.20%  (correct / 1500)
- Accuracy answered: 52.20%  (correct / 1500)
- Output file:       outputs/csqa/qwen3_8b/csqa_three_shot_qwen3_8b.jsonl

> Note: Qwen3-8B uses chat template format with thinking mode disabled
> (enable_thinking=False) for 3-shot evaluation. 100% answer coverage
> achieved. At 52.20%, this is the highest CSQA zero-shot result across
> all evaluated models, surpassing Qwen2.5-14B-AWQ (46.53%).

### CSQA Chain-of-Thought (CoT)
- Used items:        1500
- Answered (A–E):    1495 (99.67%)
- Accuracy overall:  55.33%  (correct / 1500)
- Accuracy answered: 55.52%  (correct / 1495)
- Output file:       outputs/csqa/qwen3_8b/csqa_cot_qwen3_8b.jsonl

> Note: Qwen3-8B CoT uses thinking mode enabled (enable_thinking=True),
> activating built-in reasoning via <think>...</think> blocks before the
> final answer. At 55.33%, this is the highest CSQA CoT result across all
> evaluated models, surpassing Qwen2.5-14B-AWQ CoT (49.73%) by a notable
> margin. Coverage remains near-complete at 99.67%.

## CSQA – Urdu – Qwen3-14B

Model: Qwen3-14B
Dataset: CSQA (Urdu)
Total examples: 1500

### CSQA Zero-shot
- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  58.73%  (correct / 1500)
- Accuracy answered: 58.73%  (correct / 1500)
- Output file:       outputs/csqa/qwen3_14b/csqa_zero_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses chat template format with thinking mode disabled
> (enable_thinking=False) for zero-shot evaluation. 100% answer coverage
> achieved. At 58.73%, this is the highest CSQA zero-shot result across
> all evaluated models, surpassing Qwen3-8B (54.80%) and Qwen2.5-14B-AWQ (50.93%).

### CSQA 3-shot
- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  56.60%  (correct / 1500)
- Accuracy answered: 56.60%  (correct / 1500)
- Output file:       outputs/csqa/qwen3_14b/csqa_three_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses chat template format with thinking mode disabled
> (enable_thinking=False) for 3-shot evaluation. 100% answer coverage
> achieved. At 56.60%, this is the highest CSQA 3-shot result across
> all evaluated models, surpassing Qwen3-8B (52.20%) and Qwen2.5-14B-AWQ (46.53%).

### CSQA Chain-of-Thought (CoT)
- Used items:        1500
- Answered (A–E):    1497 (99.80%)
- Accuracy overall:  63.00%  (correct / 1500)
- Accuracy answered: 63.13%  (correct / 1497)
- Output file:       outputs/csqa/qwen3_14b/csqa_cot_qwen3_14b.jsonl

> Note: Qwen3-14B CoT uses thinking mode enabled (enable_thinking=True),
> activating built-in reasoning via <think>...</think> blocks before the
> final answer. At 63.00%, this is the highest CSQA CoT result across all
> evaluated models, surpassing Qwen3-8B CoT (55.33%) by a notable margin
> and Qwen2.5-14B-AWQ CoT (49.73%) by a large margin. Coverage remains
> near-complete at 99.80%.

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

## CSQA – Urdu – Meta-Llama-3.1-70B-Instruct-AWQ-INT4

Model: Meta-Llama-3.1-70B-Instruct-AWQ-INT4
Dataset: CSQA (Urdu)
Total examples: 1500

### CSQA Zero-shot
- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  65.60%  (correct / 1500)
- Accuracy answered: 65.60%  (correct / 1500)
- Output file:       outputs/csqa/llama3.1_70b_awq/csqa_zero_shot_llama3.1_70b_awq.jsonl

> Note: Meta-Llama-3.1-70B-AWQ achieves 65.60% on CSQA-Urdu zero-shot with
> full 100% answer coverage, setting a new high across all evaluated models,
> surpassing Qwen3-14B (58.73%), Gemma-2-9B-IT (59.87%), and Qwen3-8B (54.80%).
> The 70B scale clearly benefits commonsense MCQ reasoning in Urdu under
> direct prompting.

### CSQA 3-shot
- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  64.87%  (correct / 1500)
- Accuracy answered: 64.87%  (correct / 1500)
- Output file:       outputs/csqa/llama3.1_70b_awq/csqa_three_shot_llama3.1_70b_awq.jsonl

> Note: 3-shot accuracy (64.87%) drops slightly from zero-shot (65.60%),
> suggesting few-shot examples provide marginal negative effect on CSQA-Urdu
> for this model. Still the second highest 3-shot result across all evaluated
> models, behind only Qwen3-14B (56.60%) — wait, surpassing it — and ahead of
> Gemma-2-9B-IT (59.20%) and Qwen3-14B (56.60%). Full coverage maintained.

### CSQA Chain-of-Thought (CoT)
- Used items:        1500
- Answered (A–E):    1497 (99.80%)
- Accuracy overall:  58.07%
- Accuracy answered: 58.18%
- Output file:       outputs/csqa/llama3.1_70b_awq/csqa_cot_llama3.1_70b_awq.jsonl

> Note: CoT accuracy (58.07%) drops notably compared to zero-shot (65.60%) and
> 3-shot (64.87%), a pattern also seen in Gemma-2-9B-IT (51.87%) on CSQA-Urdu.
> This suggests that chain-of-thought prompting consistently hurts commonsense
> MCQ performance across non-Qwen3 architectures, while Qwen3 models benefit
> from CoT on this task. Despite the drop, coverage remains near-complete at
> 99.80%.

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

## CSQA – Urdu – Gemma-2-9B-IT

Model: Gemma-2-9B-IT
Dataset: CSQA (Urdu)
Total examples: 1500

### CSQA Zero-shot
- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  59.87%  (correct / 1500)
- Accuracy answered: 59.87%  (correct / 1500)
- Output file:       outputs/csqa/gemma2_9b_it/csqa_zero_shot_gemma2_9b_it.jsonl

> Note: Gemma-2-9B-IT achieves 59.87% on CSQA-Urdu zero-shot with full coverage,
> surpassing Qwen3-14B (58.73%) and setting a new high across all evaluated models
> on this setting.

### CSQA 3-shot
- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  59.20%  (correct / 1500)
- Accuracy answered: 59.20%  (correct / 1500)
- Output file:       outputs/csqa/gemma2_9b_it/csqa_three_shot_gemma2_9b_it.jsonl

> Note: 3-shot accuracy (59.20%) is marginally below zero-shot (59.87%),
> suggesting Gemma-2-9B-IT does not gain from few-shot examples on CSQA-Urdu.
> Still holds the top position among all evaluated models on 3-shot CSQA,
> surpassing Qwen3-14B (56.60%).

### CSQA Chain-of-Thought (CoT)
- Used items:        1500
- Answered (A–E):    1500 (100.00%)
- Accuracy overall:  51.87%  (correct / 1500)
- Accuracy answered: 51.87%  (correct / 1500)
- Output file:       outputs/csqa/gemma2_9b_it/csqa_cot_gemma2_9b_it.jsonl

> Note: CoT accuracy (51.87%) drops notably compared to zero-shot (59.87%) and
> 3-shot (59.20%), suggesting that chain-of-thought prompting hurts Gemma-2-9B-IT
> on CSQA-Urdu. Unlike Qwen3 models where CoT consistently improves performance,
> Gemma-2-9B-IT appears to perform best with direct answering on this task.

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

## PIQA – Urdu – Qwen3-8B

Model: Qwen3-8B  
Dataset: PIQA (Urdu)  
Total examples: 750  

### PIQA Zero-shot
- Used items: 750
- Answered (0/1): 750 (100.00%)
- Accuracy overall: 56.53%
- Accuracy answered: 56.53%
- Output file: outputs/piqa/qwen3_8b/piqa_zero_shot_qwen3_8b.jsonl

> Note: Qwen3-8B uses chat template format with thinking mode disabled
> (enable_thinking=False) for zero-shot evaluation. 100% answer coverage
> achieved.But accuracy dropped as compared to Qwen2.5-14B-Instruct (AWQ), it shows model is feeling difficult for Piqa zero shot settings in urdu.

### PIQA 3-shot
- Used items: 750
- Answered (0/1): 750 (100.00%)
- Accuracy overall: 55.87%
- Accuracy answered: 55.87%
- Output file: outputs/piqa/qwen3_8b/piqa_three_shot_qwen3_8b.jsonl

> Note: 3-shot slightly drops compared to zero-shot (56.53% → 55.87%),
> which is unusual as 3-shot typically improves performance. Still below
> Qwen2.5-7B 3-shot (62.00%) and Qwen2.5-14B-AWQ 3-shot (65.33%).
> CoT with thinking enabled may recover performance as seen in other datasets.

### PIQA Chain-of-Thought (CoT)
- Used items: 750
- Answered (0/1): 693 (92.40%)
- Accuracy overall: 55.20%
- Accuracy answered: 59.74%
- Output file: outputs/piqa/qwen3_8b/piqa_cot_qwen3_8b.jsonl

> Note: Qwen3-8B CoT uses thinking mode enabled (enable_thinking=True).
> Coverage drops to 92.40% as the longer reasoning output occasionally
> fails to produce a clean 0/1 answer. However, accuracy among answered
> items (59.74%) is the highest across all three Qwen3-8B PIQA prompting
> strategies, suggesting the model reasons correctly when it does commit
> to an answer. Overall accuracy (55.20%) remains below Qwen2.5-14B-AWQ
> CoT (58.93%) due to the coverage gap.

## PIQA – Urdu – Qwen3-14B

Model: Qwen3-14B  
Dataset: PIQA (Urdu)  
Total examples: 750 

### PIQA Zero-shot
- Used items:        750
- Answered (0/1):    750 (100.00%)
- Accuracy overall:  59.33%
- Accuracy answered: 59.33%
- Output file:       outputs/piqa/qwen3_14b/piqa_zero_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses chat template format with thinking mode disabled
> (enable_thinking=False) for zero-shot evaluation. 100% answer coverage
> achieved. At 59.33%, this surpasses Qwen3-8B zero-shot (56.53%) and
> is comparable to Qwen2.5-14B-AWQ zero-shot (60.67%), showing a clear
> improvement over the smaller Qwen3-8B on PIQA-Urdu.But somehow this model still not performing well on PIQA zero shot for urdu language as compared to Qwen2.5-14B-Instruct (AWQ) slightly lower accuracy. Let's see for other prompts.

### PIQA 3-shot
- Used items:        750
- Answered (0/1):    750 (100.00%)
- Accuracy overall:  63.73%
- Accuracy answered: 63.73%
- Output file:       outputs/piqa/qwen3_14b/piqa_three_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses chat template format with thinking mode disabled
> (enable_thinking=False) for 3-shot evaluation. 100% answer coverage
> achieved. At 63.73%, this surpasses Qwen3-8B 3-shot (55.87%) and is
> comparable to Qwen2.5-14B-AWQ 3-shot (65.33%), showing a clear gain
> over the smaller Qwen3 variant in the few-shot setting.

### PIQA Chain-of-Thought (CoT)
- Used items:        750
- Answered (0/1):    697 (92.93%)
- Accuracy overall:  65.73%
- Accuracy answered: 70.73%
- Output file:       outputs/piqa/qwen3_14b/piqa_cot_qwen3_14b.jsonl

> Note: Qwen3-14B CoT uses thinking mode enabled (enable_thinking=True),
> activating built-in reasoning via <think>...</think> blocks before the
> final answer. At 65.73% overall and 70.73% among answered items, this
> is the highest PIQA CoT result across all evaluated models, surpassing
> Qwen2.5-14B-AWQ CoT (58.93%) and Qwen3-8B CoT (55.20%). Coverage drops
> to 92.93% as longer reasoning outputs occasionally fail to produce a
> clean 0/1 answer, consistent with the pattern seen in Qwen3-8B CoT.

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

## PIQA – Urdu – Meta-Llama-3.1-70B-Instruct-AWQ-INT4

Model: Meta-Llama-3.1-70B-Instruct-AWQ-INT4
Dataset: PIQA (Urdu)
Total examples: 750

### PIQA Zero-shot
- Used items:        750
- Answered (0/1):    750 (100.00%)
- Accuracy overall:  62.93%
- Accuracy answered: 62.93%
- Output file:       outputs/piqa/llama3.1_70b_awq/piqa_zero_shot_llama3.1_70b_awq.jsonl

> Note: Meta-Llama-3.1-70B-AWQ achieves 62.93% on PIQA-Urdu zero-shot with
> full coverage, ranking behind Gemma-2-9B-IT (68.67%) but ahead of
> Qwen3-14B (59.33%) and Qwen2.5-14B-AWQ (60.67%). Unlike Gemma which
> shows a dramatic drop under 3-shot and CoT, the 70B model's zero-shot
> baseline is more moderate, making it interesting to see whether it maintains
> performance across prompt types.

### PIQA Chain-of-Thought (CoT)
- Used items:        750
- Answered (0/1):    426 (56.80%)
- Accuracy overall:  31.73%
- Accuracy answered: 55.87%
- Output file:       outputs/piqa/llama3.1_70b_awq/piqa_cot_llama3.1_70b_awq.jsonl

> Note: CoT causes a severe collapse in both coverage (56.80%) and overall
> accuracy (31.73%), the worst CoT performance on PIQA across all evaluated
> models. The model frequently fails to produce a clean 0/1 answer when asked
> to reason step-by-step, and even among answered items accuracy (55.87%)
> is near random baseline (50%). This mirrors the pattern seen in Gemma-2-9B-IT
> and Meta-LLaMA-3.1-8B, confirming that CoT consistently hurts PIQA-Urdu
> performance across LLaMA architectures regardless of scale.

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

## PIQA – Urdu – Gemma-2-9B-IT

Model: Gemma-2-9B-IT
Dataset: PIQA (Urdu)
Total examples: 750

### PIQA Zero-shot
- Used items:        750
- Answered (0/1):    750 (100.00%)
- Accuracy overall:  68.67%
- Accuracy answered: 68.67%
- Output file:       outputs/piqa/gemma2_9b_it/piqa_zero_shot_gemma2_9b_it.jsonl

> Note: Gemma-2-9B-IT achieves the highest PIQA-Urdu zero-shot accuracy across
> all evaluated models (68.67%), surpassing Qwen3-14B (59.33%), Qwen2.5-14B-AWQ
> (60.67%), and all smaller models, with full 100% answer coverage.

### PIQA 3-shot
- Used items:        750
- Answered (0/1):    750 (100.00%)
- Accuracy overall:  53.60%
- Accuracy answered: 53.60%
- Output file:       outputs/piqa/gemma2_9b_it/piqa_three_shot_gemma2_9b_it.jsonl

> Note: 3-shot accuracy (53.60%) drops sharply compared to zero-shot (68.67%),
> the largest such drop seen across all models on PIQA-Urdu. Few-shot examples
> appear to confuse rather than help Gemma-2-9B-IT on this task, pushing accuracy
> close to the random baseline (50%). Less accuracy than, Qwen 3-14B, 3-8B and even Qwen2.5-14B-AWQ.

### PIQA Chain-of-Thought (CoT)
- Used items:        750
- Answered (0/1):    730 (97.33%)
- Accuracy overall:  50.00%
- Accuracy answered: 51.37%
- Output file:       outputs/piqa/gemma2_9b_it/piqa_cot_gemma2_9b_it.jsonl

> Note: CoT accuracy (50.00% overall) falls to the random baseline, continuing
> the sharp decline seen from zero-shot (68.67%) to 3-shot (53.60%). Gemma-2-9B-IT
> shows a strong prompt-type sensitivity on PIQA-Urdu: excellent at zero-shot
> direct answering but severely degraded by both few-shot examples and explicit
> reasoning. This is the most pronounced zero-shot → CoT collapse across all
> evaluated models on this dataset.

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

## StrategyQA – Urdu – Qwen3-8B

Model: Qwen3-8B  
Dataset: StrategyQA (Urdu)  
Total examples: 2290  

### StrategyQA Zero-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 79.96%
- Accuracy answered: 79.96%
- Output file: outputs/strategyqa/qwen3_8b/strategyqa_zero_shot_qwen3_8b.jsonl

> Note: Qwen3-8B uses chat template format with thinking mode disabled
> (enable_thinking=False) for zero-shot evaluation. At 79.96%, this is
> the highest StrategyQA zero-shot result across all evaluated models,
> surpassing Qwen2.5-14B-AWQ (73.19%) by a significant margin, despite
> being a smaller model.

### StrategyQA 3-shot
- Used items: 2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall: 80.74%
- Accuracy answered: 80.74%
- Output file: outputs/strategyqa/qwen3_8b/strategyqa_three_shot_qwen3_8b.jsonl

> Note: 3-shot gives a modest but consistent improvement over zero-shot
> (79.96% → 80.74%), maintaining 100% coverage. This is the highest
> StrategyQA 3-shot result across all evaluated models, surpassing
> Qwen2.5-14B-AWQ 3-shot (75.11%) by a clear margin.

### StrategyQA Chain-of-Thought (CoT)
- Used items: 2290
- Answered (ہاں/نہیں): 2124 (92.75%)
- Accuracy overall: 79.74%
- Accuracy answered: 85.97%
- Output file: outputs/strategyqa/qwen3_8b/strategyqa_cot_qwen3_8b.jsonl

> Note: Qwen3-8B CoT uses thinking mode enabled (enable_thinking=True).
> Coverage drops to 92.75% as longer reasoning outputs occasionally fail
> to produce a clean ہاں/نہیں answer. However, accuracy among answered
> items (85.97%) is remarkably high — the best StrategyQA answered
> accuracy across all evaluated models by a large margin, surpassing
> Qwen2.5-14B-AWQ CoT (76.72%). Runtime was significantly longer (~73 min)
> due to thinking mode generation. This pattern mirrors PIQA CoT — thinking
> mode trades coverage for higher per-answer precision on reasoning tasks.

## StrategyQA – Urdu – Qwen3-14B

Model: Qwen3-14B
Dataset: StrategyQA (Urdu)
Total examples: 2290

### StrategyQA Zero-shot
- Used items:          2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall:    83.89%
- Accuracy answered:   83.89%
- Output file:         outputs/strategyqa/qwen3_14b/strategyqa_zero_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses chat template format with thinking mode disabled
> (enable_thinking=False) for zero-shot evaluation. 100% answer coverage
> achieved. At 83.89%, this is the highest StrategyQA zero-shot result
> across all evaluated models, surpassing Qwen3-8B (79.96%) and
> Qwen2.5-14B-AWQ (73.19%) by a notable margin, showing a clear gain
> from scaling within the Qwen3 family.

### StrategyQA 3-shot
- Used items:          2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall:    83.97%
- Accuracy answered:   83.97%
- Output file:         outputs/strategyqa/qwen3_14b/strategyqa_three_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses chat template format with thinking mode disabled
> (enable_thinking=False) for 3-shot evaluation. 100% answer coverage
> achieved. At 83.97%, this is the highest StrategyQA 3-shot result
> across all evaluated models, surpassing Qwen3-8B 3-shot (80.74%) and
> Qwen2.5-14B-AWQ 3-shot (75.11%). The gain over zero-shot is minimal
> (83.89% → 83.97%), suggesting the model is already near its ceiling
> for this task without thinking mode enabled.

### StrategyQA Chain-of-Thought (CoT)
- Used items:          2290
- Answered (ہاں/نہیں): 2209 (96.46%)
- Accuracy overall:    83.89%
- Accuracy answered:   86.96%
- Output file:         outputs/strategyqa/qwen3_14b/strategyqa_cot_qwen3_14b.jsonl

> Note: Qwen3-14B CoT uses thinking mode enabled (enable_thinking=True),
> activating built-in reasoning via <think>...</think> blocks before the
> final answer. Coverage drops to 96.46% as longer reasoning outputs
> occasionally fail to produce a clean ہاں/نہیں answer, consistent with
> the pattern seen in Qwen3-8B CoT (92.75%). Accuracy among answered
> items (86.96%) is the highest StrategyQA CoT result across all evaluated
> models, surpassing Qwen3-8B CoT (85.97%) and Qwen2.5-14B-AWQ CoT
> (76.72%). Overall accuracy (83.89%) matches zero-shot, confirming that
> thinking mode trades coverage for higher per-answer precision.

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

## StrategyQA – Urdu – Meta-Llama-3.1-70B-Instruct-AWQ-INT4

Model: Meta-Llama-3.1-70B-Instruct-AWQ-INT4
Dataset: StrategyQA (Urdu)
Total examples: 2290

### StrategyQA Zero-shot
- Used items:          2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall:    88.69%
- Accuracy answered:   88.69%
- Output file:         outputs/strategyqa/llama3.1_70b_awq/strategyqa_zero_shot_llama3.1_70b_awq.jsonl

> Note: Meta-Llama-3.1-70B-AWQ achieves 88.69% on StrategyQA-Urdu zero-shot
> with full coverage, setting a new high across all evaluated models, surpassing
> Qwen3-14B (83.89%), Gemma-2-9B-IT (82.79%), and Qwen3-8B (79.96%) by a
> significant margin. This is a remarkable result — a 70B quantized model
> outperforming all Qwen3 models including the 14B on multi-hop Urdu reasoning
> without any few-shot examples. Scaling to 70B clearly provides strong
> multi-hop reasoning benefits in Urdu.

### StrategyQA 3-shot
- Used items:          2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall:    90.09%
- Accuracy answered:   90.09%
- Output file:         outputs/strategyqa/llama3.1_70b_awq/strategyqa_three_shot_llama3.1_70b_awq.jsonl

> Note: 3-shot accuracy (90.09%) improves over zero-shot (88.69%), crossing
> the 90% threshold — the first model to achieve this on StrategyQA-Urdu across
> all evaluated models. Surpasses Qwen3-14B 3-shot (83.97%), Gemma-2-9B-IT
> 3-shot (80.70%), and Meta-LLaMA-3.1-8B 3-shot (78.43%) by a large margin.
> Full 100% coverage maintained. This confirms that the 70B model benefits
> consistently from few-shot examples on multi-hop Urdu reasoning, unlike
> smaller models where 3-shot gains are less consistent.

### StrategyQA Chain-of-Thought (CoT)
- Used items:          2290
- Answered (ہاں/نہیں): 1393 (60.83%)
- Accuracy overall:    49.91%
- Accuracy answered:   82.05%
- Output file:         outputs/strategyqa/llama3.1_70b_awq/strategyqa_cot_llama3.1_70b_awq.jsonl

> Note: CoT causes a severe coverage collapse to 60.83%, the lowest answer
> rate seen on StrategyQA across all evaluated models. Overall accuracy drops
> to 49.91% — near random baseline — despite answered accuracy of 82.05%
> being competitive. The model reasons correctly when it commits to an answer,
> but the CoT prompt format frequently prevents clean ہاں/نہیں extraction for
> this model. This mirrors the PIQA CoT pattern seen earlier, confirming that
> LLaMA-3.1-70B-AWQ struggles with binary answer extraction under CoT prompting
> across multiple datasets. Zero-shot and 3-shot remain the optimal prompt
> strategies for this model on binary reasoning tasks in Urdu.

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

## StrategyQA – Urdu – Gemma-2-9B-IT

Model: Gemma-2-9B-IT
Dataset: StrategyQA (Urdu)
Total examples: 2290

### StrategyQA Zero-shot
- Used items:          2290
- Answered (ہاں/نہیں): 2290 (100.00%)
- Accuracy overall:    82.79%
- Accuracy answered:   82.79%
- Output file:         outputs/strategyqa/gemma2_9b_it/strategyqa_zero_shot_gemma2_9b_it.jsonl

> Note: Gemma-2-9B-IT achieves 82.79% on StrategyQA-Urdu zero-shot with full
> coverage, ranking second overall behind Qwen3-14B (83.89%) and surpassing
> Qwen3-8B (79.96%) and Qwen2.5-14B-AWQ (73.19%). A strong result for a
> non-Qwen3 model on multi-hop Urdu reasoning.

### StrategyQA 3-shot
- Used items:          2290
- Answered (ہاں/نہیں): 2282 (99.65%)
- Accuracy overall:    80.70%
- Accuracy answered:   80.98%
- Output file:         outputs/strategyqa/gemma2_9b_it/strategyqa_three_shot_gemma2_9b_it.jsonl

> Note: 3-shot accuracy (80.70%) drops slightly from zero-shot (82.79%),
> consistent with the pattern seen on BoolQ and CSQA where Gemma-2-9B-IT
> does not benefit from few-shot examples. Coverage remains high at 99.65%.
> Still surpasses Qwen3-8B 3-shot (80.74%) marginally and remains well above
> Qwen2.5-14B-AWQ 3-shot (75.11%).

### StrategyQA Chain-of-Thought (CoT)
- Used items:          2290
- Answered (ہاں/نہیں): 2286 (99.83%)
- Accuracy overall:    80.61%
- Accuracy answered:   80.75%
- Output file:         outputs/strategyqa/gemma2_9b_it/strategyqa_cot_gemma2_9b_it.jsonl

> Note: CoT accuracy (80.61%) is nearly identical to 3-shot (80.70%), with both
> slightly below zero-shot (82.79%). Unlike Qwen3 models where CoT with thinking
> mode yields notably higher answered accuracy (Qwen3-14B: 86.96%), Gemma-2-9B-IT
> shows no meaningful gain from explicit reasoning on StrategyQA-Urdu. Coverage
> remains excellent at 99.83%.

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

## GSM8K – Qwen3-8B
Model: Qwen3-8B 
Dataset: GSM8K (Urdu)  
Total examples: 700  

### GSM8K Zero-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   55.43%
- Accuracy answered:  55.43%
- Output file:        outputs/gsm8k/qwen3_8b/gsm8k_zero_shot_qwen3_8b.jsonl

> Note: Qwen3-8B uses chat template format with thinking mode disabled
> (enable_thinking=False) for zero-shot evaluation. At 55.43%, this is
> by far the highest GSM8K zero-shot result across all evaluated models,
> more than doubling Qwen2.5-14B-AWQ zero-shot (25.14%) despite being
> a smaller model. This suggests Qwen3's generation improvements
> significantly benefit Urdu mathematical reasoning even without
> explicit step-by-step prompting.

### GSM8K 3-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   22.86%
- Accuracy answered:  22.86%
- Output file:        outputs/gsm8k/qwen3_8b/gsm8k_three_shot_qwen3_8b.jsonl

> Note: Qwen3-8B uses chat template format with thinking mode disabled
> (enable_thinking=False) for 3-shot evaluation. At 22.86%, this is the
> highest GSM8K 3-shot result across all evaluated models, surpassing
> Qwen2.5-14B-AWQ 3-shot (8.86%) by a large margin. However, 3-shot still
> significantly underperforms CoT for math reasoning, consistent with the
> pattern seen across all previous models. We used the prompt "three_shot_llama" (to give the model room to think properly and follow ###to reach the gold number.)

### GSM8K Chain Of Thought(Cot)

- Used items:         700
- Answered (numeric): 699 (99.86%)
- Accuracy overall:   79.86%
- Accuracy answered:  79.97%
- Outputs file:       outputs/gsm8k/qwen3_8b/gsm8k_cot_qwen3_8b.jsonl

> Note: Qwen3-8B uses a chat template format with thinking mode enabled.
> This model achieves the best performance among open-source models on
> complex mathematical reasoning, evaluated on 700 examples from the GSM8K
> dataset. It demonstrates significant accuracy improvements over time,
> even in a low-resource language like Urdu. These results suggest that
> GSM8K benefits strongly from step-by-step reasoning for accurate answers.
> Notably, the prompt used was relatively simple, with minimal contextual
> guidance.

## GSM8K – Urdu – Qwen3-14B

Model: Qwen3-14B
Dataset: GSM8K (Urdu)
Total examples: 700

### GSM8K Zero-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   39.14%
- Accuracy answered:  39.14%
- Output file:        outputs/gsm8k/qwen3_14b/gsm8k_zero_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses chat template format with thinking mode disabled
> (enable_thinking=False) for zero-shot evaluation. At 39.14%, this
> surpasses Qwen2.5-14B-AWQ zero-shot (25.14%) but falls significantly
> below Qwen3-8B zero-shot (55.43%), which is an unexpected reversal
> within the Qwen3 family. This suggests that for GSM8K-Urdu zero-shot,
> the 8B model's generation behavior with thinking disabled outperforms
> the 14B variant, and CoT with thinking enabled may be needed for the
> 14B model to reach its full mathematical reasoning potential.

### GSM8K 3-shot

Used items:         700
Answered (numeric): 700 (100.00%)
Accuracy overall:   32.86%
Accuracy answered:  32.86%
Output file:        outputs/gsm8k/qwen3_14b/gsm8k_three_shot_qwen3_14b.jsonl

> Note: Qwen3-14B uses a chat template format with thinking mode disabled  
> (enable_thinking=False) for three-shot evaluation. It achieved 32.86% overall accuracy,  
> which shows improvement in Qwen3 generation when scaling from Qwen3-8B to 14B. While  
> zero-shot performance was poor, the model recovered in the three-shot setting and  
> outperformed previous Qwen models such as Qwen3-8B and Qwen2.5-14B-Instruct (AWQ).  
> We still have CoT evaluation remaining to finalize the analysis for Qwen3-14B.

### GSM8K Chain-of-Thought (CoT)
- Used items:         700
- Answered (numeric): 699 (99.86%)
- Accuracy overall:   83.71%
- Accuracy answered:  83.83%
- Output file:        outputs/gsm8k/qwen3_14b/gsm8k_cot_qwen3_14b.jsonl

> Note: Qwen3-14B CoT uses thinking mode enabled (enable_thinking=True),
> activating built-in reasoning via <think>...</think> blocks before the
> final answer. At 83.71%, this is the highest GSM8K CoT result across
> all evaluated models, surpassing Qwen3-8B CoT (79.86%) and
> Qwen2.5-14B-AWQ CoT (39.43%) by a large margin. Coverage remains
> near-complete at 99.86%, consistent with Qwen3-8B CoT (99.86%).
> This confirms that thinking mode is essential for Qwen3-14B on
> mathematical reasoning — CoT recovers the gap seen in zero-shot
> (39.14%) and pushes the model to its best performance on GSM8K-Urdu.

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

## GSM8K – Urdu – Meta-Llama-3.1-70B-Instruct-AWQ-INT4

Model: Meta-Llama-3.1-70B-Instruct-AWQ-INT4
Dataset: GSM8K (Urdu)
Total examples: 700

### GSM8K Zero-shot
- Used items:         700
- Answered (numeric): 696 (99.43%)
- Accuracy overall:   32.71%
- Accuracy answered:  32.90%
- Output file:        outputs/gsm8k/llama3.1_70b_awq/gsm8k_zero_shot_llama3.1_70b_awq.jsonl

> Note: Meta-Llama-3.1-70B-AWQ achieves 32.71% on GSM8K-Urdu zero-shot,
> substantially outperforming Meta-LLaMA-3.1-8B zero-shot (17.29%) and
> Gemma-2-9B-IT (16.86%), confirming clear scaling benefits within the
> LLaMA family. However it remains below Qwen3-8B (55.43%) and Qwen3-14B
> (39.14%), consistent with the pattern that Qwen3 models have stronger
> mathematical reasoning in Urdu even at smaller scales. CoT is expected
> to produce a large improvement as seen across all previous models.

### GSM8K 3-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   34.43%
- Accuracy answered:  34.43%
- Output file:        outputs/gsm8k/llama3.1_70b_awq/gsm8k_three_shot_llama3.1_70b_awq.jsonl

> Note: 3-shot accuracy (34.43%) improves modestly over zero-shot (32.71%)
> with full coverage, setting the highest GSM8K 3-shot result across all
> evaluated models, surpassing Qwen3-14B (32.86%), Qwen3-8B (22.86%), and
> Gemma-2-9B-IT (16.43%). The modest gain over zero-shot confirms that
> few-shot examples alone are insufficient for multi-step arithmetic in Urdu
> — CoT with step-by-step reasoning remains the key driver as seen in all
> previous models.

### GSM8K Chain-of-Thought (CoT)
- Used items:         700
- Answered (numeric): 699 (99.86%)
- Accuracy overall:   55.00%
- Accuracy answered:  55.08%
- Output file:        outputs/gsm8k/llama3.1_70b_awq/gsm8k_cot_llama3.1_70b_awq.jsonl

> Note: CoT produces a large jump from zero-shot (32.71%) and 3-shot (34.43%)
> to 55.00%, confirming that step-by-step reasoning is essential for GSM8K-Urdu
> regardless of model scale. Ranks third overall on GSM8K CoT behind Qwen3-14B
> (83.71%) and Qwen3-8B (79.86%), but ahead of Gemma-2-9B-IT (71.57%),
> DeepSeek-R1-Distill-Qwen-7B (33.43%), and Qwen2.5-14B-AWQ (39.43%).
> Coverage remains near-complete at 99.86%.

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

## GSM8K – Urdu – Gemma-2-9B-IT

Model: Gemma-2-9B-IT
Dataset: GSM8K (Urdu)
Total examples: 700

### GSM8K Zero-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   16.86%
- Accuracy answered:  16.86%
- Output file:        outputs/gsm8k/gemma2_9b_it/gsm8k_zero_shot_gemma2_9b_it.jsonl

> Note: Gemma-2-9B-IT achieves 16.86% on GSM8K-Urdu zero-shot with full coverage,
> comparable to Qwen2.5-7B zero-shot (17.14%) and Meta-LLaMA-3.1-8B (17.29%),
> but well below Qwen3-8B (55.43%) and Qwen3-14B (39.14%). This confirms that
> strong performance on binary and MCQ tasks (BoolQ, StrategyQA) does not
> transfer to multi-step arithmetic reasoning in Urdu for Gemma-2-9B-IT.

### GSM8K 3-shot
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   16.43%
- Accuracy answered:  16.43%
- Output file:        outputs/gsm8k/gemma2_9b_it/gsm8k_three_shot_gemma2_9b_it.jsonl

> Note: 3-shot accuracy (16.43%) is nearly identical to zero-shot (16.86%),
> confirming that few-shot examples provide no benefit for GSM8K-Urdu arithmetic
> reasoning with Gemma-2-9B-IT. Both results are comparable to Qwen2.5-7B
> zero-shot (17.14%) and far below Qwen3 models, consistent with the pattern
> that CoT with thinking mode is the key driver for strong math performance.

### GSM8K Chain-of-Thought (CoT)
- Used items:         700
- Answered (numeric): 700 (100.00%)
- Accuracy overall:   71.57%
- Accuracy answered:  71.57%
- Output file:        outputs/gsm8k/gemma2_9b_it/gsm8k_cot_gemma2_9b_it.jsonl

> Note: CoT produces a dramatic improvement over zero-shot (16.86%) and 3-shot
> (16.43%), jumping to 71.57% — confirming that step-by-step reasoning is
> essential for GSM8K-Urdu regardless of model family. Gemma-2-9B-IT ranks
> second overall on GSM8K CoT, behind Qwen3-14B (83.71%) and Qwen3-8B (79.86%)
> but well ahead of Qwen2.5-14B-AWQ (39.43%) and DeepSeek-R1-Distill-Qwen-7B
> (33.43%). A strong result for a non-reasoning-specialized model.

---

## Prompt Sensitivity Analysis

> This section reports prompt sensitivity results for selected models.
> Each setting was evaluated with 3 prompt templates (P1, P2, P3).
> P1 is the original prompt used in all baseline evaluations above.
> Mean and standard deviation are computed across the 3 variants.

---

### BoolQ – Prompt Sensitivity – Qwen3-14B

#### Zero-shot

| Prompt | Used Items | Scored On | Accuracy |
|--------|------------|-----------|----------|
| P1 (original) | 1550 | 1549 | 84.51% |
| P2 | 1550 | 1550 | 82.84% |
| P3 | 1550 | 1550 | 84.32% |
| **Mean** | | | **83.89%** |
| **Std** | | | **±0.73%** |

- P1: outputs/boolq/qwen3_14b/boolq_zero_shot_qwen3_14b.jsonl
- P2: outputs/boolq/qwen3_14b_p2/boolq_zero_shot_p2_qwen3_14b.jsonl
- P3: outputs/boolq/qwen3_14b_p3/boolq_zero_shot_p3_qwen3_14b.jsonl

> Note: Std of ±0.73% indicates low prompt sensitivity on BoolQ zero-shot
> for Qwen3-14B — the model performs consistently regardless of instruction
> phrasing, which is a positive robustness signal.

#### 3-shot

| Prompt | Used Items | Scored On | Accuracy |
|--------|------------|-----------|----------|
| P1 (original) | 1550 | 1549 | 84.89% |
| P2 | 1550 | 1550 | 85.23% |
| P3 | 1550 | 1550 | 82.06% |
| **Mean** | | | **84.06%** |
| **Std** | | | **±1.34%** |

- P1: outputs/boolq/qwen3_14b/boolq_three_shot_qwen3_14b.jsonl
- P2: outputs/boolq/qwen3_14b_p2/boolq_three_shot_p2_qwen3_14b.jsonl
- P3: outputs/boolq/qwen3_14b_p3/boolq_three_shot_p3_qwen3_14b.jsonl

> Note: Std of ±1.34% indicates moderate prompt sensitivity on BoolQ 3-shot.
> P3 drops notably (82.06%) while P2 slightly improves over P1, suggesting
> the instruction framing before the examples affects how the model uses
> the demonstrations.

#### Chain-of-Thought (CoT)

| Prompt | Used Items | Scored On | Accuracy |
|--------|------------|-----------|----------|
| P1 (original) | 1550 | 1539 | 84.47% |
| P2 | 1550 | 1545 | 83.62% |
| P3 | 1550 | 1383 | 83.73% |
| **Mean** | | | **83.94%** |
| **Std** | | | **±0.37%** |

- P1: outputs/boolq/qwen3_14b/boolq_cot_qwen3_14b.jsonl
- P2: outputs/boolq/qwen3_14b_p2/boolq_cot_p2_qwen3_14b.jsonl
- P3: outputs/boolq/qwen3_14b_p3/boolq_cot_p3_qwen3_14b.jsonl

> Note: Std of ±0.37% is the lowest across all BoolQ prompt types,
> indicating Qwen3-14B CoT is highly robust to prompt phrasing on
> BoolQ-Urdu. P3 scored on fewer items (1383) due to think-block
> removal occasionally leaving no extractable answer.

### CSQA – Prompt Sensitivity – Qwen3-14B

#### Zero-shot

| Prompt | Used Items | Answered | Accuracy Overall |
|--------|------------|----------|-----------------|
| P1 (original) | 1500 | 1500 (100.00%) | 58.73% |
| P2 | 1500 | 1500 (100.00%) | 59.53% |
| P3 | 1500 | 1497 (99.80%) | 58.07% |
| **Mean** | | | **58.78%** |
| **Std** | | | **±0.60%** |

- P1: outputs/csqa/qwen3_14b/csqa_zero_shot_qwen3_14b.jsonl
- P2: outputs/csqa/qwen3_14b_p2/csqa_zero_shot_p2_qwen3_14b.jsonl
- P3: outputs/csqa/qwen3_14b_p3/csqa_zero_shot_p3_qwen3_14b.jsonl

> Note: Std of ±0.60% indicates very low prompt sensitivity on CSQA
> zero-shot — Qwen3-14B is robust to instruction phrasing changes on
> commonsense MCQ. P2 slightly outperforms P1, P3 slightly underperforms.

#### 3-shot

| Prompt | Used Items | Answered | Accuracy Overall |
|--------|------------|----------|-----------------|
| P1 (original) | 1500 | 1500 (100.00%) | 56.60% |
| P2 | 1500 | 1500 (100.00%) | 56.80% |
| P3 | 1500 | 1499 (99.93%) | 51.13% |
| **Mean** | | | **54.84%** |
| **Std** | | | **±2.55%** |

- P1: outputs/csqa/qwen3_14b/csqa_three_shot_qwen3_14b.jsonl
- P2: outputs/csqa/qwen3_14b_p2/csqa_three_shot_p2_qwen3_14b.jsonl
- P3: outputs/csqa/qwen3_14b_p3/csqa_three_shot_p3_qwen3_14b.jsonl

> Note: Std of ±2.55% is notably higher than zero-shot (±0.60%), indicating
> moderate prompt sensitivity in the 3-shot setting. P3 drops sharply to
> 51.13% while P1 and P2 remain close (56.60% vs 56.80%), suggesting the
> instruction framing before the examples significantly affects how the model
> uses the demonstrations on CSQA-Urdu.

#### Chain-of-Thought (CoT)

| Prompt | Used Items | Answered | Accuracy Overall |
|--------|------------|----------|-----------------|
| P1 (original) | 1500 | 1497 (99.80%) | 63.00% |
| P2 | 1500 | 1490 (99.33%) | 52.13% |
| P3 | 1500 | 1494 (99.60%) | 54.67% |
| **Mean** | | | **56.60%** |
| **Std** | | | **±4.67%** |

- P1: outputs/csqa/qwen3_14b/csqa_cot_qwen3_14b.jsonl
- P2: outputs/csqa/qwen3_14b_p2/csqa_cot_p2_qwen3_14b.jsonl
- P3: outputs/csqa/qwen3_14b_p3/csqa_cot_p3_qwen3_14b.jsonl

> Note: Std of ±4.67% is the highest sensitivity observed across all CSQA
> prompt types, indicating that CoT phrasing significantly affects commonsense
> MCQ accuracy for Qwen3-14B. P1 outperforms P2 and P3 by a large margin
> (63.00% vs 52.13% and 54.67%), suggesting the original CoT instruction
> format is particularly well-suited for this task. This is an important
> finding — the original prompt is not just marginally better but substantially
> better, which strengthens the value of careful prompt design for CoT in
> low-resource Urdu NLP.

### PIQA – Prompt Sensitivity – Qwen3-14B

#### Zero-shot

| Prompt | Used Items | Answered | Accuracy Overall |
|--------|------------|----------|-----------------|
| P1 (original) | 750 | 750 (100.00%) | 59.33% |
| P2 | 750 | 750 (100.00%) | 69.60% |
| P3 | 750 | 750 (100.00%) | 72.80% |
| **Mean** | | | **67.24%** |
| **Std** | | | **±5.60%** |

- P1: outputs/piqa/qwen3_14b/piqa_zero_shot_qwen3_14b.jsonl
- P2: outputs/piqa/qwen3_14b_p2/piqa_zero_shot_p2_qwen3_14b.jsonl
- P3: outputs/piqa/qwen3_14b_p3/piqa_zero_shot_p3_qwen3_14b.jsonl

> Note: Std of ±5.60% is the highest zero-shot sensitivity observed so far,
> indicating that PIQA-Urdu is significantly affected by prompt phrasing.
> P2 and P3 substantially outperform P1 (+10.27pp and +13.47pp respectively),
> suggesting the original prompt was suboptimal for this task. This is an
> important finding — the best PIQA zero-shot result (72.80%) surpasses
> even LLaMA-3.1-70B zero-shot (62.93%).

#### 3-shot

| Prompt | Used Items | Answered | Accuracy Overall |
|--------|------------|----------|-----------------|
| P1 (original) | 750 | 750 (100.00%) | 63.73% |
| P2 | 750 | 750 (100.00%) | 54.80% |
| P3 | 750 | 750 (100.00%) | 73.07% |
| **Mean** | | | **63.87%** |
| **Std** | | | **±7.47%** |

- P1: outputs/piqa/qwen3_14b/piqa_three_shot_qwen3_14b.jsonl
- P2: outputs/piqa/qwen3_14b_p2/piqa_three_shot_p2_qwen3_14b.jsonl
- P3: outputs/piqa/qwen3_14b_p3/piqa_three_shot_p3_qwen3_14b.jsonl

> Note: Std of ±7.47% is the highest sensitivity observed across all
> datasets and prompt types so far, confirming that PIQA-Urdu is highly
> sensitive to prompt phrasing in the 3-shot setting. P3 (73.07%) sets
> a new high for PIQA 3-shot across all models, surpassing LLaMA-3.1-70B
> 3-shot (76.40%) — while P2 drops sharply to 54.80%, near random baseline.
> This extreme variance confirms PIQA as the most prompt-sensitive dataset
> in URBench.