# EFBPT D14 — AMENDMENT 1 (written after TEST-5, before any FULL run; it changes no arm, prompt, model or decoding)

Companion to docs/EFBPT_D14_MORE_TASKS_FREEZE.md (MD5 31bc15881ad8102a2b35a676abf1c268). No D14 arm has been scored. TEST-5 rows are never scored.

A1.1 Fact (teacher engine). With the D10 engine arguments and max_num_seqs 16 (freeze F4) the
teacher engine does not start on an L20 under vLLM 0.13.0. TEST-5 job 118352: "To serve at
least one request with the models's max seq len (4096), (1.25 GiB KV cache is needed, which is
larger than the available KV cache memory (0.83 GiB)". Job 118357 tried max_num_seqs 16, 12, 8,
6, 5, 4 and 3: each time 0.83 GiB, no start. With max_num_seqs 1 (D10, D13 job 116802) the same
engine has 1.45 GiB. Cause, from the engine logs: with max_num_seqs above 1 the engine sizes
its start-up run with max_num_batched_tokens 8192; with max_num_seqs 1 it uses 4096, the
max_model_len.

A1.2 Rule. The teacher engine of D14 (teacher facts and teacher arms, TEST-5 and FULL) is the
D10 engine arguments with max_num_seqs 16 and with max_num_batched_tokens 4096, the value the
D10 engine itself runs with. Nothing else changes: same model, max_model_len 4096, awq,
gpu_memory_utilization 0.90, enforce_eager, temperature 0, top_p 1, seed 0, the same prompts
and max_tokens.

A1.3 Check. TEST-5 job 118359 on L20003 with these arguments: the engine started with
max_num_seqs 16, "Available KV cache memory: 1.45 GiB", "GPU KV cache size: 4,752 tokens"; the
2 facts files and the 9 teacher arm files of TEST-5 were generated and sealed. The student
TEST-5 job 118360 then generated and sealed the 11 student arm files. The cache holds 4,752
tokens in all, so the engine runs fewer than 16 sequences at a time when their prompts and
outputs are long; this affects speed only.

A1.4 Balance (block limit, not a freeze item). The gateway answers the billing query, but the
key has no limit there (hard_limit_usd 100000000), so no balance is given. The amount used by
the key is read before and after the API runs and reported. The call cap, the token cap and
the stop on the first quota or balance error stay as in freeze 9.9.

A1.5 API order (freeze 9.9). TEST-5 token counts project about 3.4 million tokens and about
11,900 calls for the translations and all nine API arms, inside both caps. So the three tasks
run side by side, one process per task, each in the order M_ur, M_r1, M_og.
