# D11 RTG decision (DEV200, freeze §7)

Freeze: docs/EFBPT_D11_RTG_FREEZE.md (MD5 ce7d68801c302fa5a8d173bd3ee93eb7); Amendment 1
(MD5 0b7202af45eac170ec2d8fe281269e9e). Sealed student run: slurm 115173, S_SEAL.json
2026-11-10 02:22:52. Readout: docs/handoff/D11_FULL_READOUT.md (every number recomputed
from the sealed jsonl files).

Rule (§7): TR vs A_r2 on DEV200, exact two-sided McNemar.
GO ≥ +8.0 pp and p < 0.05 → eval458 once (§8). CONDITIONAL +5.0 to +7.9 → eval458 once
with the DEV200-best of {TR, TR2, TR_L1}. < +5.0 → RTG closed; fallback D12.
eval458 arm = DEV200-best of {TR, TR2, TR_L1}; tie → TR.

Observed: TR 68.50% (137/200) vs A_r2 60.00% (120/200): Δ = +8.5 pp, wins 39, losses 22,
p = 0.0396.

Verdict: **GO** (+8.5 ≥ +8.0 and p = 0.0396 < 0.05) → eval458 once (§8).
DEV200-best of {TR 68.5, TR2 65.0, TR_L1 67.0} = **TR** (reader R1 api_qwen3.8-max).
eval458 arms (§8): A_r2_eval, TR, TR_wrong_eval; readings = R1's sealed D11-R eval458 file.

Validity: V1 PASS (spread 4.5 pp), V2' PASS (5/5), V3 PASS (0 fatal), V4 PASS (3 MD5s = freeze §2),
V5 as expected (TR_wrong − A_r2 = −9.5), V8 PASS (998 prompts), V6 and V9 reported.

docs/D11_GO_EVAL458.md is NOT written by this decision: it waits for the hand check of the
TR-vs-A_r2 loss and win rows (readout §8).
