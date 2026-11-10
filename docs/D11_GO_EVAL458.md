# D11 — GO for eval458 (2026-11-10)
DEV200 (readout c529fbd): TR 68.5 vs A_r2 60.0, +8.5 pp, b 39 / c 22, exact
McNemar p = 0.0396 → GO (freeze §7). DEV200-best = TR (R1 api_qwen3.8-max).
Controls: TR_wrong −9.5 (p 0.042), TR_noq −13.0, V1 4.5 ≤ 5.0, V2' 5/5, V3,
V4, V8 pass. Hand check of the 22 TR-loss rows done (Amendment 2 A2.1): no
leak, no pipeline fault. eval458 runs once, per freeze §8 + Amendment 2.
