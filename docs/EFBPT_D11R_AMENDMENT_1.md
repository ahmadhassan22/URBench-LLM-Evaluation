# EFBPT D11-R — AMENDMENT 1 (2026-11-07, after the PROBE, before any FULL run)

Freeze: docs/EFBPT_D11R_READER_FREEZE.md (MD5 ed260e0ea2373337e8a3fb850bebe963).
Probe records: outputs/efbpt/d11r/_probe/probe_summary.json (3 RDEV300 rows ×
4 API models × 4 thinking variants, 2026-11-07).

A1.1 Thinking. From the probe the FULL run uses enable_thinking=false for all
     four API models (qwen3.8-max, deepseek-v4-pro, deepseek-v4-flash,
     qwen3.7-flash). A reply with no reasoning_tokens field, no
     reasoning_content and no <think> tag is recorded as reasoning_tokens = 0.
     deepseek-v4-pro rejects reasoning_effort="none" (HTTP 400: must be one of
     low/medium/high/xhigh/max) and, at default thinking, returned an empty
     reading on probe row 0 (256 tokens all reasoning, finish "length").

A1.2 M1 denominator. M1 is the macro average over rows with >= 1 gold proper
     noun (RDEV300: 233/300). Rows with none are counted and reported, not
     scored on M1. M2–M4 use all rows.

A1.3 data/ is gitignored; the RDEV300 jsonl and its seal were force-added
     (commit af7265f), the same way as DEV200.

A1.4 configs/d11r_api_prices.json may stay null; cost is the last tie-break
     and is filled from the provider list only if that tie occurs.

Every R_START and R_SEAL written from now on records this amendment's MD5.
