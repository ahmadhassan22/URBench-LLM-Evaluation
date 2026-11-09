# EFBPT D11-R — AMENDMENT 2 (2026-11-09, after the 2026-11-07 API chain; local readers L1–L4 untouched)

Freeze: docs/EFBPT_D11R_READER_FREEZE.md (MD5 ed260e0ea2373337e8a3fb850bebe963).
Amendment 1: docs/EFBPT_D11R_AMENDMENT_1.md (MD5 fb7bc3c922bb28d667241a7dd75b6938).
Local readers L1_llama70b_awq, L2_gemma2_27b_nf4, L3_qwen3_14b_nf4, L4_nllb200_3.3b:
all three sets sealed; nothing in this amendment touches them.

A2.1 Outage record (2026-11-07, logs/d11r_api_full.log, chain 22:07:09 → 23:32:24 CST).
     Every API request from 22:10:26 until the chain end failed with one error
     type, 3 attempts each, no HTTP status (the request never left the host):
       network: HTTPSConnectionPool(host='api.tokenroc.com', port=443):
       Max retries exceeded ... ProxyError('Unable to connect to proxy',
       NewConnectionError("HTTPSConnection(host='localhost', port=20000):
       ... [Errno 111] Connection refused"))
     i.e. the local HTTPS proxy on localhost:20000 was down; the provider was
     not reached. 3,261 error records in total; the chain kept starting sets
     (12 of 12 started), each one failing on every row.
     Completed before the outage:
       qwen3.8-max       rdev300  300/300 ok, sealed 22:09:02
                         (readings_rdev300.jsonl MD5 917ed0fac0b0e79b1446bf33a0d50704)
       deepseek-v4-flash rdev300  271/300 ok (first error 22:10:26; 29 qids open)
     Every other (model, set): 0 ok — qwen3.7-flash and deepseek-v4-pro on
     rdev300; all four models on dev200 and eval458.
     2026-11-09 check before resuming: proxy listening on 127.0.0.1:20000; one
     test request (qwen3.8-max, enable_thinking=false, RDEV300 row 0) → HTTP 200,
     prompt 127 / completion 16 tokens, finish "stop", a one-line English
     reading; GET /models → HTTP 200.

A2.2 Concurrency and circuit breaker (d11r_api_reader.py, FULL runs only).
     Concurrency 4 → 2 (FULL_CONCURRENCY in the reader; the probe is finished
     and keeps its recorded setting). Circuit breaker: after 10 consecutive
     failed rows (completion order) the run stops submitting, lets the in-flight
     requests finish and record, sleeps 60 s and sends one probe request = the
     frozen request for the next pending row. If the probe succeeds its record
     is kept and the run continues with the remaining rows; if it fails the run
     prints the error and exits with code 4, and the chain stops (no further
     set is started). Chain (eval/error_analysis_tests/efbpt/d11r_api_chain.sh,
     same order as 2026-11-07: sets rdev300, dev200, eval458; models
     qwen3.8-max, deepseek-v4-flash, qwen3.7-flash, deepseek-v4-pro): continues
     after exit 3 (INCOMPLETE, resumable scattered errors), stops on exit 4 or
     any other non-zero code. Operational only: request body, temperature 0,
     max_tokens 256, retry policy (429/5xx and connection errors, backoff
     2/4/8 s, 3 attempts), prompt and post-processing are unchanged, so readings
     are unaffected. R_START records concurrency 2, the breaker rule and this
     amendment's MD5 (in the "runs" entry for readers whose R_START exists).

A2.3 Resume. The same commands are re-run. Rows with an ok record are kept and
     skipped; error records stay in the file until the set completes, when the
     file is compacted to one ok record per qid in item order and the set is
     sealed into R_SEAL.json (unchanged from the freeze). A set already listed
     in R_SEAL sets_complete (qwen3.8-max rdev300) is left untouched: the reader
     prints "already sealed" and exits 0 without rewriting file or seal.
     Resume log: logs/d11r_api_full_resume.log.

Every R_START written from now on records this amendment's MD5.
