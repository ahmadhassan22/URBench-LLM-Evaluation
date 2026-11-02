# EFBPT D10 — AMENDMENT 4 (before the v2 FULL run): JSON parse repair

Trigger: v2 teacher TEST-5 (job 110566), row 01c27885…: a complete 5-key
object with only the final "}" missing, finish reason "stop". The strict
parser (first balanced block) returns none, so the row loses its facts.

A4.1 Parse repair (teacher v2 only). After fence stripping, if no balanced
{...} block is found, try appending "}" and parse again; if that fails, try
appending "]}" and parse again. At most these two attempts, in that order.
A row parsed this way records repaired = "brace" or "bracket_brace",
otherwise repaired = false. The parse rule itself (five keys, types) is
unchanged. The number of repaired rows is printed, stored in T_START and
T_SEAL, and reported in the readout as a descriptive (A3.3).
A4.2 A 25-row teacher-only TEST (TEST mode, no student arm, no accuracy) is
run as a format check; its outputs are TEST outputs and are overwritten.
A4.3 TEST-5 (teacher and student) is re-run with the repaired parser before
FULL. Nothing else changes.
