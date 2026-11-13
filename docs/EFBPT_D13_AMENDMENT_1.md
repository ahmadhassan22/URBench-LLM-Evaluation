# EFBPT D13 — AMENDMENT 1 (written after TEST-5, before any FULL run; it changes no arm, prompt, model or decoding, and touches only the readout)
A1.1 Truncation. In TEST-5, 3 of 30 small-student rows reached 1024 new tokens in a
repetition loop (Llama8B_TR 1, Qwen8B_A 2). E0 reads the last mention, so a loop can change
the label. Decoding stays as frozen (greedy, max_new 1024, no repetition penalty).
A1.2 Added line L6 = E0 on rows truncated in neither arm, printed for every comparison,
with the truncated count of each arm.
A1.3 Rule (mirror of D11 Amendment 3 A3.1): if the truncated rates of the two arms of a test
differ by more than 5.0 pp, that test is read on L6 and the L1 numbers are reported beside it.
