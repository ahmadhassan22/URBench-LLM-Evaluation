# EFBPT D10 — AMENDMENT 2 (before FULL; freeze b3a822d8…, Amendment 1 applied)

A2.1 Sign rule for the R3 TF_wrong check (pre-declared before any FULL number).
The script flags |TF_wrong − A_r2| > 4.0 pp as FORMAT_EFFECT. It is read by sign:
- TF_wrong − A_r2 > +4.0: FORMAT-EFFECT. The facts block helps regardless of
  content; the method claim is reduced to TF − TF_wrong (freeze §7).
- TF_wrong − A_r2 < −4.0: WRONG-FACTS HARM. Facts about the wrong entity hurt
  (the D8 GEN mechanism). The R1 claim (TF − A_r2) stands; TF − TF_wrong is
  reported as descriptive; the harm is reported as a finding.
Motivation: in student TEST-5 (job 109063) TF_wrong predicted "yes" on 0/5
rows, so the negative direction is plausible and must not be argued after
the fact.

A2.2 Job S FULL time limit: raised from 08:00:00 to 12:00:00. Basis: TEST-5
timing 24.7 s/row at batch size 1 → 4 × 200 rows + 40-generation V2 probe +
load ≈ 5.9 h. Arm order A_r2, TF, TF_wrong, TF_noq is unchanged; arm files
are sealed as each arm completes.

A2.3 FULL Job T and FULL Job S are submitted together, S with
--dependency=afterok on T. S's own preflight (V7, V8, R4 and the leak guards)
is the gate before any student generation. Nothing else changes.
