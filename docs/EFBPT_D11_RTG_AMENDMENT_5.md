# EFBPT D11 RTG — AMENDMENT 5 (eval458 readout fixed before scoring)
A5.1 Audit result (e3fa2ca). Validated scorer = E3 (opening verdict):
 stratified agreement with the adjudicated labels 95.75 % against 86.00 % for
 the frozen extractor E0; same order under the ALT labels. E3 is the function
 in d11_scorer_audit.py at that commit; its MD5 is pinned in the readout
 script.
A5.2 The eval458 readout is one new script, written and committed now, tested
 on the sealed DEV200 arms (it must reproduce the sealed E0 numbers and the
 A4.4 E3 table), and run unchanged on eval458. It edits no frozen file.
A5.3 For every comparison the readout prints four lines. L1: E0, all rows
 (the pre-registered gate). L2: E3, all rows (validated scorer; A4.4 tier).
 L3: E0, rows parsed in both arms (A3.1). L4: rows where E0 and E3 give the
 same label in both arms (scorer-unambiguous rows; descriptive). Each line:
 n, accuracy of both arms, delta, b, c, exact two-sided McNemar p. Per arm,
 under E0 and E3: unparsed, predicted-yes rate, accuracy on gold-yes and on
 gold-no rows, balanced accuracy (descriptive; separates knowledge from a
 shift to "no").
A5.4 Verdict tiers as A4.4. L1 passes (>= +5.0 pp and p < 0.05; A3.1 decides
 between L1 and L3) -> CONFIRMED. L1 missed and L2 passes -> CONFIRMED UNDER
 THE VALIDATED SCORER; PRE-REGISTERED GATE MISSED. Neither -> NOT CONFIRMED.
 The secondary TR_L1_eval is tested at 0.05 in the tier that confirmed the
 primary (L1 if the gate passed, L2 if only the validated scorer passed) and
 is descriptive otherwise.
A5.5 Content control = TR vs TR_wrong_eval (all four lines). TR_wrong_eval vs
 A_r2_eval is reported; under E3 on DEV200 it is -4.5 pp (p 0.38), so "wrong
 facts harm the baseline" is not claimed under E3.
A5.6 Limits of the audit: one adjudicator, not fully blind; the order among
 E1-E3 rests on three items; the agree stratum is 20 rows per arm. Two more
 adjudicators (API models, same 105 items, same rubric, blind) are added for
 agreement only; they select nothing.
