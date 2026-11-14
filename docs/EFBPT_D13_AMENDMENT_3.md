# EFBPT D13 — AMENDMENT 3 (before any D13 arm is scored)
A3.1 Fact. For API_tr_R1 on eval458 the provider refuses two rows (qids 90538ec7aa1100a34829
and 78f2f99d04b9acd8bada) with HTTP 400 data_inspection_failed, on the first run and again on
a resume of only those rows. It is not a network fault.
A3.2 Rule. A row refused by the provider's input inspection on two separate attempts is
recorded as refused, with the error text, and is not requested again. The arm is sealed with
456 ok records and 2 refused records (458 rows present).
A3.3 Scoring. A refused row has no answer. It is counted as unparsed = wrong, and it is
treated as an unparsed row on every line. For every comparison with API_tr_R1 on eval458 the
readout adds one line on the 456 rows the provider answered.
A3.4 The seal is written by a new script after all GPU jobs are sealed. No existing d13
script is edited while jobs are pending, because the C1 and C2 passes are tied to the
script MD5s.
