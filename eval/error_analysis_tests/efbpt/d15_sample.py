#!/usr/bin/env python3
"""
d15_sample.py — the D15 sample (freeze N1, 11.3). Reads the pinned UrduMMLU file; no model.

Valid rows: correct_key in A-D; options with exactly the keys A-D, all non-empty after strip;
question non-empty after strip. Exact duplicates (question + the four options) over the whole
file: the lowest id is kept. For each subdomain (sorted names) a new random.Random(1515) samples
100 of its kept rows sorted by id (all of them if there are at most 100). Run order = ascending id.

Writes eval/error_analysis_tests/efbpt/d15_sample_ids.txt (one id per line) and prints its MD5
and the counts per subdomain. If the file exists it must hold exactly this list (never rewritten).

  python eval/error_analysis_tests/efbpt/d15_sample.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d15_common as C  # noqa: E402
from d15_common import die  # noqa: E402


def main():
    C.verify_pins(need_sample=False)
    ids, rep = C.select_sample()
    text = "".join("%d\n" % i for i in ids)
    path = C.rp(C.SAMPLE_PATH)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            if f.read() != text:
                die("%s exists and differs from the sample drawn now; it is never rewritten" % C.SAMPLE_PATH)
        print("[ok] %s exists and equals the sample drawn now" % C.SAMPLE_PATH)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("[wrote] %s" % C.SAMPLE_PATH)
    print("rows %d  valid %d  invalid %s  duplicates dropped %d  kept %d  sampled %d"
          % (rep["rows"], rep["valid"], dict(rep["invalid"]), rep["duplicates_dropped"], rep["kept"], rep["sampled"]))
    print("%-36s %-8s %6s %6s %7s" % ("subdomain", "group", "rows", "kept", "sampled"))
    for sub, d in rep["per_subdomain"].items():
        print("%-36s %-8s %6d %6d %7d" % (sub, d["group"], d["rows"], d["kept"], d["sampled"]))
    print("sample MD5 %s" % C.md5_file(C.SAMPLE_PATH))


if __name__ == "__main__":
    main()
