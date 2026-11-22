#!/usr/bin/env python3
"""
d15_balance.py — the amount used by the API key, before and after the D15 API runs (freeze 11.9).

Runs d14_balance.main unchanged (imported) with the reused ledger code pointed at the D15 ledger
by d15_common: billing endpoints only, no chat request, nothing added to the call or token counts;
HTTP status codes and numbers only are printed; the key is read inside this process and never
shown. The reading is appended to outputs/efbpt/d15/api_ledger.json.

  python eval/error_analysis_tests/efbpt/d15_balance.py --note before
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d15_common as C  # noqa: E402

if __name__ == "__main__":
    C.verify_pins(verbose=False)
    C.ledger_init()
    import d14_balance
    d14_balance.main()
