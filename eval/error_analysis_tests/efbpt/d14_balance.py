#!/usr/bin/env python3
"""
d14_balance.py — read the gateway's balance, if it has a balance query (D14 block limits).

Tries the usual billing endpoints of an OpenAI-compatible gateway with the key in the
Authorization header. Prints HTTP status codes and numbers only: no header, no key, and
every printed or stored string is scrubbed. The result is appended to the D14 API ledger.
No chat request is made, so nothing is added to the call or token counts.

  --min X   if a balance is found and it is below X: mark API work as stopped in the ledger, exit 5
  --note S  a short label stored with the reading (e.g. "before", "after")
Exit codes: 0 balance read (or no balance query available); 5 balance below --min.
"""
import argparse
import datetime
import json
import os
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import d14_common as C  # noqa: E402


def numbers(obj, prefix=""):
    """Numeric leaves of a JSON object as {path: number} (no strings are kept)."""
    out = OrderedDict()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numbers(v, prefix + str(k) + "."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            out.update(numbers(v, prefix + "%d." % i))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix.rstrip(".")] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    api = C.Api()
    root = api.base[:-3] if api.base.endswith("/v1") else api.base
    today = datetime.date.today()
    span = "start_date=%s&end_date=%s" % ((today - datetime.timedelta(days=99)).isoformat(), (today + datetime.timedelta(days=1)).isoformat())
    tries = [("subscription", api.base + "/dashboard/billing/subscription"), ("usage", api.base + "/dashboard/billing/usage?" + span),
             ("subscription_root", root + "/dashboard/billing/subscription"), ("usage_root", root + "/dashboard/billing/usage?" + span),
             ("credit_grants", api.base + "/dashboard/billing/credit_grants"), ("user_balance", root + "/user/balance"),
             ("token_usage", root + "/api/usage/token")]
    seen = OrderedDict()
    for name, url in tries:
        try:
            r = api.session.get(url, headers={"Authorization": "Bearer " + api._key}, timeout=30)
            try:
                j = r.json()
            except Exception:
                j = None
            nums = numbers(j) if r.status_code == 200 and j is not None else OrderedDict()
            seen[name] = OrderedDict([("http_status", r.status_code), ("numbers", OrderedDict((api.scrub(k, api.secrets), v) for k, v in nums.items()))])
        except api.requests.RequestException as e:
            seen[name] = OrderedDict([("http_status", None), ("error", type(e).__name__)])
        print("  %-18s HTTP %s  %s" % (name, seen[name]["http_status"], json.dumps(seen[name].get("numbers", {}))), flush=True)

    def num(name, key):
        return (seen.get(name, {}).get("numbers") or {}).get(key)

    balance, how = None, None
    for sub, use in (("subscription", "usage"), ("subscription_root", "usage_root")):
        limit, used = num(sub, "hard_limit_usd"), num(use, "total_usage")
        if limit is not None and used is not None:
            balance, how = limit - used / 100.0, "hard_limit_usd - total_usage / 100 (%s, %s)" % (sub, use)
            break
    if balance is None:
        for name, key in (("credit_grants", "total_available"), ("user_balance", "balance"), ("user_balance", "data.balance"),
                          ("token_usage", "data.total_available")):
            if num(name, key) is not None:
                balance, how = num(name, key), "%s: %s" % (name, key)
                break
    rec = OrderedDict([("ts", time.strftime("%Y-%m-%d %H:%M:%S")), ("note", args.note), ("api_host", api.host),
                       ("balance", balance), ("how", how), ("endpoints", seen)])

    def fn(led):
        led["balance"].append(rec)
        if args.min is not None and balance is not None and balance < args.min and not led["stopped"]:
            led["stopped"] = "balance %.4f is below %.2f" % (balance, args.min)
    C._ledger_update(fn)
    if balance is None:
        print("[balance] no balance query answered with a balance; recorded in the ledger")
        return
    print("[balance] %.4f  (%s)" % (balance, how))
    if args.min is not None and balance < args.min:
        print("[balance] below %.2f: API work is marked as stopped in the ledger" % args.min)
        sys.exit(5)


if __name__ == "__main__":
    main()
