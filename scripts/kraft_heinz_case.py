#!/usr/bin/env python3
"""独立核对：Kraft Heinz 2017 年那条静默重述。

用 Company Facts 这个独立数据源核对单个已知案例：-9,000,000 在次年年报的
可比列里被改成 627,000,000，form 是 10-K 而不是 10-K/A，「只看 10-K/A
就能识别重述」的常见做法会完全漏掉它。

用法：python3 scripts/kraft_heinz_case.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secpit.config import COMPANYFACTS_URL, Config, sec_headers   # noqa: E402

CIK = 1637459                       # Kraft Heinz Co
CONCEPT = "OtherNonoperatingIncomeExpense"
UNIT = "USD"
PERIOD_END = "2017-12-30"           # KHC 的财年末
EXPECT_FIRST = -9_000_000.0
EXPECT_FINAL = 627_000_000.0


def fetch_companyfacts(cfg: Config, cik: int) -> dict:
    cache = cfg.raw_dir / "companyfacts"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"CIK{cik:010d}.json"
    if not path.exists():
        url = COMPANYFACTS_URL.format(cik=cik)
        print(f"GET {url}")
        resp = requests.get(url, headers=sec_headers(cfg.user_agent, "data.sec.gov"),
                            timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        time.sleep(cfg.rate_limit_sleep)
    return json.loads(path.read_text())


def main() -> int:
    cfg = Config.default()
    cfg.ensure_dirs()
    facts = fetch_companyfacts(cfg, CIK)

    series = [
        r for r in facts["facts"]["us-gaap"][CONCEPT]["units"][UNIT]
        if r["end"] == PERIOD_END and r.get("start", "").startswith("2017-01")
    ]
    series.sort(key=lambda r: (r["filed"], r["accn"]))

    print(f"\n{facts['entityName']}  (CIK {CIK})")
    print(f"{CONCEPT}  end={PERIOD_END}  unit={UNIT}\n")
    print(f"  {'filed':<12}{'form':<10}{'accn':<24}{'value':>18}")
    print("  " + "-" * 64)
    for r in series:
        print(f"  {r['filed']:<12}{r['form']:<10}{r['accn']:<24}{r['val']:>18,}")

    if not series:
        print("\n未找到该 (concept, end) 的任何记录")
        return 1

    first, final = series[0], series[-1]
    checks = [
        ("首次申报值为 -9,000,000", first["val"] == EXPECT_FIRST),
        ("最终值为 627,000,000", final["val"] == EXPECT_FINAL),
        ("原始值未被覆盖（序列中两个值并存）",
         len({r["val"] for r in series}) >= 2),
        ("重述由 10-K 而非 10-K/A 带来（静默重分类）",
         all(not r["form"].endswith("/A") for r in series)),
    ]
    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed

    if ok:
        lag = (dt.date.fromisoformat(final["filed"])
               - dt.date.fromisoformat(first["filed"])).days
        print(f"\n  原始值发布 {lag} 天后被改写，且全程没有任何正式修正案。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
