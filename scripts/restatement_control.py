"""Separate publication lag from restatement in the vintage differential.

`naive` in secpit/demo.py leaks through two channels that truncation cannot see:
it reads a quarter's figures before they are filed (publication lag), and it
reads the latest revision of every figure (restatement). This script isolates
the two.

  restated   picks the period exactly as `correct` does (ddate <= t and
             filed <= t) but reads that period's value from the latest version
             visible in the world, so in D_rev it sees later restatements.
             Its only channel is restatement.

  no-revision control
             every business key is frozen at its first-filed value, so the
             table contains no revisions at all. Whatever Λ survives is not
             restatement.

The split depends on where the as-of dates sit. At a quarter-end none of that
quarter's reports are filed yet, so the lag channel is at its largest; the
script therefore runs the same comparison a second time with every as-of date
moved 60 days past the quarter-end, by which point most quarterly reports are in.

Run from the repository root after `build`:

    python3 scripts/restatement_control.py

Output on 2009Q2–2026Q2 (24 as-of dates):

    as-of at quarter-end
      naive     original      Λ_cell 1.000  Λ_rank 0.0829
      naive     no-revision   Λ_cell 1.000  Λ_rank 0.0715
      correct   original      Λ_cell 0.000  Λ_rank 0.0000
      restated  original      Λ_cell 0.952  Λ_rank 0.0070
      restated  no-revision   Λ_cell 0.000  Λ_rank 0.0000
    as-of 60 days after quarter-end
      naive     original      Λ_cell 1.000  Λ_rank 0.0308
      naive     no-revision   Λ_cell 1.000  Λ_rank 0.0178
      correct   original      Λ_cell 0.000  Λ_rank 0.0000
      restated  original      Λ_cell 0.954  Λ_rank 0.0128
      restated  no-revision   Λ_cell 0.000  Λ_rank 0.0000
    restated truncation: pass
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secpit import demo  # noqa: E402
from secpit.config import (  # noqa: E402
    BUSINESS_KEY, TAG_ASSETS, TAG_EQUITY, TAG_NET_INCOME, Config,
)


# 第二组 as-of 日期距季度末的天数：此时多数季报已经申报，发布滞后通道大幅收窄。
OFFSET_DAYS = 60


def _restated_table(db: demo.DataHandle, tag: str, qtrs: int, t) -> pl.DataFrame:
    """Period chosen as of t; value = latest version of that period in this world."""
    fr = db.frame(tag, qtrs)
    period = (fr.filter((pl.col("ddate") <= t) & (pl.col("filed") <= t))
                .group_by("cik").agg(pl.col("ddate").max()))
    return (fr.join(period, on=["cik", "ddate"], how="inner")
              .sort(["filed", "adsh"], descending=True)
              .group_by("cik").first()
              .select("cik", "ddate", "value")
              .collect())


def program_restated(db: demo.DataHandle, t, n: int) -> dict[int, float]:
    return demo._combine(
        _restated_table(db, TAG_ASSETS, 0, t),
        _restated_table(db, TAG_NET_INCOME, 4, t),
        _restated_table(db, TAG_EQUITY, 0, t),
        n,
    )


def freeze_first_filed(df: pl.DataFrame) -> pl.DataFrame:
    """Replace every value with the first-filed value of its business key."""
    key = list(BUSINESS_KEY)
    first = (df.sort(key + ["filed", "adsh"])
               .group_by(key, maintain_order=True)
               .agg(pl.col("value").first().alias("v0")))
    return (df.join(first, on=key, how="left")
              .with_columns(pl.col("v0").alias("value"))
              .drop("v0"))


def lam(cfg: Config, program, frame: pl.DataFrame, name: str) -> dict:
    lf = frame.lazy()
    panel = pl.concat([
        demo._panel(program, demo.world_pit, lf, cfg, name, "pit"),
        demo._panel(program, lambda f, _t: demo.world_rev(f), lf, cfg, name, "rev"),
    ])
    return demo.lambda_diff(panel, name, cfg.lambda_tol)


def main() -> int:
    cfg = Config.default(str(ROOT))
    df = demo._demo_frame(cfg).collect()
    norev = freeze_first_filed(df)

    rows = [
        ("naive", "original", demo.program_naive, df),
        ("naive", "no-revision", demo.program_naive, norev),
        ("correct", "original", demo.program_correct, df),
        ("restated", "original", program_restated, df),
        ("restated", "no-revision", program_restated, norev),
    ]
    for offset, label in ((0, "as-of at quarter-end"),
                          (OFFSET_DAYS, f"as-of {OFFSET_DAYS} days after quarter-end")):
        shifted = dataclasses.replace(cfg, asof_dates_override=tuple(
            d + dt.timedelta(days=offset) for d in cfg.asof_dates))
        print(label)
        for name, data, program, frame in rows:
            r = lam(shifted, program, frame, name)
            print(f"  {name:9s} {data:12s}  Λ_cell {r['lambda_cell']:.3f}  "
                  f"Λ_rank {r['lambda_rank']:.4f}")

    trunc = demo.truncation_test(cfg, df.lazy(), program_restated, "restated")
    print(f"restated truncation: {'pass' if trunc['truncation_pass'] else 'fail'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
