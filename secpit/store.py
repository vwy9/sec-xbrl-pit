"""观测事实表的读入口，以及 D_PIT / D_rev 两个世界。

as_of 与 latest 共用「按业务键取受理次序最晚」的逻辑，差别仅在前者先施加
filed <= t。排序键固定为 (filed DESC, adsh DESC)：filed 是全库统一的可见性
口径，adsh 作次级键保证同一天多份 filing 时次序确定。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import polars as pl

from secpit.config import BUSINESS_KEY, PRIMARY_KEY, Config

log = logging.getLogger(__name__)

_SORT_KEYS = ["filed", "adsh"]


class PrimaryKeyViolation(RuntimeError):
    """观测表主键不唯一。"""


def load(cfg: Config,
         ciks: list[int] | None = None,
         tags: list[str] | None = None,
         ddate_max: dt.date | None = None,
         qtrs: int | None = None) -> pl.LazyFrame:
    """惰性读取观测表，把过滤条件下推到扫描层。"""
    if not any(cfg.observations_dir.rglob("*.parquet")):
        raise FileNotFoundError(
            f"{cfg.observations_dir} 下没有分区文件，请先执行 build"
        )
    lf = pl.scan_parquet(cfg.observations_glob)
    if ciks is not None:
        lf = lf.filter(pl.col("cik").is_in(ciks))
    if tags is not None:
        lf = lf.filter(pl.col("tag").is_in(tags))
    if ddate_max is not None:
        lf = lf.filter(pl.col("ddate") <= ddate_max)
    if qtrs is not None:
        lf = lf.filter(pl.col("qtrs") == qtrs)
    return lf


def _pick_latest(lf: pl.LazyFrame) -> pl.LazyFrame:
    """每个业务键取受理次序最晚的一条观测。

    结果保留 filed 与 adsh，使「这个数字是哪天、由哪份申报带来的」可追溯。
    """
    return (
        lf.sort(_SORT_KEYS, descending=True)
          .group_by(list(BUSINESS_KEY), maintain_order=False)
          .first()
    )


def as_of(lf: pl.LazyFrame, t: dt.date) -> pl.LazyFrame:
    """D_PIT(t)：只用截至 t 已受理的申报，取其中最新的一版。"""
    return _pick_latest(lf.filter(pl.col("filed") <= t))


def latest(lf: pl.LazyFrame) -> pl.LazyFrame:
    """D_rev：不带时点约束的全历史最新值，含此后发布的全部修订。"""
    return _pick_latest(lf)


def diff_worlds(cfg: Config, t: dt.date, **kw) -> pl.DataFrame:
    """两个世界在 t 时刻取值不同的业务键。"""
    lf = load(cfg, **kw)
    pit = as_of(lf, t).select(
        *BUSINESS_KEY,
        pl.col("value").alias("value_pit"),
        pl.col("filed").alias("filed_pit"),
        pl.col("adsh").alias("adsh_pit"),
    )
    rev = latest(lf).select(
        *BUSINESS_KEY,
        pl.col("value").alias("value_rev"),
        pl.col("filed").alias("filed_rev"),
        pl.col("adsh").alias("adsh_rev"),
    )
    joined = pit.join(rev, on=list(BUSINESS_KEY), how="inner")
    return joined.filter(
        pl.col("value_pit").ne_missing(pl.col("value_rev"))
    ).collect()


def year_partitions(cfg: Config) -> list[Path]:
    """观测表的年份分区目录，按年份升序。"""
    return sorted(
        (d for d in cfg.observations_dir.glob("filed_year=*") if d.is_dir()),
        key=lambda d: int(d.name.split("=")[1]),
    )


def assert_primary_key(cfg: Config) -> int:
    """断言 (adsh, tag, ddate, qtrs, uom) 全局唯一，返回最大组大小。

    逐年份分区独立检验，这是精确的：主键含 adsh，一份 filing 只在一个季度受理，
    因此只会落在一个 filed_year 分区里，跨分区不可能出现同一个 adsh。
    全量窗口下观测表有 8100 万行，一次性按主键分组需要约 8.8GB，
    逐分区则把峰值压到单个年份的规模。
    """
    for part in year_partitions(cfg):
        dup = (
            pl.scan_parquet(str(part / "*" / "*.parquet"))
            .select(list(PRIMARY_KEY))
            .group_by(list(PRIMARY_KEY))
            .len()
            .filter(pl.col("len") > 1)
            .sort("len", descending=True)
            .head(10)
            .collect(engine="streaming")
        )
        if dup.height:
            raise PrimaryKeyViolation(
                f"{part.name} 主键不唯一，最大组 {dup['len'][0]}，"
                f"前若干冲突键：\n{dup}"
            )
    return 1


def summary(cfg: Config) -> dict:
    """观测表的规模与跨度，写入 _summary.json。"""
    lf = pl.scan_parquet(cfg.observations_glob)
    agg = lf.select(
        pl.len().alias("rows"),
        pl.col("cik").n_unique().alias("companies"),
        pl.col("tag").n_unique().alias("tags"),
        pl.col("adsh").n_unique().alias("filings"),
        pl.col("filed").min().alias("filed_min"),
        pl.col("filed").max().alias("filed_max"),
        pl.col("ddate").min().alias("ddate_min"),
        pl.col("ddate").max().alias("ddate_max"),
    ).collect(engine="streaming").to_dicts()[0]

    # 业务键去重必须跨年份进行（同一个键会在多个年份被反复报告），
    # 但可以按 cik 取模分区——业务键含 cik，分区之间不会有同一个键。
    n_parts = max(1, cfg.key_partitions)
    agg["business_keys"] = sum(
        lf.filter(pl.col("cik") % n_parts == i)
          .select(list(BUSINESS_KEY)).unique()
          .select(pl.len()).collect(engine="streaming").item()
        for i in range(n_parts)
    )
    agg["max_group_size"] = assert_primary_key(cfg)
    agg["by_form"] = (
        lf.group_by("form").len().sort("len", descending=True)
          .collect(engine="streaming").to_dicts()
    )
    agg["quarters"] = list(cfg.quarters)

    out = {k: (str(v) if isinstance(v, dt.date) else v) for k, v in agg.items()}
    path = cfg.parquet_dir / "_summary.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("summary → %s", path)
    return out
