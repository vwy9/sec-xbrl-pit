"""重述普查的描述性统计。

每一项都按截断状态分四组报告：全体 / 剔右 / 剔左 / 两者都剔。
两种截断压低重述率的方向相同，只给一个混合数字无法判断偏差量级。
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from secpit.config import (BUSINESS_KEY, MIN_TAG_SAMPLE, SCALE_EXPONENTS,
                           THRESHOLDS, Config)
from secpit import store

log = logging.getLogger(__name__)

_KEY = list(BUSINESS_KEY)
_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

# 四组截断口径。值为施加在 census 上的过滤条件。
CENSORING = {
    "all":        None,
    "drop_right": lambda: ~pl.col("right_censored"),
    "drop_left":  lambda: ~pl.col("left_censored"),
    "drop_both":  lambda: ~pl.col("right_censored") & ~pl.col("left_censored"),
}


def _tag_events(cfg: Config, census: pl.LazyFrame,
                events: pl.LazyFrame) -> pl.LazyFrame:
    """把截断标记与观测次数附到事件表上，按 cik 取模分区做连接。

    直接 left join 要在 4177 万行的普查表上建哈希表（实测峰值 8.65GB）；
    按 cik 分区后每次哈希表只有约 1/N 的键，业务键含 cik，分区之间不会有
    同一个键，结果不变。产出只有 233 万行，物化后此后的分组都是对它的过滤。
    """
    n = max(1, cfg.key_partitions)
    cols = _KEY + ["n_obs", "right_censored", "left_censored"]
    parts = []
    for i in range(n):
        c = census.filter(pl.col("cik") % n == i).select(cols)
        e = events.filter(pl.col("cik") % n == i)
        parts.append(e.join(c, on=_KEY, how="left").collect(engine="streaming"))
    return pl.concat(parts).lazy()


def _groups(census: pl.LazyFrame, events: pl.LazyFrame):
    """逐组产出 (标签, 该组的 census 子集, 该组的 events 子集)，全部惰性。

    events 已由 `_tag_events` 带上截断标记，故这里对两侧施加同一个条件即可，
    不需要再做连接。
    """
    for label, cond in CENSORING.items():
        if cond is None:
            yield label, census, events
        else:
            yield label, census.filter(cond()), events.filter(cond())


def _rate_by_threshold(census: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """重述率 = 被重述的键数 / 有过两次以上观测的键数，四档阈值。"""
    rows = []
    for label, cen, ev in _groups(census, events):
        denom = cen.filter(pl.col("n_obs") >= 2).select(pl.len()).collect().item()
        ev_multi = ev.filter(pl.col("n_obs") >= 2)
        for thr in THRESHOLDS:
            hit = (
                ev_multi.filter(pl.col("rel_delta") > thr)
                        .select(_KEY).unique().select(pl.len()).collect().item()
            )
            rows.append({
                "censoring": label, "threshold": thr,
                "n_keys_multi": denom, "n_keys_revised": hit,
                "rate": hit / denom if denom else None,
            })
    return pl.DataFrame(rows)


def _quantile_table(lf: pl.LazyFrame, col: str, label: str, extra: dict) -> list[dict]:
    """在惰性帧上一次算完全部分位数，只把这几个标量拉回内存。"""
    got = lf.select(
        [pl.col(col).quantile(q).alias(f"q{i}") for i, q in enumerate(_QUANTILES)]
    ).collect().row(0)
    return [{**extra, "quantile": q, "value": v} for q, v in zip(_QUANTILES, got)]


def _magnitude_quantiles(census: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """首末相对幅度（magnitude）的分位数，分剔除与不剔除单位跃迁两组。

    这里必须用 census.magnitude 而不是 events.rel_delta：
    rel_delta 的分母是 max(|v_old|,|v_new|,1)，取值被构造性地限制在 [0, 2]，
    单位跃迁在其中体现不出来；magnitude 的分母是 max(|v_first|,1)，无上界，
    73.6 → 73,600,000 这类跃迁会给出约 1e6 的幅度，剔除前后差异才看得见。
    """
    rows = []
    for label, cen, _ in _groups(census, events):
        # 排除首值为 0 的键：那时 magnitude 无定义，兜底后退化为绝对变化，
        # 会以约 1% 的键主导整条尾巴。它们的数量在 census_composition 表中单独报告。
        revised = cen.filter(pl.col("revised") & ~pl.col("from_zero"))
        for scale_filter, sub in (
            ("with_scale", revised),
            ("without_scale", revised.filter(~pl.col("ever_scale_change"))),
        ):
            rows += _quantile_table(sub, "magnitude", label, {
                "censoring": label, "scale_filter": scale_filter,
                "n_keys": sub.select(pl.len()).collect().item(),
            })
    return pl.DataFrame(rows)


def _rel_delta_quantiles(census: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """单次变化的相对幅度分位数（事件层面，取值域 [0, 2]）。"""
    rows = []
    for label, _, ev in _groups(census, events):
        rows += _quantile_table(ev, "rel_delta", label,
                                {"censoring": label,
                                 "n_events": ev.select(pl.len()).collect().item()})
    return pl.DataFrame(rows)


_LAG_MEASURES = {
    "since_first": "lag_since_first_days",   # 距该事实首次发布
    "adjacent": "lag_days",                  # 距上一次观测
}


def _lag_quantiles(census: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """两种滞后口径的分位数，全体与按 qtrs 分别报告。

    按 qtrs 拆开是因为两种口径的差异几乎全在时点存量（qtrs=0）上：
    上年末余额在其后的季报里反复出现，相邻口径会把「公开一年后才改」
    量成「距上一份季报约 100 天」。
    """
    rows = []
    for label, cen, ev in _groups(census, events):
        for measure, col in _LAG_MEASURES.items():
            for q in ("all", 0, 1, 2, 3, 4):
                sub = ev if q == "all" else ev.filter(pl.col("qtrs") == q)
                rows += _quantile_table(sub, col, label, {
                    "censoring": label, "measure": measure, "qtrs": str(q),
                    "n_events": sub.select(pl.len()).collect().item(),
                })
    return pl.DataFrame(rows)


# 滞后分箱。窗口按披露周期取：下一份季报（60–129 天）、一年后的比较列（300–399 天）、
# 两年后的比较列（700–759 天；年报的利润表与现金流量表最多列三年）。
_LAG_BINS = (60, 130, 300, 400, 700, 760)
_LAG_LABELS = ("0-59", "60-129", "130-299", "300-399", "400-699", "700-759", "760+")


def _lag_bins(census: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """两种滞后口径下各分箱的事件占比，全体与按 qtrs 分别报告。

    分位数说不清分布的形状；峰在哪里、各占多少，要看分箱。
    """
    rows = []
    for label, _, ev in _groups(census, events):
        for measure, col in _LAG_MEASURES.items():
            for q in ("all", 0, 1, 2, 3, 4):
                sub = ev if q == "all" else ev.filter(pl.col("qtrs") == q)
                got = (sub.select(pl.col(col).cut(list(_LAG_BINS), labels=list(_LAG_LABELS),
                                                  left_closed=True).alias("bin"))
                          .group_by("bin").len().collect())
                n = int(got["len"].sum())
                counts = {str(b): int(c) for b, c in got.iter_rows()}
                for b in _LAG_LABELS:
                    rows.append({"censoring": label, "measure": measure, "qtrs": str(q),
                                 "bin_days": b, "n_events": counts.get(b, 0),
                                 "share": counts.get(b, 0) / n if n else None})
    return pl.DataFrame(rows)


def _rate_by_tag(census: pl.DataFrame, events: pl.DataFrame, top: int = 20) -> pl.DataFrame:
    """重述率按标签排名，取观测量最大的若干标签。

    回答的是"最常用的字段里哪个最容易被改"，不是"哪些字段最容易被改"，
    后者见 `_rate_by_tag_highest`。两者相差两倍以上，引用时不可混用。
    """
    frames = []
    for label, cen, ev in _groups(census, events):
        denom = (cen.filter(pl.col("n_obs") >= 2)
                    .group_by("tag").agg(pl.len().alias("n_keys_multi")))
        hit = (
            ev.filter(pl.col("n_obs") >= 2)
              .select(_KEY).unique()          # tag 已在业务键中
              .group_by("tag").agg(pl.len().alias("n_keys_revised"))
        )
        t = (
            denom.join(hit, on="tag", how="left")
                 .with_columns(pl.col("n_keys_revised").fill_null(0))
                 .with_columns(
                     (pl.col("n_keys_revised") / pl.col("n_keys_multi")).alias("rate"),
                     pl.lit(label).alias("censoring"),
                 )
                 .sort("n_keys_multi", descending=True)
                 .head(top)
                 .select("censoring", "tag", "n_keys_multi", "n_keys_revised", "rate")
                 .collect()
        )
        frames.append(t)
    return pl.concat(frames)


def _rate_by_tag_highest(census: pl.LazyFrame, events: pl.LazyFrame,
                        top: int = 20) -> pl.DataFrame:
    """重述率最高的字段排行——在全部标签上排序，不限于最常见的那些。

    实测与 `_rate_by_tag` 相差两倍以上：最常用字段里最高的约 15%，全局
    最高的是终止经营与房地产类科目，达 42%。说错口径会把结论讲反，
    故分成两张表各自落盘。低于 MIN_TAG_SAMPLE 的标签不参与排序。
    """
    frames = []
    for label, cen, _ in _groups(census, events):
        t = (
            cen.filter(pl.col("n_obs") >= 2)
               .group_by("tag")
               .agg(pl.len().alias("n_keys_multi"),
                    pl.col("revised").sum().alias("n_keys_revised"))
               .filter(pl.col("n_keys_multi") >= MIN_TAG_SAMPLE)
               .with_columns(
                   (pl.col("n_keys_revised") / pl.col("n_keys_multi")).alias("rate"),
                   pl.lit(label).alias("censoring"),
               )
               .sort("rate", descending=True)
               .head(top)
               .select("censoring", "tag", "n_keys_multi", "n_keys_revised", "rate")
               .collect(engine="streaming")
        )
        frames.append(t)
    return pl.concat(frames)


def _silent_vs_amendment(census: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """静默重分类占比——后一次并非修正案的事件占全部事件的比例。

    这是本项目最主要的一个数字：若绝大多数重述不走正式修正案，
    则「只看 10-K/A 就能识别重述」这个常见做法是失效的。
    """
    rows = []
    for label, cen, ev in _groups(census, events):
        agg = ev.select(pl.len().alias("n"),
                        pl.col("is_amendment").sum().alias("a")).collect().row(0)
        n, amend = int(agg[0]), int(agg[1] or 0)
        rows += [
            {"censoring": label, "category": "silent", "n_events": n - amend,
             "share": (n - amend) / n if n else None},
            {"censoring": label, "category": "amendment", "n_events": amend,
             "share": amend / n if n else None},
        ]
    return pl.DataFrame(rows)


def _version_and_scale_share(census: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for label, cen, ev in _groups(census, events):
        agg = ev.select(
            pl.len().alias("n"),
            pl.col("version_changed").sum().alias("vc"),
            (pl.col("scale_exponent").is_not_null()
             & pl.col("scale_exponent").cast(pl.Int32).is_in(list(SCALE_EXPONENTS)))
            .sum().alias("sc"),
        ).collect().row(0)
        n, vc, sc = int(agg[0]), int(agg[1] or 0), int(agg[2] or 0)
        for cat, cnt in (("version_changed", vc), ("scale_change", sc)):
            rows.append({"censoring": label, "category": cat, "n_events": cnt,
                         "share": cnt / n if n else None})
    return pl.DataFrame(rows)


def _census_composition(census: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """被重述键的构成，说明幅度统计排除了哪些、为什么。"""
    rows = []
    for label, cen, _ in _groups(census, events):
        rv = cen.filter(pl.col("revised"))
        agg = rv.select(
            pl.len().alias("n"),
            pl.col("from_zero").sum().alias("fz"),
            pl.col("ever_scale_change").sum().alias("sc"),
            pl.col("ever_amendment").sum().alias("am"),
            (~pl.col("from_zero")).sum().alias("keep"),
        ).collect().row(0)
        n = int(agg[0])
        for cat, cnt in (
            ("revised_total", n),
            ("from_zero", int(agg[1] or 0)),
            ("ever_scale_change", int(agg[2] or 0)),
            ("ever_amendment", int(agg[3] or 0)),
            ("in_magnitude_stats", int(agg[4] or 0)),
        ):
            rows.append({"censoring": label, "category": cat, "n_keys": cnt,
                         "share_of_revised": cnt / n if n else None})
    return pl.DataFrame(rows)


def _left_censoring_sources(census: pl.LazyFrame) -> pl.DataFrame:
    """左截断的两个来源各有多少键、各自的重述率（分母同为观测不少于两次的键）。

    比较列来源的重述率低于其余键，说明丢失原始申报会漏掉一部分重述，
    左截断压低重述率的方向与右截断相同。
    """
    return (
        census.filter(pl.col("n_obs") >= 2)
              .group_by("left_censored_window", "first_is_comparative")
              .agg(pl.len().alias("n_keys_multi"),
                   pl.col("revised").sum().alias("n_keys_revised"))
              .with_columns((pl.col("n_keys_revised") / pl.col("n_keys_multi")).alias("rate"))
              .sort("left_censored_window", "first_is_comparative")
              .collect(engine="streaming")
    )


def _accepted_after_1600_share(cfg: Config) -> pl.DataFrame:
    """受理时刻晚于 16:00 的 filing 占比。

    量化「可见性以日粒度的 filed 为准」这个近似的影响面：16:00 之后受理的
    文件当天实际不可交易。这是 filing 层面的属性，与截断状态无关，不分组。
    """
    # 逐年份分区统计再累加。一份 filing 只在一个季度受理，因此只落在一个
    # filed_year 分区里，跨分区不会有同一个 adsh，所以这个拆分是精确的。
    # 直接对 8100 万行做 unique() 去重得到 40 万份 filing 需要 8GB——
    # 这是整个统计层唯一的内存热点。
    n = late = rolled = affected = 0
    for part in store.year_partitions(cfg):
        got = (
            pl.scan_parquet(str(part / "*" / "*.parquet"))
            .select("adsh", "filed", "accepted")
            .unique()
            .with_columns(
                (pl.col("accepted").dt.hour() >= 16).alias("late"),
                (pl.col("accepted").dt.date() < pl.col("filed")).alias("rolled"),
            )
            .select(
                pl.len().alias("n"),
                pl.col("late").sum().alias("late"),
                pl.col("rolled").sum().alias("rolled"),
                (pl.col("late") & ~pl.col("rolled")).sum().alias("affected"),
            )
            .collect(engine="streaming")
            .row(0)
        )
        n += int(got[0]); late += int(got[1])
        rolled += int(got[2]); affected += int(got[3])

    return pl.DataFrame([{
        "censoring": "n/a",
        "n_filings": n,
        "n_after_1600": late,
        "share_after_1600": late / n if n else None,
        # EDGAR 对晚于截止时间受理的申报顺延日期，这批的 filed 已经是次日，不受影响
        "n_rolled_next_day": rolled,
        # 真正受 N6 近似影响的：收盘后受理但 filed 仍标当天，盘中实际不可得
        "n_same_day_after_1600": affected,
        "share_affected": affected / n if n else None,
    }])


def compute(cfg: Config) -> dict[str, Path]:
    cfg.ensure_dirs()
    census = pl.scan_parquet(cfg.out_dir / "census.parquet")
    events = _tag_events(cfg, census, pl.scan_parquet(cfg.out_dir / "revisions.parquet"))

    tables = {
        "rate_by_threshold": _rate_by_threshold(census, events),
        "magnitude_quantiles": _magnitude_quantiles(census, events),
        "rel_delta_quantiles": _rel_delta_quantiles(census, events),
        "census_composition": _census_composition(census, events),
        "left_censoring_sources": _left_censoring_sources(census),
        "lag_quantiles": _lag_quantiles(census, events),
        "lag_bins": _lag_bins(census, events),
        "rate_by_tag": _rate_by_tag(census, events),
        "rate_by_tag_highest": _rate_by_tag_highest(census, events),
        "silent_vs_amendment": _silent_vs_amendment(census, events),
        "version_and_scale_share": _version_and_scale_share(census, events),
        "accepted_after_1600_share": _accepted_after_1600_share(cfg),
    }

    paths = {}
    for name, table in tables.items():
        path = cfg.out_dir / f"stats_{name}.csv"
        table.write_csv(path)
        paths[name] = path
        log.info("stats_%s → %d 行", name, table.height)
    return paths
