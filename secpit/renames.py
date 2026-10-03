"""从数据本身推导 us-gaap 概念的改名关系。

业务键含 tag，概念被改名后旧键停止、新键开始，在本库看来是两个不相干的键，
「旧概念报过、新概念在次年可比列里重报同一期」的重述会整个漏掉。实测
SalesRevenueNet 2013 年被报 33,424 次、2022 年归零，同期
RevenueFromContractWithCustomerExcludingAssessedTax 从 0 升到 34,391 次；
收入这种最常用的字段整个被覆盖，不修正的重述率只是下界。

从数据推导而不查 FASB 的 deprecated/replacement 对照，一是后者没有稳定的
机器可读发布，二是准则说的是「应该怎么改」，要修正的是「申报人实际怎么改的」。

判据，对候选对 (A, B)：
1. A 是濒死标签：历史峰值足够高，近三年用量跌到峰值的很小比例；
2. B 在 A 报过的那些 (cik, ddate, qtrs, uom) 上、在 A 最后一次申报之后接手；
3. 主判据：同一个键上 A 与 B 的取值大量精确相同。早先版本用的是
   「几乎从不在同一份申报里同时出现」，但共现率把「语义互为替代」和
   「时间上根本不重叠」混为一谈——CostOfGoodsSold 2022 年就消失、
   RevenueFromContractWithCustomer 2018 年才出现，共现率天然接近 0，
   尽管语义毫无关系。取值一致性分离得很干净：真改名 35%–92% 精确一致、
   相对差中位数 <= 0.10；假阳性约 0.1% 一致、中位差 >= 0.45；
4. 支持该改名的公司数达到下限。

共现率仍计算并记录，只作诊断量。
产出是等价类而非有向对——A→B→C 这样的链要并成一个类。
"""

from __future__ import annotations

import logging

import polars as pl

from secpit.config import Config

log = logging.getLogger(__name__)

# 候选濒死标签的历史峰值下限。太低的标签本就噪声大。
MIN_PEAK = 2000
# 近三年用量低于峰值的这个比例，才算濒死。
DECAY_RATIO = 0.02
# 判定为改名所需的最低"取值精确一致"比例。
# 实测真改名在 35%–92% 之间，假阳性在 0.1% 以下，取 20% 两侧都有充裕余量。
MIN_AGREEMENT = 0.20
# 判定为改名所允许的最大"取值相对差中位数"。
# 实测真改名 ≤0.10，假阳性 ≥0.45。
MAX_MEDIAN_REL = 0.20
# 参与一致性检验所需的最少共同键数，低于此数统计量不稳。
MIN_SHARED_KEYS = 200
# 支持一条改名所需的最少公司数。
MIN_SUPPORT_CIKS = 50
# 最近三年的定义。
RECENT_YEARS = (2024, 2025, 2026)

_KEY_NO_TAG = ["cik", "ddate", "qtrs", "uom"]


def _usage_by_year(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.select(pl.col("filed").dt.year().alias("y"), "tag")
          .group_by("y", "tag").len()
          .collect(engine="streaming")
    )


def dying_tags(usage: pl.DataFrame) -> pl.DataFrame:
    """历史用量可观、但近三年基本消失的标签。"""
    peak = usage.group_by("tag").agg(pl.col("len").max().alias("peak"))
    recent = (usage.filter(pl.col("y").is_in(list(RECENT_YEARS)))
                   .group_by("tag").agg(pl.col("len").sum().alias("recent")))
    return (
        peak.join(recent, on="tag", how="left").fill_null(0)
            .filter((pl.col("peak") >= MIN_PEAK)
                    & (pl.col("recent") < DECAY_RATIO * pl.col("peak")))
            .sort("peak", descending=True)
    )


def _cooccurrence(lf: pl.LazyFrame, a: str, b: str) -> float:
    """在两者中较罕见者出现的那些申报里，另一个也出现的比例。

    分母必须是 min(count_A, count_B)，不能是「A 或 B 出现的申报数」。
    用后者会把罕见标签对无处不在标签的共现率稀释到接近 0：
    LicensesRevenue 出现在约 2 千份申报、经营现金流出现在约 30 万份，
    两者几乎每次同现，但 2000/300000 = 0.007 会误判成「从不同现」，
    于是把「经营现金流」判成「许可收入」的改名——这是个荒谬的结论。
    以罕见者为分母时该比值接近 1.0，正确地被排除。
    """
    got = (
        lf.filter(pl.col("tag").is_in([a, b]))
          .select("adsh", "tag").unique()
          .group_by("adsh").agg(pl.col("tag").n_unique().alias("k"))
          .select(
              (pl.col("k") >= 1).sum().alias("n_any"),
              (pl.col("k") == 2).sum().alias("n_both"),
          )
          .collect(engine="streaming").row(0)
    )
    n_both = int(got[1])
    counts = (
        lf.filter(pl.col("tag").is_in([a, b]))
          .select("adsh", "tag").unique()
          .group_by("tag").agg(pl.len().alias("n"))
          .collect(engine="streaming")
    )
    if counts.height < 2:
        return 1.0
    denom = int(counts["n"].min())
    return n_both / denom if denom else 1.0


def _agreement(lf: pl.LazyFrame, a: str, b: str) -> tuple[int, float, float]:
    """A 与 B 在同一个 (cik, ddate, qtrs, uom) 上的取值一致程度。

    返回 (共同键数, 精确一致比例, 相对差中位数)。
    各自取该键上受理次序最晚的一版——比较的是"两个标签各自的最终说法"。
    """
    def last_value(tag: str, alias: str) -> pl.LazyFrame:
        return (lf.filter(pl.col("tag") == tag).sort("filed")
                  .group_by(_KEY_NO_TAG)
                  .agg(pl.col("value").last().alias(alias)))

    j = (last_value(a, "va").join(last_value(b, "vb"), on=_KEY_NO_TAG, how="inner")
           .filter(pl.col("va").is_not_null() & pl.col("vb").is_not_null()))
    den = pl.max_horizontal(pl.col("va").abs(), pl.col("vb").abs(), pl.lit(1.0))
    got = (j.with_columns(((pl.col("vb") - pl.col("va")).abs() / den).alias("rel"))
             .select(pl.len().alias("n"),
                     (pl.col("rel") < 1e-9).sum().alias("exact"),
                     pl.col("rel").median().alias("med"))
             .collect(engine="streaming").row(0))
    n = int(got[0])
    if not n:
        return 0, 0.0, 1.0
    return n, int(got[1]) / n, float(got[2] if got[2] is not None else 1.0)


def _successors(lf: pl.LazyFrame, a: str, rising: set[str],
                top: int = 5) -> pl.DataFrame:
    """在 A 报过的那些键上、A 最后一次申报之后接手的标签，按支持公司数排序。

    只保留自身在上升的标签，而且必须先过滤再截断。
    反过来的话，前几名会被 NetIncomeLoss、Assets 这类几乎每份申报都有的标签占满，
    它们随后被上升过滤器剔除，真正的接手者根本进不了候选——
    实测那样会漏掉 SalesRevenueNet → RevenueFromContractWithCustomer… 这条已知为真的改名。
    """
    a_keys = (
        lf.filter(pl.col("tag") == a)
          .group_by(_KEY_NO_TAG)
          .agg(pl.col("filed").max().alias("a_last"))
    )
    return (
        lf.join(a_keys, on=_KEY_NO_TAG, how="inner")
          .filter((pl.col("tag") != a)
                  & (pl.col("filed") > pl.col("a_last"))
                  & pl.col("tag").is_in(list(rising)))
          .select("tag", "cik").unique()
          .group_by("tag").agg(pl.len().alias("support_ciks"))
          .sort("support_ciks", descending=True)
          .head(top)
          .collect(engine="streaming")
    )


def rising_tags(usage: pl.DataFrame) -> set[str]:
    """近三年用量明显高于早期的标签。

    真正的接手者自身必须在上升。不加这一条的话，任何一个长期稳定、
    到处都在用的标签都会被当成接手者。
    """
    early = (usage.filter(pl.col("y") <= 2013)
                  .group_by("tag").agg(pl.col("len").sum().alias("early")))
    recent = (usage.filter(pl.col("y").is_in(list(RECENT_YEARS)))
                   .group_by("tag").agg(pl.col("len").sum().alias("recent")))
    t = recent.join(early, on="tag", how="left").fill_null(0)
    grew = t.filter((pl.col("recent") >= MIN_PEAK)
                    & (pl.col("recent") > 3 * pl.col("early")))
    return set(grew["tag"].to_list())


def detect(cfg: Config, write: bool = True) -> pl.DataFrame:
    """产出改名候选表：每行一条 A→B，含支持公司数与共现率。"""
    lf = pl.scan_parquet(cfg.observations_glob)
    usage = _usage_by_year(lf)
    dying = dying_tags(usage)
    rising = rising_tags(usage)
    log.info("上升标签 %d 个", len(rising))
    log.info("濒死标签 %d 个（峰值 >= %d 且近三年 < %.0f%% 峰值）",
             dying.height, MIN_PEAK, 100 * DECAY_RATIO)

    rows = []
    for a, peak, recent in dying.iter_rows():
        for b, support in _successors(lf, a, rising).iter_rows():
            if support < MIN_SUPPORT_CIKS:
                continue
            n_shared, exact, med = _agreement(lf, a, b)
            accepted = (n_shared >= MIN_SHARED_KEYS
                        and exact >= MIN_AGREEMENT
                        and med <= MAX_MEDIAN_REL)
            rows.append({
                "old_tag": a, "new_tag": b,
                "old_peak": peak, "old_recent": recent,
                "support_ciks": support,
                "shared_keys": n_shared,
                "exact_agreement": exact,
                "median_rel_diff": med,
                "cooccurrence": _cooccurrence(lf, a, b),   # 仅诊断
                "accepted": accepted,
            })
        log.debug("已检查 %s", a)

    out = (pl.DataFrame(rows) if rows else
           pl.DataFrame(schema={"old_tag": pl.Utf8, "new_tag": pl.Utf8,
                                "old_peak": pl.UInt32, "old_recent": pl.UInt32,
                                "support_ciks": pl.UInt32, "shared_keys": pl.UInt32,
                                "exact_agreement": pl.Float64,
                                "median_rel_diff": pl.Float64,
                                "cooccurrence": pl.Float64, "accepted": pl.Boolean}))
    if out.height:
        out = out.sort(["accepted", "exact_agreement"], descending=[True, True])
    if write:
        cfg.ensure_dirs()
        path = cfg.out_dir / "tag_renames.csv"
        out.write_csv(path)
        n_ok = int(out["accepted"].sum()) if out.height else 0
        log.info("tag_renames → %s（候选 %d，采纳 %d）", path, out.height, n_ok)
    return out


def equivalence_classes(renames: pl.DataFrame) -> dict[str, str]:
    """把采纳的改名对合并成等价类，返回 标签 → 类代表 的映射。

    A→B→C 这样的链必须合并成一个类，否则跨两次改名的重述仍会漏掉。
    类代表取该类中字典序最大者——通常是较新的概念名，且与输入顺序无关。
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)     # 先并到字典序小的，最后统一取大的

    if renames.height:
        for a, b in renames.filter(pl.col("accepted")).select("old_tag", "new_tag").iter_rows():
            union(a, b)

    members: dict[str, list[str]] = {}
    for tag in parent:
        members.setdefault(find(tag), []).append(tag)
    return {t: max(group) for group in members.values() for t in group}
