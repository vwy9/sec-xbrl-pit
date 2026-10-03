"""vintage 微分演示：两个世界 × 三个程序 × 两个检验。

要演示的命题：截断检验检查的是「函数 f 对输入张量 D 是因果的」，即
y_t = f(D_0:t)；它不约束 D 本身是否 F_t-可测。修订改写的是过去而不是未来，
所以用了修订后数据的程序能原样通过截断检验。

构造上，两个世界是同一张观测表的两个子集，程序拿到的是一个数据句柄，
句柄背后是哪个世界由本模块决定，程序不知道：

    world_pit(t) = 观测表中 filed <= t 的子集
    world_rev    = 观测表全部

自行按 filed 过滤的程序在两个世界里看到的东西完全一样（Λ = 0），
不过滤的程序在 D_rev 里会看到后来才发布的修订值（Λ > 0）。
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import polars as pl

from secpit.config import DEMO_TAGS, TAG_ASSETS, TAG_EQUITY, TAG_NET_INCOME, Config
from secpit import store

log = logging.getLogger(__name__)

_EPS = 1e-12


class DataHandle:
    """只读数据句柄，三个接口的可得性刻意不同：

      frame()   原始表，带 filed 列——想做对就得自己用它
      latest()  预先取好每键最新值的便利视图，不返回 filed 列   ← 陷阱
      as_of()   按 filed 过滤后取最新的辅助函数                  ← 正确路径

    latest() 只按报告期取最新一版，不带受理日约束——按报告期对齐时最顺手的那个接口。
    """

    def __init__(self, lf: pl.LazyFrame, world: str):
        self._lf = lf
        self.world = world

    def frame(self, tag: str | None = None, qtrs: int | None = None) -> pl.LazyFrame:
        lf = self._lf
        if tag is not None:
            lf = lf.filter(pl.col("tag") == tag)
        if qtrs is not None:
            lf = lf.filter(pl.col("qtrs") == qtrs)
        return lf

    @staticmethod
    def _newest_period_per_cik(lf: pl.LazyFrame) -> pl.LazyFrame:
        """每个 cik 取 ddate 最大的一期；同一期有多版时取 filed 最晚的那版。"""
        return (
            lf.sort(["ddate", "filed", "adsh"], descending=True)
              .group_by("cik", maintain_order=False)
              .first()
        )

    def latest(self, tag: str, qtrs: int, ddate_max: dt.date) -> pl.DataFrame:
        """便利视图：不带 filed 约束，也不返回 filed 列。"""
        lf = self.frame(tag, qtrs).filter(pl.col("ddate") <= ddate_max)
        return (
            self._newest_period_per_cik(lf)
            .select("cik", "ddate", "value")        # 刻意不给 filed
            .collect()
        )

    def as_of(self, tag: str, qtrs: int, date: dt.date) -> pl.DataFrame:
        """正确路径：先按 filed <= date 过滤，再取最近一期。"""
        lf = (
            self.frame(tag, qtrs)
            .filter(pl.col("ddate") <= date)
            .filter(pl.col("filed") <= date)
        )
        return (
            self._newest_period_per_cik(lf)
            .select("cik", "ddate", "value", "filed", "adsh")
            .collect()
        )


def world_pit(lf: pl.LazyFrame, t: dt.date) -> DataHandle:
    """D_PIT(t)：只含截至 t 已受理的申报。"""
    return DataHandle(lf.filter(pl.col("filed") <= t), "pit")


def world_rev(lf: pl.LazyFrame) -> DataHandle:
    """D_rev：全历史，含此后发布的全部修订。"""
    return DataHandle(lf, "rev")


# 三个程序，同一个任务：给定 as-of 日期 t，在按 t 时刻可见总资产取前 N 家
# 公司的样本上，计算 ROE = 年度净利润 / 期末股东权益 的截面标准化得分。

def _zscore(df: pl.DataFrame) -> dict[int, float]:
    v = df["roe"].to_numpy()
    if len(v) < 2:
        return {}
    sd = v.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return {}
    z = (v - v.mean()) / sd
    return {int(c): float(x) for c, x in zip(df["cik"].to_list(), z)}


def _combine(assets: pl.DataFrame, ni: pl.DataFrame, eq: pl.DataFrame,
             n: int) -> dict[int, float]:
    universe = (
        assets.filter(pl.col("value").is_not_null() & (pl.col("value") > 0))
              .sort("value", descending=True)
              .head(n)
              .select("cik")
    )
    df = (
        universe
        .join(ni.select("cik", pl.col("value").alias("ni")), on="cik", how="inner")
        .join(eq.select("cik", pl.col("value").alias("eq")), on="cik", how="inner")
        .filter(pl.col("eq").is_not_null() & (pl.col("eq") != 0)
                & pl.col("ni").is_not_null())
        .with_columns((pl.col("ni") / pl.col("eq")).alias("roe"))
        .filter(pl.col("roe").is_finite())
        .sort("cik")
    )
    return _zscore(df)


def program_correct(db: DataHandle, t: dt.date, n: int) -> dict[int, float]:
    """vintage-正确：自行施加 filed <= t，因此在两个世界里看到的完全一样。"""
    return _combine(
        db.as_of(TAG_ASSETS, 0, t),
        db.as_of(TAG_NET_INCOME, 4, t),
        db.as_of(TAG_EQUITY, 0, t),
        n,
    )


def program_naive(db: DataHandle, t: dt.date, n: int) -> dict[int, float]:
    """vintage-疏忽：只按报告期对齐，不按公告日对齐。

    这是从业者常犯的一种错法。关键在于它同样不读取 ddate > t 的事实，
    因此会通过截断检验。
    """
    return _combine(
        db.latest(TAG_ASSETS, 0, t),
        db.latest(TAG_NET_INCOME, 4, t),
        db.latest(TAG_EQUITY, 0, t),
        n,
    )


def program_lookahead(db: DataHandle, t: dt.date, n: int) -> dict[int, float]:
    """对照组：故意取用 t 之后一年内的期数，且不按 filed 过滤。

    它存在的目的是证明截断检验不是空转的：没有一个会失败的对照，
    「correct 与 naive 都通过了截断检验」无法排除「截断检验根本没在工作」。

    顺带一个经验规律：按 filed <= t 过滤在实践中蕴含了 ddate <= t——期间几乎
    不可能在结束前被申报（三个演示字段的 347 万行里有 275 行期末晚于受理日）。
    因此自行按 filed 过滤的程序在实践中都会通过截断检验；要构造一个截断失败的
    程序就得放弃 filed 约束，而那通常也让版本微分失败。2×2 表里
    「截断失败、版本微分通过」这格在本任务上为空，但不是逻辑上不可能；
    「截断通过、版本微分失败」那格却是可达的。
    """
    future = t + dt.timedelta(days=365)
    return _combine(
        db.latest(TAG_ASSETS, 0, future),
        db.latest(TAG_NET_INCOME, 4, future),
        db.latest(TAG_EQUITY, 0, future),
        n,
    )


PROGRAMS = {
    "correct": program_correct,
    "naive": program_naive,
    "lookahead": program_lookahead,
}


# 秩相关（numpy 自实现，避免为一个函数引入 scipy）

def _rankdata(a: np.ndarray) -> np.ndarray:
    """并列取平均秩。"""
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    srt = a[sorter]
    obs = np.r_[True, srt[1:] != srt[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    n = len(a)
    ra, rb = _rankdata(a), _rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")

    # 无并列时用教科书的精确公式 ρ = 1 − 6Σd² / (n(n²−1))。
    # 两组秩相同时 d 恒为 0，结果精确等于 1.0；
    # 若改用秩上的 Pearson（无论手算还是 corrcoef），末位误差会给出
    # 0.9999999999999999，于是 1 − ρ 成为一个非零的假 Λ_rank。
    if len(np.unique(ra)) == n and len(np.unique(rb)) == n:
        d = ra - rb
        rho = 1.0 - 6.0 * float((d * d).sum()) / (n * (n * n - 1))
    else:                                     # 有并列时退回秩上的 Pearson
        rho = float(np.corrcoef(ra, rb)[0, 1])
    return min(1.0, max(-1.0, rho))           # ρ 数学上有界于 [-1, 1]


def _panel(program, make_world, lf: pl.LazyFrame, cfg: Config,
           program_name: str, world_name: str) -> pl.DataFrame:
    rows = []
    for t in cfg.asof_dates:
        db = make_world(lf, t)
        for cik, value in program(db, t, cfg.universe_size).items():
            rows.append({"program": program_name, "world": world_name,
                         "asof": t, "cik": cik, "value": value})
    schema = {"program": pl.Utf8, "world": pl.Utf8, "asof": pl.Date,
              "cik": pl.Int64, "value": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def truncation_test(cfg: Config, lf: pl.LazyFrame, program, name: str) -> dict:
    """把库按 ddate <= T 截断，检查 t <= T 的输出在重叠区间上精确相等。

    捕捉的是「函数偷看未来期数」。按 ddate 而非 filed 截断——
    按 filed 截断会与版本微分混淆。
    """
    failures = []
    for T in cfg.truncation_points:
        trunc = lf.filter(pl.col("ddate") <= T)
        for t in cfg.asof_dates:
            if t > T:
                continue
            full_out = program(world_rev(lf), t, cfg.universe_size)
            trunc_out = program(world_rev(trunc), t, cfg.universe_size)
            if full_out.keys() != trunc_out.keys() or any(
                full_out[k] != trunc_out[k] for k in full_out
            ):
                failures.append({"truncation": str(T), "asof": str(t)})
    return {"program": name, "truncation_pass": not failures, "failures": failures}


def lambda_diff(panel: pl.DataFrame, program: str, tol: float) -> dict:
    """两个世界之间的版本微分。

    Λ_cell 为面板中相对差异超过容差的单元格占比；
    只在一个世界中出现的单元格同样计为差异——覆盖面不同本身就是差异。
    Λ_rank 为各 as-of 日期上 (1 − 截面 Spearman 相关) 的均值。
    """
    p = panel.filter(pl.col("program") == program)
    pit = p.filter(pl.col("world") == "pit").select("asof", "cik", pl.col("value").alias("y_pit"))
    rev = p.filter(pl.col("world") == "rev").select("asof", "cik", pl.col("value").alias("y_rev"))
    joined = pit.join(rev, on=["asof", "cik"], how="full", coalesce=True)

    n_cells = joined.height
    if n_cells == 0:
        return {"program": program, "lambda_cell": None, "lambda_rank": None,
                "n_cells": 0, "n_diff": 0}

    diff = joined.with_columns(
        pl.when(pl.col("y_pit").is_null() | pl.col("y_rev").is_null())
          .then(True)
          .otherwise(
              ((pl.col("y_pit") - pl.col("y_rev")).abs()
               / (pl.col("y_pit").abs() + _EPS)) > tol
          ).alias("differs")
    )
    n_diff = int(diff["differs"].sum())

    rank_terms = []
    both = joined.filter(pl.col("y_pit").is_not_null() & pl.col("y_rev").is_not_null())
    for (asof,), grp in both.group_by(["asof"], maintain_order=True):
        if grp.height >= 2:
            rho = _spearman(grp["y_pit"].to_numpy(), grp["y_rev"].to_numpy())
            if np.isfinite(rho):
                rank_terms.append(1.0 - rho)

    return {
        "program": program,
        "lambda_cell": n_diff / n_cells,
        "lambda_rank": float(np.mean(rank_terms)) if rank_terms else 0.0,
        "n_cells": n_cells,
        "n_diff": n_diff,
    }


def _verdict(truncation_pass: bool, vintage_pass: bool) -> str:
    if truncation_pass and vintage_pass:
        return "clean"
    if truncation_pass and not vintage_pass:
        return "silent_leak"          # 截断检验放过的那一格
    if not truncation_pass and vintage_pass:
        return "obvious_bug"
    return "double_failure"


def _demo_frame(cfg: Config) -> pl.LazyFrame:
    """把演示用到的三个标签一次性物化，再交给两个世界。

    演示要跑 3 个程序 × 2 个世界 × N 个 as-of 日期，加上截断检验的重复调用，
    总计约 1100 次查询。若每次都去扫 69 个季度的分区文件，代价全在 IO 上。
    这三个标签的数据量很小（百万行级），一次读进内存即可。
    结果与逐次扫描完全相同——只是把重复的 IO 换成一次。
    """
    df = store.load(cfg, tags=list(DEMO_TAGS)).collect(engine="streaming")
    log.info("演示数据集：%d 行（%s）", df.height, "、".join(DEMO_TAGS))
    return df.lazy()


def run(cfg: Config, write: bool = True) -> pl.DataFrame:
    cfg.ensure_dirs()
    lf = _demo_frame(cfg)

    panels = []
    for name, program in PROGRAMS.items():
        panels.append(_panel(program, world_pit, lf, cfg, name, "pit"))
        panels.append(_panel(program, lambda f, _t: world_rev(f), lf, cfg, name, "rev"))
    panel = pl.concat(panels)

    rows = []
    for name, program in PROGRAMS.items():
        trunc = truncation_test(cfg, lf, program, name)
        lam = lambda_diff(panel, name, cfg.lambda_tol)
        vintage_pass = (lam["lambda_cell"] or 0.0) == 0.0
        rows.append({
            "program": name,
            "truncation_pass": trunc["truncation_pass"],
            "lambda_cell": lam["lambda_cell"],
            "lambda_rank": lam["lambda_rank"],
            "n_cells": lam["n_cells"],
            "n_diff": lam["n_diff"],
            "vintage_pass": vintage_pass,
            "verdict": _verdict(trunc["truncation_pass"], vintage_pass),
        })
    result = pl.DataFrame(rows)

    if write:
        panel.write_parquet(cfg.out_dir / "demo_panel.parquet")
        result.write_csv(cfg.out_dir / "demo_lambda.csv")
        log.info("demo_panel → %d 行 | demo_lambda → %d 行", panel.height, result.height)
    return result
