"""两张图，每张都要能脱离 README 单独看懂。

图一 重述普查：问题真实存在，有多普遍、多大、隔多久。
图二 Λ 演示：截断检验放过了哪一类程序。

图内文本一律英文：matplotlib 默认字体不含 CJK 字形，中文会渲染成豆腐块。
统计表已经很小，这里用 pandas 读 CSV 交给 matplotlib 最顺手。
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")                      # 无显示环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from secpit.config import Config

log = logging.getLogger(__name__)

_C_MAIN = "#2b6cb0"
_C_ALT = "#c05621"
_C_HL = "#c53030"
_C_GREY = "#a0aec0"


def _window_label(cfg: Config) -> str:
    return f"{cfg.quarters[0].upper()}-{cfg.quarters[-1].upper()} ({cfg.window_start} to {cfg.window_end})"


# 图一：重述普查

def census_figure(cfg: Config, censoring: str = "all"):
    out = cfg.out_dir
    rate = pd.read_csv(out / "stats_rate_by_threshold.csv")
    by_tag = pd.read_csv(out / "stats_rate_by_tag.csv")
    mag = pd.read_csv(out / "stats_magnitude_quantiles.csv")
    silent = pd.read_csv(out / "stats_silent_vs_amendment.csv")
    comp = pd.read_csv(out / "stats_census_composition.csv")
    # 只惰性取绘图需要的那一列、那些行。全量窗口下普查表有数千万行，
    # 整张读进内存没有必要，也放不下。
    magnitudes = (
        pl.scan_parquet(out / "census.parquet")
        .filter(pl.col("revised") & ~pl.col("from_zero") & ~pl.col("ever_scale_change"))
        .select("magnitude")
        .collect(engine="streaming")["magnitude"]
        .to_numpy()
    )
    lag_tbl = (
        pl.scan_parquet(out / "revisions.parquet")
        .select("lag_since_first_days", "lag_days")
        .collect(engine="streaming")
    )
    lags = lag_tbl["lag_since_first_days"].drop_nulls().to_numpy()
    lags_adj = lag_tbl["lag_days"].drop_nulls().to_numpy()

    sel = lambda df: df[df["censoring"] == censoring]
    r0 = sel(rate).sort_values("threshold")
    n_multi = int(r0["n_keys_multi"].iloc[0])
    n_rev = int(r0["n_keys_revised"].iloc[0])
    sil = sel(silent).set_index("category")
    silent_share = float(sil["share"]["silent"])
    n_events = int(sil["n_events"].sum())
    # 中图的中位数取统计表里的值（含首末相等、幅度为 0 的键），与 README、REPORT 同一口径；
    # 直方图是对数轴，画不出 0，只能画正值，被略去的个数写进标题。
    m_all = sel(mag)
    m_all = m_all[(m_all["scale_filter"] == "without_scale")].set_index("quantile")
    mag_median = float(m_all["value"][0.5])
    mag_n = int(m_all["n_keys"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    fig.suptitle(
        f"Restatement census of US XBRL financial statement facts   |   window {_window_label(cfg)}\n"
        f"Of {n_multi:,} facts reported by two or more filings, {n_rev:,} ({n_rev/n_multi:.2%}) "
        f"had their value changed; {silent_share:.1%} of the {n_events:,} revision events "
        f"came with no formal amendment",
        fontsize=13, y=0.99,
    )

    # 左：重述率按标签排名
    ax = axes[0]
    t = sel(by_tag).nlargest(15, "n_keys_multi").sort_values("rate")
    ax.barh(range(len(t)), t["rate"] * 100, color=_C_MAIN)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([s[:34] for s in t["tag"]], fontsize=8)
    ax.set_xlabel("share of facts revised (%)")
    ax.set_title("Revision rate by field\n(15 most frequently reported us-gaap tags)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # 中：首末相对幅度分布
    ax = axes[1]
    m = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
    n_zero = int((magnitudes == 0).sum())
    if len(m):
        ax.hist(np.log10(m), bins=50, color=_C_ALT, alpha=0.85)
        ax.axvline(np.log10(mag_median), color=_C_HL, ls="--", lw=1.5,
                   label=f"median {mag_median:.1%} (all {mag_n:,} facts)")
        ax.legend(fontsize=9)
    ax.set_xlabel("first-to-final relative magnitude, log10 scale")
    ax.set_ylabel("number of facts")
    ax.set_title(f"Magnitude of revisions (n={mag_n:,} facts)\n"
                 f"unit-scale jumps and zero-origin facts excluded;\n"
                 f"{n_zero:,} facts back at their first value (magnitude 0) not drawn on log axis",
                 fontsize=10)
    ax.grid(alpha=0.3)

    # 右：滞后天数分布。主量是距首次发布的天数——「一个已公开的数多久之后被改」；
    # 距上一次观测的天数只画轮廓作对照。两者在时点存量上差别很大：上年末余额在
    # 其后三份季报里反复出现，相邻口径会把「公开一年后才改」量成「距上一份季报约 100 天」。
    ax = axes[2]
    lag = lags[lags >= 0]
    adj = lags_adj[lags_adj >= 0]
    tail_note = ""
    if len(lag):
        med = float(np.median(lag))
        # 尾部可以长到十年以上，直接画会把一年处那根尖峰压扁。
        # 截到 p99.5 作图，被截掉的部分在标题里注明，不藏。
        cut = float(np.quantile(lag, 0.995))
        bins = np.linspace(0, cut, 61)
        n_hidden = int((lag > cut).sum())
        ax.hist(lag[lag <= cut], bins=bins, color=_C_MAIN, alpha=0.85,
                label="since first publication")
        ax.hist(adj[adj <= cut], bins=bins, histtype="step", color=_C_GREY, lw=1.5,
                label="since previous filing")
        ax.axvline(med, color=_C_HL, ls="--", lw=1.5, label=f"median {med:.0f} days")
        ax.legend(fontsize=9)
        if n_hidden:
            tail_note = (f"\nx clipped at {cut:.0f}d; {n_hidden/len(lag):.1%} run longer "
                         f"(max {lag.max():.0f}d)")
    ax.set_xlabel("days from first publication to the changed value")
    ax.set_ylabel("number of revision events")
    ax.set_title(f"Revision lag (n={len(lag):,} events)\n"
                 f"how long after publication the value changed{tail_note}", fontsize=10)
    ax.grid(alpha=0.3)

    # 口径脚注
    rates = "   ".join(f"threshold>{r.threshold:g}: {r.rate:.2%}"
                       for r in r0.itertuples() if pd.notna(r.rate))
    c = sel(comp).set_index("category")["n_keys"]
    fig.text(0.5, 0.015,
             f"censoring: {censoring}   |   revision rate by threshold -- {rates}\n"
             f"Revised facts: {int(c['revised_total']):,} total; "
             f"zero-origin {int(c['from_zero']):,}, unit-scale jumps {int(c['ever_scale_change']):,}, "
             f"via formal amendment {int(c['ever_amendment']):,}   |   "
             f"10-K/10-Q and amendments only; consolidated rows only; standard us-gaap tags only   |   "
             f"Source: SEC Financial Statement Data Sets",
             ha="center", fontsize=8, color="#4a5568")

    fig.tight_layout(rect=[0, 0.055, 1, 0.90])
    path = cfg.figures_dir / "fig_census.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log.info("fig_census → %s", path)
    return path


# 图二：Λ 演示

def _lambda_by_date(panel: pl.DataFrame, tol: float) -> pd.DataFrame:
    rows = []
    for program in panel["program"].unique(maintain_order=True):
        p = panel.filter(pl.col("program") == program)
        pit = p.filter(pl.col("world") == "pit").select("asof", "cik", pl.col("value").alias("y_pit"))
        rev = p.filter(pl.col("world") == "rev").select("asof", "cik", pl.col("value").alias("y_rev"))
        j = pit.join(rev, on=["asof", "cik"], how="full", coalesce=True)
        j = j.with_columns(
            pl.when(pl.col("y_pit").is_null() | pl.col("y_rev").is_null()).then(True)
            .otherwise(((pl.col("y_pit") - pl.col("y_rev")).abs()
                        / (pl.col("y_pit").abs() + 1e-12)) > tol).alias("d")
        )
        g = j.group_by("asof").agg(pl.col("d").mean().alias("lambda_cell")).sort("asof")
        for r in g.iter_rows(named=True):
            rows.append({"program": program, "asof": r["asof"],
                         "lambda_cell": r["lambda_cell"]})
    return pd.DataFrame(rows)


def lambda_figure(cfg: Config):
    out = cfg.out_dir
    res = pd.read_csv(out / "demo_lambda.csv")
    panel = pl.read_parquet(out / "demo_panel.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4))
    fig.suptitle(
        "Three implementations of one cross-sectional signal task, run against two worlds: D_PIT and D_rev\n"
        "The truncation test checks y_t = f(D_0:t) but does not constrain whether D itself is F_t-measurable\n"
        "-- the top-right cell is the class of program it lets through",
        fontsize=13, y=0.985,
    )

    # 左：2x2 判定表
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(-0.95, 2.05)
    ax.set_ylim(-0.75, 2.62)

    cells = {(True, True): [], (True, False): [], (False, True): [], (False, False): []}
    for r in res.itertuples():
        cells[(bool(r.truncation_pass), bool(r.vintage_pass))].append(r.program)

    notes = {
        (True, True): "genuinely clean",
        (True, False): "SILENT LEAK\n<- the entire argument sits here",
        (False, True): "empty in practice\nfiltering on filed almost\nalways implies filtering on ddate",
        (False, False): "double failure",
    }
    for i, tp in enumerate([True, False]):
        for j, vp in enumerate([True, False]):
            who = cells[(tp, vp)]
            focus = tp and not vp
            y0 = 1 - i
            ax.add_patch(plt.Rectangle(
                (j, y0), 1, 1,
                facecolor="#fed7d7" if focus else "#edf2f7",
                edgecolor=_C_HL if focus else "#cbd5e0",
                lw=2.5 if focus else 1.0, zorder=1))
            ax.text(j + 0.5, y0 + 0.66, "\n".join(who) if who else "(empty)",
                    ha="center", va="center", fontsize=13, zorder=2,
                    color=_C_HL if focus else "#2d3748",
                    fontweight="bold" if focus else "normal")
            ax.text(j + 0.5, y0 + 0.26, notes[(tp, vp)], ha="center", va="center",
                    fontsize=8, color="#4a5568", zorder=2, linespacing=1.5)

    for j, lx in enumerate(["vintage differential PASS\n(L = 0)",
                            "vintage differential FAIL\n(L > 0)"]):
        ax.text(j + 0.5, 2.10, lx, ha="center", va="bottom",
                fontsize=10, fontweight="bold", linespacing=1.4)
    for i, ly in enumerate(["truncation\nPASS", "truncation\nFAIL"]):
        ax.text(-0.10, 1 - i + 0.5, ly, ha="right", va="center",
                fontsize=10, fontweight="bold", linespacing=1.4)

    for k, r in enumerate(res.itertuples()):
        ax.text(-0.90, -0.16 - 0.17 * k,
                f"{r.program:<10s} L_cell={r.lambda_cell:5.3f}   "
                f"L_rank={r.lambda_rank:7.4f}   "
                f"{int(r.n_diff)}/{int(r.n_cells)} cells differ",
                ha="left", va="top", fontsize=8.5, color="#4a5568",
                family="monospace")

    # 右：L_cell 随 as-of 日期
    ax = axes[1]
    lbd = _lambda_by_date(panel, cfg.lambda_tol)
    colors = {"correct": _C_MAIN, "naive": _C_HL, "lookahead": _C_GREY}
    styles = {"correct": "-", "naive": "-", "lookahead": "--"}
    for program, grp in lbd.groupby("program"):
        grp = grp.sort_values("asof")
        ax.plot(pd.to_datetime(grp["asof"]), grp["lambda_cell"], marker="o",
                label=program, color=colors.get(program),
                ls=styles.get(program, "-"), lw=2, ms=5)
    ax.set_ylabel("L_cell  --  share of panel cells that differ between worlds")
    ax.set_xlabel("as-of date")
    ax.set_title("Vintage differential across as-of dates\n"
                 "the vintage-correct program is flat at zero: it sees the same thing in both worlds",
                 fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    fig.text(0.5, 0.015,
             f"Task: over the {cfg.universe_size} largest companies by total assets visible at the as-of date, "
             f"compute the cross-sectional z-score of ROE = annual net income / period-end equity\n"
             f"correct: filters on filed itself   |   naive: aligns on reporting period only   |   "
             f"lookahead: reaches into future periods   |   window {_window_label(cfg)}",
             ha="center", fontsize=8, color="#4a5568")

    fig.tight_layout(rect=[0, 0.06, 1, 0.89])
    path = cfg.figures_dir / "fig_lambda.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log.info("fig_lambda → %s", path)
    return path


def build_all(cfg: Config, censoring: str = "all") -> list:
    return [census_figure(cfg, censoring), lambda_figure(cfg)]
