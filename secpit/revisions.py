"""重述事件检出与重述普查。

事件表记录「相邻两次观测之间发生了什么」，普查表记录「一个键一生被改成什么样」。
两者都不设幅度阈值——任何数值差异都入表，阈值留到 stats 阶段施加，
使阈值敏感性可以事后分析而不必重跑检出。
"""

from __future__ import annotations

import glob
import logging
import tempfile
from pathlib import Path

import polars as pl

from secpit.config import BUSINESS_KEY, COMPARATIVE_GAP_DAYS, SCALE_LOG_TOL, Config
from secpit import store

log = logging.getLogger(__name__)

_ORDER = list(BUSINESS_KEY) + ["filed", "adsh"]


def _partitions(cfg: Config) -> list[int]:
    """有数据的 cik 桶。

    空桶必须跳过：polars 对匹配不到任何文件的 glob 会抛 ComputeError，
    而当公司数少于桶数时（小样本、测试夹具）空桶是正常的。
    """
    n = max(1, cfg.key_partitions)
    return [i for i in range(n) if glob.glob(cfg.bucket_glob(i))]


def _apply_tag_map(lf: pl.LazyFrame, tag_map: dict[str, str] | None) -> pl.LazyFrame:
    """把标签替换为其等价类代表。

    不做这一步时，一个概念被改名后旧键停止、新键开始，在库里是两个不相干的键，
    「旧概念报过、新概念在次年可比列重报同一期」的重述会被整个漏掉。
    映射由 `secpit.renames` 从数据本身推导，见该模块的判据说明。
    """
    if not tag_map:
        return lf
    return lf.with_columns(
        pl.col("tag").replace(tag_map).alias("tag")
    )


def _resolve_merged_duplicates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """标签合并后，对同一份申报内撞到同一个业务键的多行施加 A6。

    原始口径下主键 (adsh, tag, ddate, qtrs, uom) 唯一，同一份申报对一个业务键
    至多一行。合并成等价类后不再如此：一份申报可能同时报了旧标签和新标签
    （例如 Revenues 与 SalesRevenueNet 被并入同一类），两者取值往往不同。
    不处理的话，按 (filed, adsh) 排序后这两行相邻，被当成一次滞后 0 天的
    「重述」——实测这类假事件有 32,977 条，占合并后新增事件的六成以上。

    处理与解析层同一条纪律：取值相同的留一行；取值矛盾的整组丢弃，
    因为一份申报对同一事实给出两个数时，它在这份申报里不可判定。
    version 可能因新旧标签所属的 taxonomy 年份不同而不同，统一取最小者，
    使去重结果与行序无关。
    """
    key = list(BUSINESS_KEY) + ["adsh"]
    return (
        lf.with_columns(
            pl.col("value").n_unique().over(key).alias("_nv"),
            pl.col("version").min().over(key),
        )
        .filter(pl.col("_nv") == 1)
        .drop("_nv")
        .unique(subset=key, keep="any")
    )


def merged_conflicts(cfg: Config, tag_map: dict[str, str]) -> int:
    """标签合并后，同一份申报内取值矛盾而被整组丢弃的 (业务键, adsh) 组数。"""
    key = list(BUSINESS_KEY) + ["adsh"]
    n = 0
    for part in _partitions(cfg) or [None]:
        lf = (store.load(cfg) if part is None
              else pl.scan_parquet(cfg.bucket_glob(part)))
        n += (
            _apply_tag_map(lf.select(*BUSINESS_KEY, "adsh", "value"), tag_map)
            .filter(pl.col("value").is_not_null())
            .group_by(key).agg(pl.col("value").n_unique().alias("nv"))
            .filter(pl.col("nv") > 1)
            .select(pl.len()).collect(engine="streaming").item()
        )
    return n


def _valued(cfg: Config, part: int | None = None,
            tag_map: dict[str, str] | None = None) -> pl.LazyFrame:
    """只保留有取值的观测，按业务键与受理次序排好。

    观测表本身保留 value 为空的行——它们是 filing 的真实内容（某科目本期未填报）。
    但空值不是「对一个数的观测」，把它计入会造成两类错误：
    先空后有值被误判成重述；n_unique 把 null 算作一个不同取值。
    故重述检出与普查一律在有值的观测上进行。
    """
    # accepted/fy/fp 在检出与普查中都用不到，先丢掉再排序。
    # period 留着：普查要据此判断首次观测是不是比较列（见 _census_partition）。
    keep = list(BUSINESS_KEY) + ["value", "filed", "adsh", "form", "version", "period"]
    # tag / uom / version 高度重复（各约 4.8k / 40 / 20 个取值），转 Categorical
    # 压排序内存。adsh 不转（7 万个取值，且作次级排序键需要字典序）；
    # form 不转（is_amendment 要做 str.ends_with，字符串方法不接受 Categorical）。
    dict_cols = ["tag", "uom", "version"]
    if part is None:
        lf = store.load(cfg)
    else:
        # 直接扫该 cik 桶的分区目录——这是真正的分区裁剪，只读 1/N 的文件。
        # 若改用 `filter(cik % N == part)`，那只是个计算条件，仍要扫完整张表。
        lf = pl.scan_parquet(cfg.bucket_glob(part))
    lf = _apply_tag_map(lf.select(keep), tag_map).filter(pl.col("value").is_not_null())
    if tag_map:
        lf = _resolve_merged_duplicates(lf)
    return (
        lf.with_columns([pl.col(c).cast(pl.Categorical) for c in dict_cols])
          .sort(_ORDER)
    )


def _scale_exponent_expr() -> pl.Expr:
    """疑似单位跃迁的十进制指数，非跃迁为 null。

    样例：GE 的 StockRepurchasedDuringPeriodShares FY2014 由 73.6 变为
    73,600,000，比值恰为 1e6——申报口径由百万股改为股，并非重述。
    存指数而非布尔，「哪些幂算跃迁」可在统计阶段调整而不必重跑检出。
    """
    ratio = pl.col("v_new") / pl.col("v_old")
    lg = ratio.log10()
    k = lg.round(0)
    ok = (
        pl.col("v_old").is_not_null()
        & pl.col("v_new").is_not_null()
        & (pl.col("v_old") != 0)
        & (ratio > 0)
        & ratio.is_finite()
        & ((lg - k).abs() < SCALE_LOG_TOL)
        & (k != 0)
    )
    # cast 必须在 when/otherwise 之外：polars 会对全部行求值 then 分支，
    # 而 v_old 极小时 ratio 溢出为 inf，直接 cast 到 Int8 会失败。
    return (
        pl.when(ok).then(k).otherwise(None)
        .cast(pl.Int8, strict=False)
        .alias("scale_exponent")
    )


def _detect_partition(cfg: Config, part: int | None,
                      tag_map: dict[str, str] | None = None) -> pl.DataFrame:
    lf = _valued(cfg, part, tag_map)

    over = list(BUSINESS_KEY)
    shifted = lf.with_columns(
        pl.col("filed").first().over(over).alias("filed_first"),
        pl.col("value").shift(1).over(over).alias("v_old"),
        pl.col("adsh").shift(1).over(over).alias("adsh_old"),
        pl.col("form").shift(1).over(over).alias("form_old"),
        pl.col("filed").shift(1).over(over).alias("filed_old"),
        pl.col("version").shift(1).over(over).alias("version_old"),
    ).rename({
        "value": "v_new", "adsh": "adsh_new", "form": "form_new",
        "filed": "filed_new", "version": "version_new",
    })

    # _valued 已滤掉空值，此处的非空判断只为拦住每组首行（shift 产生的 null）
    events = shifted.filter(
        pl.col("v_old").is_not_null()
        & (pl.col("v_old") != pl.col("v_new"))
    )

    denom = pl.max_horizontal(
        pl.col("v_old").abs(), pl.col("v_new").abs(), pl.lit(1.0)
    )
    # 两种滞后，回答两个不同的问题：
    #   lag_days             与上一次观测相隔多久——「改动随哪一份申报到来」；
    #   lag_since_first_days 与该事实首次发布相隔多久——「一个已公开的数在多久之后被改」。
    # 时点存量尤其要分开：上年末余额会在其后三份 10-Q 里反复出现，次年 10-K 改它时
    # lag_days 只有约 100 天，而这个数其实已经公开了一年。度量泄漏用后者。
    events = events.with_columns(
        (pl.col("v_new") - pl.col("v_old")).alias("abs_delta"),
        ((pl.col("v_new") - pl.col("v_old")).abs() / denom).alias("rel_delta"),
        (pl.col("filed_new") - pl.col("filed_old")).dt.total_days()
            .cast(pl.Int32).alias("lag_days"),
        (pl.col("filed_new") - pl.col("filed_first")).dt.total_days()
            .cast(pl.Int32).alias("lag_since_first_days"),
        (pl.col("version_old") != pl.col("version_new")).alias("version_changed"),
        pl.col("form_new").str.ends_with("/A").alias("is_amendment"),
        _scale_exponent_expr(),
    ).with_columns(
        (pl.col("v_old").cum_count().over(over)).cast(pl.Int32).alias("seq")
    )

    out = events.select(
        *BUSINESS_KEY, "seq", "v_old", "v_new", "abs_delta", "rel_delta",
        "adsh_old", "adsh_new", "form_old", "form_new",
        "filed_old", "filed_new", "lag_days", "lag_since_first_days",
        "version_old", "version_new", "version_changed",
        "is_amendment", "scale_exponent",
    ).collect(engine="streaming")
    return out


def detect(cfg: Config, write: bool = True,
           tag_map: dict[str, str] | None = None) -> pl.DataFrame:
    """相邻两次观测值不同即一条事件。

    按 cik 取模分区逐块处理。重述完全发生在业务键内部，而业务键含 cik，
    所以分区之间没有依赖，拼接结果与不分区逐位相同——只是峰值内存降为约 1/N。
    """
    cfg.ensure_dirs()
    parts = _partitions(cfg)
    if not parts:
        return _detect_partition(cfg, None, tag_map)

    # 每个分区算完立刻落盘再释放。攒在列表里会让峰值随分区数累积——
    # 实测单分区 2.91GB、八个分区跑完 4.42GB，多出来的就是这份累积。
    with tempfile.TemporaryDirectory(dir=cfg.out_dir) as tmp:
        for part in parts:
            got = _detect_partition(cfg, part, tag_map)
            got.write_parquet(Path(tmp) / f"{part}.parquet")
            log.debug("detect 分区 %d → %d 条", part, got.height)
            del got
        out = pl.read_parquet(str(Path(tmp) / "*.parquet"))

    if write:
        path = cfg.out_dir / ("revisions_canonical.parquet" if tag_map
                              else "revisions.parquet")
        out.write_parquet(path)
        log.info("revisions → %s（%d 条事件，%d 个分区）",
                 path, out.height, len(parts))
    return out


def _census_partition(cfg: Config, part: int | None, events: pl.DataFrame,
                      tag_map: dict[str, str] | None = None) -> pl.DataFrame:
    lf = _valued(cfg, part, tag_map)

    base = lf.group_by(list(BUSINESS_KEY)).agg(
        pl.len().alias("n_obs"),
        pl.col("value").n_unique().alias("n_distinct_val"),
        pl.col("value").first().alias("v_first"),
        pl.col("value").last().alias("v_final"),
        pl.col("filed").first().alias("filed_first"),
        pl.col("filed").last().alias("filed_final"),
        pl.col("period").first().alias("period_first"),
    )

    # magnitude 分母取 max(|v_first|, 1)：首值为 0 时按定义的 |v_first| 无意义、
    # 极小时溢出。但兜底只是避免除零，并不能让 magnitude 在 v_first=0 时有意义——
    # 那时它退化成"以计量单位计的绝对变化"。实测这类键占被重述键的约 1%，
    # 却贡献了幅度分布的整条尾巴（max 达 6.6e9，而其余键 p99 仅 999）。
    # 故单独标记 from_zero，由统计阶段排除。
    mag_denom = pl.max_horizontal(pl.col("v_first").abs(), pl.lit(1.0))
    base = base.with_columns(
        (pl.col("n_distinct_val") >= 2).alias("revised"),
        (pl.col("n_distinct_val") - 1).cast(pl.Int32).alias("n_revisions"),
        ((pl.col("v_final") - pl.col("v_first")).abs() / mag_denom).alias("magnitude"),
        (pl.col("v_first") == 0).alias("from_zero"),
        (pl.col("filed_final") - pl.col("filed_first")).dt.total_days()
            .cast(pl.Int32).alias("lag_days"),
        (pl.lit(cfg.window_end) - pl.col("filed_first")).dt.total_days()
            .cast(pl.Int32).alias("obs_window_days"),
    ).with_columns(
        (pl.col("obs_window_days") < cfg.right_truncation_days).alias("right_censored"),
        # 左截断有两个来源，各记一列，并集为 left_censored：
        #   窗口：首次观测落在窗口第一个季度，原始申报可能在窗口之前；
        #   比较列：首次观测所在申报的报告期晚于该事实的期末日，即这个数第一次出现
        #   就是在比较列里，它所属期间的原始申报不在库中——XBRL 2009–2011 年分批强制、
        #   IPO、由 20-F 转报 10-K、改用标准标签等都会造成这种情况。
        # 只看窗口会漏掉后者：全量窗口下前者只有约 1,300 个键，后者约 148 万个。
        (pl.col("filed_first") < pl.lit(cfg.left_censor_cutoff)).alias("left_censored_window"),
        (pl.col("ddate") < pl.col("period_first") - pl.duration(days=COMPARATIVE_GAP_DAYS))
            .fill_null(False).alias("first_is_comparative"),
    ).with_columns(
        (pl.col("left_censored_window") | pl.col("first_is_comparative")).alias("left_censored"),
    )

    flags = (
        events.lazy()
        .group_by(list(BUSINESS_KEY))
        .agg(
            pl.col("is_amendment").any().alias("ever_amendment"),
            pl.col("scale_exponent").is_not_null().any().alias("ever_scale_change"),
        )
    )

    out = (
        base.join(flags, on=list(BUSINESS_KEY), how="left")
        .with_columns(
            pl.col("ever_amendment").fill_null(False),
            pl.col("ever_scale_change").fill_null(False),
        )
        .collect(engine="streaming")
    )
    return out


def census(cfg: Config, events: pl.DataFrame | None = None,
           write: bool = True, return_frame: bool = True,
           tag_map: dict[str, str] | None = None) -> pl.DataFrame | int:
    """每个业务键一行。与 detect 一样按 cik 取模分区，理由相同。"""
    if events is None:
        events = detect(cfg, write=False, tag_map=tag_map)

    cfg.ensure_dirs()
    parts = _partitions(cfg)
    if not parts:
        return _census_partition(cfg, None, events, tag_map)

    n = max(1, cfg.key_partitions)
    with tempfile.TemporaryDirectory(dir=cfg.out_dir) as tmp:
        total = 0
        for part in parts:
            ev_part = events.filter(pl.col("cik") % n == part)
            got = _census_partition(cfg, part, ev_part, tag_map)
            got.write_parquet(Path(tmp) / f"{part}.parquet")
            total += got.height
            log.debug("census 分区 %d → %d 键", part, got.height)
            del got, ev_part

        pattern = str(Path(tmp) / "*.parquet")
        if write:
            # 流式合并直接落盘，绝不把 4177 万行读回内存再写出去——
            # 那一步本身就要约 1.8GB，而调用方通常只需要行数。
            path = cfg.out_dir / ("census_canonical.parquet" if tag_map
                                  else "census.parquet")
            pl.scan_parquet(pattern).sink_parquet(path)
            log.info("census → %s（%d 个业务键，%d 个分区）", path, total, len(parts))
            return pl.read_parquet(path) if return_frame else total
        return pl.read_parquet(pattern) if return_frame else total
