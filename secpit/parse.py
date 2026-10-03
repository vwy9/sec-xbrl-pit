"""把季度 ZIP 解析成观测事实表的分区文件。

表头先断言再读，不一致即报错退出；过滤逐条计数，丢弃数与保留数之和
必须等于输入行数；同季度重跑覆盖同一批文件而非追加，保证幂等。
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from secpit.config import (
    FORM_WHITELIST,
    NUM_COLUMNS,
    OBSERVATION_COLUMNS,
    QTRS_WHITELIST,
    SUB_COLUMNS,
    TAG_COLUMNS,
    Config,
)

log = logging.getLogger(__name__)


class SchemaDriftError(RuntimeError):
    """ZIP 内某张表的表头与预期不符。"""


# 只读需要的列。num.footnote 与 tag.doc 是长文本，跳过它们能显著降低内存峰值。
_SUB_USE = ("adsh", "cik", "form", "period", "fy", "fp", "filed", "accepted")
_NUM_USE = ("adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg", "value")
_TAG_USE = ("tag", "version", "custom")


def _read_header(zf: zipfile.ZipFile, member: str) -> tuple[str, ...]:
    """只读首行，避免为了断言表头而载入整张表。"""
    with zf.open(member) as fh:
        first = fh.readline()
    return tuple(first.decode("utf-8", "replace").rstrip("\r\n").split("\t"))


def _assert_header(actual: tuple[str, ...], expected: tuple[str, ...],
                   quarter: str, member: str) -> None:
    if actual == expected:
        return
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    raise SchemaDriftError(
        f"{quarter}/{member} 表头与预期不符：\n"
        f"  预期 {len(expected)} 列，实得 {len(actual)} 列\n"
        f"  缺失 {missing}\n"
        f"  多出 {extra}\n"
        f"  实得表头 {actual}"
    )


def _read_table(zip_path: Path, member: str, expected: tuple[str, ...],
                use: tuple[str, ...], quarter: str) -> pl.DataFrame:
    """从 ZIP 内读一张表。先断言表头，再只读 `use` 指定的列，全部按字符串读入。

    SEC 的 TXT 中存在未转义的引号，故 quote_char 必须为 None；
    编码存在非 UTF-8 字节，故用 utf8-lossy。
    """
    with zipfile.ZipFile(zip_path) as zf:
        _assert_header(_read_header(zf, member), expected, quarter, member)
        with zf.open(member) as fh:
            raw = fh.read()
    df = pl.read_csv(
        io.BytesIO(raw),
        separator="\t",
        quote_char=None,
        has_header=True,
        columns=list(use),
        infer_schema_length=0,          # 全列 Utf8，类型转换稍后显式做
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
    )
    del raw
    return df


@dataclass
class FilterCounts:
    """单季度的过滤审计。各丢弃数与保留数之和必须等于 total_in。"""

    quarter: str
    total_in: int = 0
    drop_segments: int = 0
    drop_qtrs: int = 0
    drop_form: int = 0
    drop_custom: int = 0
    drop_dup_identical: int = 0
    drop_ambiguous: int = 0
    kept: int = 0

    def is_conserved(self) -> bool:
        return (self.drop_segments + self.drop_qtrs + self.drop_form
                + self.drop_custom + self.drop_dup_identical
                + self.drop_ambiguous + self.kept) == self.total_in

    def as_dict(self) -> dict:
        d = asdict(self)
        d["conserved"] = self.is_conserved()
        return d

    def __str__(self) -> str:
        return (f"{self.quarter}: in={self.total_in:,} "
                f"segments={self.drop_segments:,} qtrs={self.drop_qtrs:,} "
                f"form={self.drop_form:,} custom={self.drop_custom:,} "
                f"dup={self.drop_dup_identical:,} ambig={self.drop_ambiguous:,} "
                f"kept={self.kept:,} conserved={self.is_conserved()}")


def _eligible_submissions(sub: pl.DataFrame) -> pl.DataFrame:
    """按表单白名单过滤申报表，返回带时点列的子集。"""
    return sub.filter(pl.col("form").is_in(list(FORM_WHITELIST)))


def _custom_tag_keys(tag: pl.DataFrame) -> pl.DataFrame:
    """自定义标签的 (tag, version) 集合。

    注意：自定义标签的 version 列存的是定义它的那份 filing 的 adsh，
    而非 us-gaap/2024 这类 taxonomy 版本，故必须按二元组匹配。
    """
    return tag.filter(pl.col("custom") == "1").select("tag", "version").unique()


def _is_blank(name: str) -> pl.Expr:
    """字段为空。

    polars 在 infer_schema_length=0 下把 CSV 的空字段读成 null 而非空串，
    但个别行可能带空白字符，故两种情况都算空。
    """
    col = pl.col(name)
    return col.is_null() | (col.str.strip_chars() == "")


def _filter_num(num: pl.DataFrame, eligible: pl.DataFrame, custom: pl.DataFrame,
                counts: FilterCounts) -> pl.DataFrame:
    """依次施加四个条件，每步记录被剔除的行数。"""
    counts.total_in = num.height

    # 条件一：仅合并口径（维度列与共同注册人列皆空）
    df = num.filter(_is_blank("segments") & _is_blank("coreg"))
    counts.drop_segments = counts.total_in - df.height

    # 条件二：qtrs 属白名单
    before = df.height
    df = df.with_columns(pl.col("qtrs").cast(pl.Int32, strict=False).alias("_qtrs"))
    df = df.filter(pl.col("_qtrs").is_in(list(QTRS_WHITELIST)))
    counts.drop_qtrs = before - df.height

    # 条件三：adsh 属于合格申报
    before = df.height
    df = df.join(eligible, on="adsh", how="inner")
    counts.drop_form = before - df.height

    # 条件四：剔除自定义标签
    before = df.height
    df = df.join(custom.with_columns(pl.lit(True).alias("_custom")),
                 on=["tag", "version"], how="left")
    df = df.filter(pl.col("_custom").is_null()).drop("_custom")
    counts.drop_custom = before - df.height

    # 条件五：同一份 filing 内同键取值矛盾。
    #
    # 原以为 (adsh, tag, ddate, qtrs, uom) 在单份申报内天然唯一——这个断言只在
    # 2025Q1 上验证过。全量历史推翻了它：2011 年有 3 组、2012 年有 10 组，
    # 同一份 filing、同一个 taxonomy version，同一个事实报了两个不同的值
    # （例如 EarningsPerShareDiluted 同时是 -0.01 和 0.40）。
    # 发生率约每百万行 1 处，成因是申报本身的内部矛盾，不是 taxonomy 差异。
    #
    # 处理：取值相同的重复去重保留一行；取值矛盾的整组丢弃——
    # 一份申报对同一事实给出两个数时，它在这份申报里就是不可判定的，
    # 保留任何一个都是我们在替申报人做选择。两类分开计数。
    key = ["adsh", "tag", "ddate", "qtrs", "uom"]
    before = df.height
    df = df.with_columns(
        pl.len().over(key).alias("_n"),
        pl.col("value").n_unique().over(key).alias("_nv"),
    )
    counts.drop_ambiguous = df.filter(pl.col("_nv") > 1).height
    df = df.filter(pl.col("_nv") <= 1)
    df = df.unique(subset=key, keep="first", maintain_order=True)
    counts.drop_dup_identical = before - counts.drop_ambiguous - df.height
    df = df.drop("_n", "_nv")

    counts.kept = df.height
    return df


def _to_observations(df: pl.DataFrame) -> pl.DataFrame:
    """类型转换并选定观测表的列顺序。"""
    out = df.with_columns(
        pl.col("cik").cast(pl.Int64, strict=False),
        pl.col("filed").str.to_date("%Y%m%d", strict=False),
        pl.col("period").str.to_date("%Y%m%d", strict=False),
        pl.col("ddate").str.to_date("%Y%m%d", strict=False),
        pl.col("accepted").str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
        pl.col("fy").cast(pl.Int32, strict=False),
        pl.col("_qtrs").cast(pl.Int8).alias("qtrs"),
        pl.col("value").cast(pl.Float64, strict=False),
    )
    out = out.with_columns(pl.col("filed").dt.year().cast(pl.Int32).alias("filed_year"))
    return out.select(OBSERVATION_COLUMNS)


def _write_partitions(cfg: Config, obs: pl.DataFrame, quarter: str) -> list[Path]:
    """两级分区写出：filed_year / cik_bucket。同季度同格一个文件，重跑覆盖（幂等）。

    两个分区键各自服务一类访问：
    - `filed_year` 让主键断言与受理时刻统计可以逐年独立进行
      （一份 filing 只在一个季度受理，故只落在一个年份分区里）；
    - `cik_bucket` 让重述检出与普查的分区过滤变成真正的分区裁剪。
      没有它时 `cik % N == i` 只是一个计算出来的过滤条件，每个分区仍要扫完整张表——
      实测那样加分区数完全不降内存（8 分区 5.05GB，16 分区 5.04GB）。
    """
    n = max(1, cfg.key_partitions)
    obs = obs.with_columns((pl.col("cik") % n).alias("cik_bucket"))
    written = []
    for (year, bucket), part in obs.group_by(["filed_year", "cik_bucket"],
                                             maintain_order=True):
        if year is None:
            log.warning("%s: %d 行 filed 为空，跳过", quarter, part.height)
            continue
        d = (cfg.observations_dir / f"filed_year={int(year)}"
             / f"cik_bucket={int(bucket)}")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{quarter}.parquet"
        part.drop("filed_year", "cik_bucket").write_parquet(path)
        written.append(path)
    return written


def parse_quarter(cfg: Config, quarter: str) -> FilterCounts:
    zp = cfg.raw_dir / f"{quarter}.zip"
    if not zp.exists():
        raise FileNotFoundError(f"{zp} 不存在，请先执行 download")

    sub = _read_table(zp, "sub.txt", SUB_COLUMNS, _SUB_USE, quarter)
    tag = _read_table(zp, "tag.txt", TAG_COLUMNS, _TAG_USE, quarter)
    num = _read_table(zp, "num.txt", NUM_COLUMNS, _NUM_USE, quarter)

    counts = FilterCounts(quarter=quarter)
    kept = _filter_num(num, _eligible_submissions(sub), _custom_tag_keys(tag), counts)
    del num, tag

    if not counts.is_conserved():
        raise RuntimeError(f"{quarter}: 过滤计数不守恒 {counts}")

    obs = _to_observations(kept)
    paths = _write_partitions(cfg, obs, quarter)
    log.info("%s → %d 个分区文件 | %s", quarter, len(paths), counts)
    return counts


def parse_all(cfg: Config, quarters: tuple[str, ...] | None = None) -> list[FilterCounts]:
    cfg.ensure_dirs()
    quarters = quarters or cfg.quarters
    all_counts = [parse_quarter(cfg, q) for q in quarters]

    pl.DataFrame([c.as_dict() for c in all_counts]).write_csv(
        cfg.parquet_dir / "_filter_counts.csv"
    )
    return all_counts
