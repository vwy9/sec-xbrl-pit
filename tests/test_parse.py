"""解析层测试：表头断言、计数守恒、四个过滤条件。

用人工构造的小 ZIP，不碰真实数据。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl
import pytest

from secpit import parse
from secpit.config import NUM_COLUMNS, SUB_COLUMNS, TAG_COLUMNS, Config


def _tsv(header, rows):
    lines = ["\t".join(header)]
    lines += ["\t".join("" if v is None else str(v) for v in r) for r in rows]
    return ("\n".join(lines) + "\n").encode()


def _sub_row(adsh, form, **kw):
    d = {c: "" for c in SUB_COLUMNS}
    d.update(adsh=adsh, cik="1", form=form, period="20241231", fy="2024",
             fp="FY", filed="20250201", accepted="2025-02-01 16:30:00.0")
    d.update(kw)
    return [d[c] for c in SUB_COLUMNS]


def _num_row(adsh, tag, version="us-gaap/2024", ddate="20241231", qtrs="4",
             uom="USD", segments="", coreg="", value="1.0"):
    d = dict(adsh=adsh, tag=tag, version=version, ddate=ddate, qtrs=qtrs,
             uom=uom, segments=segments, coreg=coreg, value=value, footnote="")
    return [d[c] for c in NUM_COLUMNS]


def _tag_row(tag, version="us-gaap/2024", custom="0"):
    d = {c: "" for c in TAG_COLUMNS}
    d.update(tag=tag, version=version, custom=custom)
    return [d[c] for c in TAG_COLUMNS]


def _make_zip(path: Path, sub_rows, num_rows, tag_rows,
              sub_header=SUB_COLUMNS):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("sub.txt", _tsv(sub_header, sub_rows))
        zf.writestr("num.txt", _tsv(NUM_COLUMNS, num_rows))
        zf.writestr("tag.txt", _tsv(TAG_COLUMNS, tag_rows))


@pytest.fixture
def mini_zip(tmp_path) -> Config:
    """一个恰好命中每个过滤条件的小 ZIP。"""
    cfg = Config.for_quarters("2025q1", root=tmp_path)
    cfg.ensure_dirs()
    sub = [_sub_row("keep-1", "10-K"), _sub_row("drop-form", "8-K")]
    tag = [_tag_row("Assets"), _tag_row("MyOwnTag", version="keep-1", custom="1")]
    num = [
        _num_row("keep-1", "Assets"),                                  # 保留
        _num_row("keep-1", "Assets", segments="Risk=Hedging"),         # 条件一：有维度
        _num_row("keep-1", "Assets", coreg="SUB"),                     # 条件一：共同注册人
        _num_row("keep-1", "Assets", qtrs="7", ddate="20231231"),      # 条件二：qtrs 越界
        _num_row("drop-form", "Assets"),                               # 条件三：表单不合格
        _num_row("keep-1", "MyOwnTag", version="keep-1"),              # 条件四：自定义标签
    ]
    _make_zip(cfg.raw_dir / "2025q1.zip", sub, num, tag)
    return cfg


def test_each_filter_drops_exactly_one_row(mini_zip):
    c = parse.parse_quarter(mini_zip, "2025q1")
    assert c.total_in == 6
    assert c.drop_segments == 2          # 维度行 + 共同注册人行
    assert c.drop_qtrs == 1
    assert c.drop_form == 1
    assert c.drop_custom == 1
    assert c.kept == 1


def test_counts_are_conserved(mini_zip):
    assert parse.parse_quarter(mini_zip, "2025q1").is_conserved()


def test_header_drift_raises(tmp_path):
    """表头不符必须报错退出，不能静默继续。"""
    cfg = Config.for_quarters("2025q1", root=tmp_path)
    cfg.ensure_dirs()
    bad = tuple(["WRONG"] + list(SUB_COLUMNS[1:]))
    _make_zip(cfg.raw_dir / "2025q1.zip", [_sub_row("a", "10-K")],
              [_num_row("a", "Assets")], [_tag_row("Assets")], sub_header=bad)
    with pytest.raises(parse.SchemaDriftError) as e:
        parse.parse_quarter(cfg, "2025q1")
    assert "WRONG" in str(e.value) and "sub.txt" in str(e.value)


def test_header_drift_produces_no_output(tmp_path):
    cfg = Config.for_quarters("2025q1", root=tmp_path)
    cfg.ensure_dirs()
    bad = tuple(["WRONG"] + list(SUB_COLUMNS[1:]))
    _make_zip(cfg.raw_dir / "2025q1.zip", [_sub_row("a", "10-K")],
              [_num_row("a", "Assets")], [_tag_row("Assets")], sub_header=bad)
    with pytest.raises(parse.SchemaDriftError):
        parse.parse_quarter(cfg, "2025q1")
    assert not list(cfg.observations_dir.rglob("*.parquet"))


def test_types_are_converted(mini_zip):
    parse.parse_quarter(mini_zip, "2025q1")
    df = pl.read_parquet(next(mini_zip.observations_dir.rglob("*.parquet")))
    assert df.schema["filed"] == pl.Date
    assert df.schema["ddate"] == pl.Date
    assert df.schema["value"] == pl.Float64
    assert df.schema["qtrs"] == pl.Int8
    assert df.schema["cik"] == pl.Int64


def test_reparsing_is_idempotent(mini_zip):
    """同季度同年写同一个文件，重跑覆盖而非追加。"""
    parse.parse_quarter(mini_zip, "2025q1")
    n1 = pl.read_parquet(mini_zip.observations_glob).height
    parse.parse_quarter(mini_zip, "2025q1")
    n2 = pl.read_parquet(mini_zip.observations_glob).height
    assert n1 == n2 == 1


def test_missing_zip_raises(tmp_path):
    cfg = Config.for_quarters("2025q1", root=tmp_path)
    cfg.ensure_dirs()
    with pytest.raises(FileNotFoundError):
        parse.parse_quarter(cfg, "2025q1")
