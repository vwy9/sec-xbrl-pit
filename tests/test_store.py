import datetime as dt

import polars as pl
import pytest

from secpit import store
from secpit.config import BUSINESS_KEY


def _value(df, cik):
    return df.filter(pl.col("cik") == cik)["value"].item()


def test_as_of_walks_the_vintage_series(tiny_cfg):
    """同一个键在三个时点上应分别返回第一、第二、第三版取值。"""
    lf = store.load(tiny_cfg, ciks=[1])
    assert _value(store.as_of(lf, dt.date(2025, 3, 1)).collect(), 1) == 100.0
    assert _value(store.as_of(lf, dt.date(2025, 6, 1)).collect(), 1) == 100.0
    assert _value(store.as_of(lf, dt.date(2025, 9, 1)).collect(), 1) == 130.0


def test_as_of_never_returns_future_filings(tiny_cfg):
    lf = store.load(tiny_cfg)
    for t in (dt.date(2025, 2, 15), dt.date(2025, 5, 15), dt.date(2025, 8, 15)):
        got = store.as_of(lf, t).collect()
        assert got["filed"].max() <= t


def test_as_of_converges_to_latest(tiny_cfg):
    lf = store.load(tiny_cfg)
    late = store.as_of(lf, dt.date(2030, 1, 1)).collect().sort(list(BUSINESS_KEY))
    rev = store.latest(lf).collect().sort(list(BUSINESS_KEY))
    assert late.equals(rev)


def test_as_of_traces_back_to_a_filing(tiny_cfg):
    """返回值须带 filed 与 adsh，否则无法追溯是哪份申报带来的。"""
    got = store.as_of(store.load(tiny_cfg, ciks=[2]), dt.date(2025, 2, 15)).collect()
    assert got["adsh"].item() == "b-1"
    got = store.as_of(store.load(tiny_cfg, ciks=[2]), dt.date(2025, 3, 15)).collect()
    assert got["adsh"].item() == "b-2"      # 修正案


def test_diff_worlds_shrinks_to_zero_at_late_dates(tiny_cfg):
    assert store.diff_worlds(tiny_cfg, dt.date(2025, 2, 15)).height > 0
    assert store.diff_worlds(tiny_cfg, dt.date(2030, 1, 1)).height == 0


def test_primary_key_is_unique(tiny_cfg):
    assert store.assert_primary_key(tiny_cfg) == 1


def test_missing_data_raises(tmp_path):
    from secpit.config import Config
    cfg = Config.for_quarters("2025q1", root=tmp_path)
    cfg.ensure_dirs()
    with pytest.raises(FileNotFoundError):
        store.load(cfg)
