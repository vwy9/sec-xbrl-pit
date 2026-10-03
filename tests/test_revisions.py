import polars as pl

from secpit import revisions
from secpit.config import BUSINESS_KEY


def test_repeated_values_yield_one_event(tiny_cfg):
    """值序列 100 → 100 → 130 只应产出 1 条事件，不是 2 条。"""
    ev = revisions.detect(tiny_cfg, write=False)
    a = ev.filter(pl.col("cik") == 1)
    assert a.height == 1
    assert a["v_old"].item() == 100.0 and a["v_new"].item() == 130.0


def test_no_event_has_equal_values(tiny_cfg):
    ev = revisions.detect(tiny_cfg, write=False)
    assert ev.filter(pl.col("v_old") == pl.col("v_new")).height == 0


def test_amendment_flag(tiny_cfg):
    ev = revisions.detect(tiny_cfg, write=False)
    assert ev.filter(pl.col("cik") == 2)["is_amendment"].item() is True
    assert ev.filter(pl.col("cik") == 1)["is_amendment"].item() is False


def test_scale_exponent_column(tiny_cfg):
    ev = revisions.detect(tiny_cfg, write=False)
    assert ev.filter(pl.col("cik") == 3)["scale_exponent"].item() == 6
    assert ev.filter(pl.col("cik") == 1)["scale_exponent"].item() is None


def test_null_then_value_is_not_a_revision(tiny_cfg):
    """先空值后有值是补报。空值不是"对一个数的观测"。"""
    ev = revisions.detect(tiny_cfg, write=False)
    assert ev.filter(pl.col("cik") == 4).height == 0
    ce = revisions.census(tiny_cfg, events=ev, write=False)
    d = ce.filter(pl.col("cik") == 4)
    assert d["n_obs"].item() == 1 and d["revised"].item() is False


def test_census_is_consistent_with_events(tiny_cfg):
    ev = revisions.detect(tiny_cfg, write=False)
    ce = revisions.census(tiny_cfg, events=ev, write=False)
    revised = ce.filter(pl.col("revised")).select(list(BUSINESS_KEY))
    have_events = ev.select(list(BUSINESS_KEY)).unique()
    assert revised.join(have_events, on=list(BUSINESS_KEY), how="anti").height == 0
    assert ce.filter((pl.col("n_distinct_val") == 1) & pl.col("revised")).height == 0


def test_from_zero_flag(tiny_cfg):
    ce = revisions.census(tiny_cfg, write=False)
    assert ce.filter(pl.col("cik") == 1)["from_zero"].item() is False
