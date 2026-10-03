import datetime as dt

import numpy as np

from secpit import demo, store


def test_rankdata_average_ties():
    assert list(demo._rankdata(np.array([10.0, 20.0, 20.0, 30.0]))) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_bounds():
    assert demo._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert demo._spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert demo._spearman([1, 2, 3, 4], [1, 4, 9, 16]) == 1.0   # 单调非线性


def test_spearman_identical_is_exactly_one():
    """必须精确为 1.0——否则 1 − ρ 会给出一个非零的假 Λ_rank。"""
    a = np.random.default_rng(0).normal(size=50)
    assert demo._spearman(a, a) == 1.0


def test_latest_view_hides_filed(tiny_cfg):
    """陷阱接口不返回 filed 列——这是它成为陷阱的原因。"""
    db = demo.world_rev(store.load(tiny_cfg))
    got = db.latest("Revenues", 4, dt.date(2025, 12, 31))
    assert "filed" not in got.columns
    assert "filed" in db.as_of("Revenues", 4, dt.date(2025, 12, 31)).columns


def test_as_of_handle_respects_filed(tiny_cfg):
    db = demo.world_rev(store.load(tiny_cfg))
    assert db.as_of("Revenues", 4, dt.date(2025, 3, 1))["value"].item() == 100.0
    assert db.as_of("Revenues", 4, dt.date(2025, 9, 1))["value"].item() == 130.0


def test_world_pit_is_a_subset(tiny_cfg):
    lf = store.load(tiny_cfg)
    full = demo.world_rev(lf).frame().collect().height
    early = demo.world_pit(lf, dt.date(2025, 2, 15)).frame().collect().height
    assert 0 < early < full


def test_lambda_separates_the_two_programs(tiny_cfg):
    """核心断言：correct 的 Λ 为 0，naive 的 Λ 大于 0。"""
    tiny_cfg.universe_size = 10
    tiny_cfg.asof_dates_override = (dt.date(2025, 3, 1), dt.date(2025, 6, 1))
    tiny_cfg.truncation_points_override = (dt.date(2025, 6, 30),)
    res = demo.run(tiny_cfg, write=False)
    by = {r["program"]: r for r in res.to_dicts()}
    assert by["correct"]["lambda_cell"] == 0.0
    assert by["correct"]["verdict"] == "clean"
    assert by["naive"]["lambda_cell"] > 0.0
    assert by["naive"]["truncation_pass"] is True
    assert by["naive"]["verdict"] == "silent_leak"


def test_verdict_labels():
    assert demo._verdict(True, True) == "clean"
    assert demo._verdict(True, False) == "silent_leak"
    assert demo._verdict(False, True) == "obvious_bug"
    assert demo._verdict(False, False) == "double_failure"
