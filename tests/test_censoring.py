"""滞后口径、左截断来源、标签合并后的同申报冲突。"""

import datetime as dt

import polars as pl

from secpit import revisions

D = dt.date


def test_two_lag_measures(tiny_cfg):
    """100 → 100 → 130：距上一次观测 92 天，距首次发布 181 天。"""
    ev = revisions.detect(tiny_cfg, write=False)
    a = ev.filter(pl.col("cik") == 1)
    assert a["lag_days"].item() == 92
    assert a["lag_since_first_days"].item() == 181


def test_first_observation_in_comparative_column_is_left_censored(make_cfg, row):
    """首次出现就在比较列里的事实，其原始申报不在库中，应记为左截断。"""
    rows = [
        # 键 X：上年末余额首次出现在一季度 10-Q 的比较列里（报告期 2025-03-31）
        row(30, "Assets", D(2024, 12, 31), 0, "USD", 10.0, D(2025, 5, 1), "x-1",
            period=D(2025, 3, 31)),
        row(30, "Assets", D(2024, 12, 31), 0, "USD", 11.0, D(2025, 8, 1), "x-2",
            period=D(2025, 6, 30)),
        # 键 Y：首次观测就是本期数
        row(31, "Assets", D(2025, 3, 31), 0, "USD", 20.0, D(2025, 5, 1), "y-1"),
        row(31, "Assets", D(2025, 3, 31), 0, "USD", 20.0, D(2025, 8, 1), "y-2",
            period=D(2025, 6, 30)),
    ]
    ce = revisions.census(make_cfg(rows), write=False)
    x = ce.filter(pl.col("cik") == 30)
    y = ce.filter(pl.col("cik") == 31)
    # 两者都晚于窗口首季，窗口口径不标记
    assert not x["left_censored_window"].item() and not y["left_censored_window"].item()
    assert x["first_is_comparative"].item() and x["left_censored"].item()
    assert not y["first_is_comparative"].item() and not y["left_censored"].item()


def _merged_rows(row):
    return [
        # cik 20：同一份申报里新旧标签取值矛盾 → 这份申报不可判定，整组丢弃
        row(20, "OldTag", D(2024, 12, 31), 4, "USD", 100.0, D(2025, 2, 1), "m-1"),
        row(20, "NewTag", D(2024, 12, 31), 4, "USD", 120.0, D(2025, 2, 1), "m-1"),
        row(20, "NewTag", D(2024, 12, 31), 4, "USD", 120.0, D(2025, 5, 1), "m-2"),
        # cik 21：同一份申报里新旧标签取值相同 → 留一行；次年真被改了
        row(21, "OldTag", D(2024, 12, 31), 4, "USD", 50.0, D(2025, 2, 1), "n-1"),
        row(21, "NewTag", D(2024, 12, 31), 4, "USD", 50.0, D(2025, 2, 1), "n-1"),
        row(21, "NewTag", D(2024, 12, 31), 4, "USD", 60.0, D(2025, 5, 1), "n-2"),
    ]


_TAG_MAP = {"OldTag": "NewTag", "NewTag": "NewTag"}


def test_merged_tags_in_one_filing_are_not_a_revision(make_cfg, row):
    cfg = make_cfg(_merged_rows(row))
    ev = revisions.detect(cfg, write=False, tag_map=_TAG_MAP)
    assert ev.filter(pl.col("adsh_old") == pl.col("adsh_new")).height == 0
    assert ev.filter(pl.col("cik") == 20).height == 0
    e21 = ev.filter(pl.col("cik") == 21)
    assert e21.height == 1
    assert (e21["v_old"].item(), e21["v_new"].item()) == (50.0, 60.0)

    ce = revisions.census(cfg, events=ev, write=False, tag_map=_TAG_MAP)
    assert ce.filter(pl.col("cik") == 20)["n_obs"].item() == 1
    assert ce.filter(pl.col("cik") == 21)["n_obs"].item() == 2


def test_merged_conflicts_are_counted(make_cfg, row):
    assert revisions.merged_conflicts(make_cfg(_merged_rows(row)), _TAG_MAP) == 1
