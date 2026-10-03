"""Company Facts 与本库的口径对齐逻辑。

这两个函数是交叉校验的正确性基础：对不齐就会产生大量假性不匹配，
把一致率压到毫无意义。
"""

import datetime as dt

from secpit import crosscheck as cc


def test_month_end_rounds_late_dates_forward():
    """FSDS 把 ddate 四舍五入到最近月末；下半月归本月末。"""
    assert cc._month_end(dt.date(2024, 9, 28)) == dt.date(2024, 9, 30)
    assert cc._month_end(dt.date(2024, 12, 31)) == dt.date(2024, 12, 31)
    assert cc._month_end(dt.date(2024, 2, 29)) == dt.date(2024, 2, 29)


def test_month_end_rounds_early_dates_back():
    """上半月归上月末。"""
    assert cc._month_end(dt.date(2024, 10, 2)) == dt.date(2024, 9, 30)
    assert cc._month_end(dt.date(2024, 1, 3)) == dt.date(2023, 12, 31)


def test_qtrs_instant_is_zero():
    assert cc._qtrs(None, dt.date(2024, 12, 31)) == 0
    assert cc._qtrs("", dt.date(2024, 12, 31)) == 0


def test_qtrs_from_duration():
    assert cc._qtrs("2024-10-01", dt.date(2024, 12, 31)) == 1   # 单季
    assert cc._qtrs("2024-07-01", dt.date(2024, 12, 31)) == 2   # 半年
    assert cc._qtrs("2024-04-01", dt.date(2024, 12, 31)) == 3   # 前三季
    assert cc._qtrs("2024-01-01", dt.date(2024, 12, 31)) == 4   # 全年


def test_qtrs_never_zero_for_a_duration():
    """有起始日就是流量，qtrs 至少为 1——0 专表时点存量。"""
    assert cc._qtrs("2024-12-01", dt.date(2024, 12, 31)) >= 1


def test_companyfacts_rows_flattens_and_normalises():
    payload = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2024-01-01", "end": "2024-12-28", "val": 100.0, "filed": "2025-02-01"},
        {"end": "2024-12-28", "val": 50.0, "filed": "2025-02-01"},
    ]}}}}}
    df = cc._companyfacts_rows(payload, cik=42)
    assert df.height == 2
    assert set(df["ddate"].to_list()) == {dt.date(2024, 12, 31)}   # 已归一到月末
    assert sorted(df["qtrs"].to_list()) == [0, 4]                  # 时点与年度各一


def test_companyfacts_rows_skips_malformed():
    payload = {"facts": {"us-gaap": {"X": {"units": {"USD": [
        {"end": "not-a-date", "val": 1.0, "filed": "2025-02-01"},
    ]}}}}}
    assert cc._companyfacts_rows(payload, cik=1).height == 0


def test_companyfacts_rows_empty_payload():
    assert cc._companyfacts_rows({}, cik=1).height == 0
