"""测试夹具：在临时目录里造一个小观测表，不碰真实数据。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from secpit import parse
from secpit.config import OBSERVATION_COLUMNS, Config


def _row(cik, tag, ddate, qtrs, uom, value, filed, adsh, form="10-Q",
         version="us-gaap/2024", period=None):
    period = period or ddate
    return {
        "cik": cik, "adsh": adsh, "form": form,
        "filed": filed, "accepted": dt.datetime.combine(filed, dt.time(16, 30)),
        "filed_year": filed.year,
        "period": period, "fy": period.year, "fp": "FY",
        "tag": tag, "version": version, "ddate": ddate, "qtrs": qtrs,
        "uom": uom, "value": value,
    }


D = dt.date


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Config:
    """一个含三次 vintage、一次修正案、一次单位跃迁、一个空值的小观测表。"""
    rows = [
        # 键 A：三次 vintage，值 100 → 100 → 130（只应产出 1 条事件）
        _row(1, "Revenues", D(2024, 12, 31), 4, "USD", 100.0, D(2025, 2, 1), "a-1"),
        _row(1, "Revenues", D(2024, 12, 31), 4, "USD", 100.0, D(2025, 5, 1), "a-2"),
        _row(1, "Revenues", D(2024, 12, 31), 4, "USD", 130.0, D(2025, 8, 1), "a-3"),
        # 键 B：修正案改写
        _row(2, "Assets", D(2024, 12, 31), 0, "USD", 500.0, D(2025, 2, 1), "b-1"),
        _row(2, "Assets", D(2024, 12, 31), 0, "USD", 550.0, D(2025, 3, 1), "b-2",
             form="10-K/A"),
        # 键 C：单位跃迁 73.6 → 73_600_000
        _row(3, "CommonStockSharesIssued", D(2024, 12, 31), 0, "shares",
             73.6, D(2025, 2, 1), "c-1"),
        _row(3, "CommonStockSharesIssued", D(2024, 12, 31), 0, "shares",
             73_600_000.0, D(2025, 6, 1), "c-2"),
        # 键 D：先空值后有值——补报，不是重述
        _row(4, "Liabilities", D(2024, 12, 31), 0, "USD", None, D(2025, 2, 1), "d-1"),
        _row(4, "Liabilities", D(2024, 12, 31), 0, "USD", 42.0, D(2025, 5, 1), "d-2"),
    ]

    # 一个能算 ROE 的迷你样本：三家公司各有 Assets / NetIncomeLoss / StockholdersEquity。
    # cik 10 的净利润在 5 月被改写 10 → 90，于是 D_PIT(3月) 与 D_rev 取值不同。
    for cik, assets, ni, eq in ((10, 1000.0, 10.0, 200.0),
                                (11, 900.0, 30.0, 300.0),
                                (12, 800.0, 50.0, 400.0)):
        rows += [
            _row(cik, "Assets", D(2024, 12, 31), 0, "USD", assets, D(2025, 2, 1),
                 f"u{cik}-1", form="10-K"),
            _row(cik, "StockholdersEquity", D(2024, 12, 31), 0, "USD", eq,
                 D(2025, 2, 1), f"u{cik}-1", form="10-K"),
            _row(cik, "NetIncomeLoss", D(2024, 12, 31), 4, "USD", ni,
                 D(2025, 2, 1), f"u{cik}-1", form="10-K"),
        ]
    rows.append(_row(10, "NetIncomeLoss", D(2024, 12, 31), 4, "USD", 90.0,
                     D(2025, 5, 1), "u10-2", form="10-K"))

    df = pl.DataFrame(rows).select(OBSERVATION_COLUMNS)

    cfg = Config.for_quarters("2025q1,2025q2", root=tmp_path)
    cfg.ensure_dirs()
    # 用真正的写入函数，而不是在这里复制一份分区布局——
    # 否则分区结构一改，夹具就会悄悄和实现脱节。
    parse._write_partitions(cfg, df, "tiny")
    return cfg


@pytest.fixture
def row():
    return _row


@pytest.fixture
def make_cfg(tmp_path: Path):
    """由任意观测行造一个临时库，供需要专门构造数据的测试使用。"""
    def _make(rows: list[dict]) -> Config:
        df = pl.DataFrame(rows).select(OBSERVATION_COLUMNS)
        cfg = Config.for_quarters("2025q1,2025q2", root=tmp_path)
        cfg.ensure_dirs()
        parse._write_partitions(cfg, df, "tiny")
        return cfg
    return _make
