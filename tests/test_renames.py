"""改名检测的纯逻辑部分：等价类合并。

检测本身要跑在全量数据上，不适合做单元测试；
但等价类合并是纯函数，而且传递链（A→B→C 必须并成一个类）是最容易写错的地方。
"""

import polars as pl

from secpit import renames


def _table(pairs, accepted=True):
    return pl.DataFrame(
        [{"old_tag": a, "new_tag": b, "accepted": accepted} for a, b in pairs]
    )


def test_single_pair_merges():
    m = renames.equivalence_classes(_table([("A", "B")]))
    assert m["A"] == m["B"]


def test_chain_merges_into_one_class():
    """A→B→C 必须并成一个类，否则跨两次改名的重述仍会漏掉。"""
    m = renames.equivalence_classes(_table([("A", "B"), ("B", "C")]))
    assert m["A"] == m["B"] == m["C"]


def test_fan_in_merges():
    """两个旧概念并入同一个新概念——实测 SalesRevenueNet 与
    SalesRevenueGoodsNet 都并入 RevenueFromContractWithCustomer…"""
    m = renames.equivalence_classes(_table([("old1", "new"), ("old2", "new")]))
    assert m["old1"] == m["old2"] == m["new"]


def test_rejected_pairs_are_not_merged():
    m = renames.equivalence_classes(_table([("A", "B")], accepted=False))
    assert m == {} or m.get("A") != m.get("B")


def test_disjoint_classes_stay_disjoint():
    m = renames.equivalence_classes(_table([("A", "B"), ("X", "Y")]))
    assert m["A"] == m["B"] and m["X"] == m["Y"] and m["A"] != m["X"]


def test_representative_is_deterministic():
    """类代表必须由集合本身决定，与输入顺序无关。"""
    m1 = renames.equivalence_classes(_table([("A", "B"), ("B", "C")]))
    m2 = renames.equivalence_classes(_table([("B", "C"), ("A", "B")]))
    assert m1 == m2


def test_empty_input():
    assert renames.equivalence_classes(pl.DataFrame(
        schema={"old_tag": pl.Utf8, "new_tag": pl.Utf8, "accepted": pl.Boolean})) == {}
