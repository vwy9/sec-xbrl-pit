"""命令行入口：build / census / figure / demo。"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from secpit.config import Config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _config(args) -> Config:
    cfg = (Config.for_quarters(args.quarters, args.root) if args.quarters
           else Config.default(args.root))
    cfg.ensure_dirs()
    return cfg


def cmd_build(args) -> int:
    from secpit import fetch, parse, store
    cfg = _config(args)
    print(f"季度区间 {cfg.quarters[0]}–{cfg.quarters[-1]}（{len(cfg.quarters)} 个）")
    got = fetch.download(cfg)
    print(f"下载 {len(got['downloaded'])} 个，跳过 {len(got['skipped'])} 个，"
          f"合计 {got['total_bytes']/1e9:.2f} GB")
    counts = parse.parse_all(cfg)
    print(f"解析 {len(counts)} 个季度，保留 {sum(c.kept for c in counts):,} 行观测")
    s = store.summary(cfg)
    print(json.dumps({k: s[k] for k in
                      ("rows", "business_keys", "companies", "tags", "filings",
                       "filed_min", "filed_max", "max_group_size")},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_census(args) -> int:
    import gc

    from secpit import revisions, stats
    cfg = _config(args)

    # 三步跑在同一个进程里，中间结果必须显式释放再进入下一步。
    # census 返回的普查表在全量窗口下有 4177 万行；一直持有它到 stats 跑完，
    # 会让整条命令的峰值比三步各自的峰值高出约 1.8GB。
    # 两张表都已落盘，这里只需要它们的行数。
    ev = revisions.detect(cfg)
    n_events = ev.height
    n_keys = revisions.census(cfg, events=ev, return_frame=False)
    del ev
    gc.collect()

    paths = stats.compute(cfg)
    print(f"事件 {n_events:,} 条 | 普查 {n_keys:,} 键 | 统计表 {len(paths)} 张")
    for name, p in paths.items():
        print(f"  {name:28s} {p}")
    return 0


def cmd_renames(args) -> int:
    """推导 taxonomy 改名等价类，并对比修正前后的重述率。"""
    import gc

    from secpit import renames, revisions
    cfg = _config(args)
    r = renames.detect(cfg)
    tag_map = renames.equivalence_classes(r)
    n_ok = int(r["accepted"].sum()) if r.height else 0
    print(f"改名候选 {r.height} 条，采纳 {n_ok} 条 → "
          f"{len(tag_map)} 个标签合并为 {len(set(tag_map.values()))} 个等价类")
    if not tag_map:
        return 0

    n_conflict = revisions.merged_conflicts(cfg, tag_map)
    print(f"  合并后同一份申报内取值矛盾、整组丢弃的 (业务键, adsh) 组：{n_conflict:,}")
    ev = revisions.detect(cfg, tag_map=tag_map)
    revisions.census(cfg, events=ev, return_frame=False, tag_map=tag_map)
    del ev
    gc.collect()

    import polars as pl
    for label, f in (("raw", "census.parquet"), ("canonical", "census_canonical.parquet")):
        path = cfg.out_dir / f
        if not path.exists():
            continue
        m = pl.scan_parquet(path).filter(pl.col("n_obs") >= 2)
        d = m.select(pl.len()).collect(engine="streaming").item()
        v = m.filter(pl.col("revised")).select(pl.len()).collect(engine="streaming").item()
        print(f"  {label:10s} 重述率 {v/d:.4%}  （{v:,} / {d:,}）")
    return 0


def cmd_crosscheck(args) -> int:
    from secpit import crosscheck
    cfg = _config(args)
    s = crosscheck.run(cfg, n_ciks=args.n_ciks, seed=args.seed)
    for k, v in s.items():
        print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    return 0


def cmd_demo(args) -> int:
    from secpit import demo
    cfg = _config(args)
    res = demo.run(cfg)
    print(res)
    leak = res.filter(res["verdict"] == "silent_leak")
    if leak.height:
        print(f"\n「截断通过但版本微分失败」: {leak['program'].to_list()}")
    return 0


def cmd_figure(args) -> int:
    from secpit import figures
    cfg = _config(args)
    for p in figures.build_all(cfg, args.censoring):
        print(f"  {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="secpit", description=__doc__)
    ap.add_argument("--quarters", help="逗号分隔，如 2025q1,2025q2；缺省用默认窗口")
    ap.add_argument("--root", help="项目根目录")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 公共参数在每个子命令上重复注册一次，使 `-v census` 与 `census -v` 都能用。
    # 子命令侧默认 SUPPRESS：没在子命令里给时不要用默认值把顶层的值盖掉。
    def _add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--quarters", default=argparse.SUPPRESS)
        sp.add_argument("--root", default=argparse.SUPPRESS)
        sp.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS)
        return sp

    _add("build", "下载 + 解析 + 主键断言").set_defaults(fn=cmd_build)
    _add("census", "事件检出 + 普查 + 统计").set_defaults(fn=cmd_census)
    _add("renames", "推导 taxonomy 改名等价类并对比修正前后").set_defaults(fn=cmd_renames)
    x = _add("crosscheck", "用 Company Facts 抽样校验 vintage 序列")
    x.add_argument("--n-ciks", type=int, default=100)
    x.add_argument("--seed", type=int, default=0)
    x.set_defaults(fn=cmd_crosscheck)
    _add("demo", "两个世界 × 三个程序 × 两个检验").set_defaults(fn=cmd_demo)
    f = _add("figure", "两张图")
    f.add_argument("--censoring", default="all",
                   choices=["all", "drop_right", "drop_left", "drop_both"])
    f.set_defaults(fn=cmd_figure)

    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
