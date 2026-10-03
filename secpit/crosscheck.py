"""用 Company Facts 这个独立来源抽样校验本库的 vintage 序列。

SEC 对 FSDS 的免责声明明说抽取与编纂过程可能引入误差，所以单凭 FSDS
无法区分「真实重述」与「抽取误差」；两个独立来源在同一条 vintage 序列上
一致才排除后者，且只核对单个已知案例不足以排除系统性误差，需要随机抽样。

口径对齐上要注意：FSDS 的 ddate 四舍五入到最近月末，Company Facts 的 end
是实际日期（财年末 2024-09-28 在 FSDS 里是 2024-09-30），两边都归一到
月末才能对上，否则会产生大量假性不匹配。
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import logging
import random
import time

import polars as pl
import requests

from secpit.config import COMPANYFACTS_URL, Config, sec_headers

log = logging.getLogger(__name__)

# 判定两个取值一致的相对容差。
VALUE_RTOL = 1e-9


def _month_end(d: dt.date) -> dt.date:
    """归一到所在月的月末；若已过该月中旬则取本月末，否则取上月末。

    FSDS 的规则是"四舍五入到最近月末"，等价于：日期在下半月归本月末，
    在上半月归上月末。
    """
    if d.day >= 15:
        return dt.date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
    prev = d.replace(day=1) - dt.timedelta(days=1)
    return prev


def _qtrs(start: str | None, end: dt.date) -> int:
    """由起止日推出 FSDS 的 qtrs：时点为 0，否则按月跨度除以 3 取整。"""
    if not start:
        return 0
    s = dt.date.fromisoformat(start)
    months = (end.year - s.year) * 12 + (end.month - s.month)
    return max(1, round(months / 3))


def fetch_companyfacts(cfg: Config, cik: int) -> dict | None:
    cache = cfg.raw_dir / "companyfacts"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"CIK{cik:010d}.json"
    if not path.exists():
        url = COMPANYFACTS_URL.format(cik=cik)
        try:
            resp = requests.get(
                url, headers=sec_headers(cfg.user_agent, "data.sec.gov"), timeout=120)
        except requests.RequestException as exc:            # noqa: BLE001
            log.warning("CIK %d 请求失败：%s", cik, exc)
            return None
        time.sleep(cfg.rate_limit_sleep)
        if resp.status_code != 200:
            log.warning("CIK %d → HTTP %d", cik, resp.status_code)
            return None
        path.write_bytes(resp.content)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("CIK %d 的缓存无法解析", cik)
        return None


def _companyfacts_rows(payload: dict, cik: int) -> pl.DataFrame:
    """把 Company Facts 的嵌套结构摊平成与本库同构的观测行。"""
    rows = []
    for concept, cdef in payload.get("facts", {}).get("us-gaap", {}).items():
        for unit, arr in cdef.get("units", {}).items():
            for r in arr:
                try:
                    end = dt.date.fromisoformat(r["end"])
                except (KeyError, ValueError):
                    continue
                rows.append({
                    "cik": cik,
                    "tag": concept,
                    "uom": unit,
                    "ddate": _month_end(end),
                    "qtrs": _qtrs(r.get("start"), end),
                    "filed": dt.date.fromisoformat(r["filed"]),
                    "value": float(r["val"]) if r.get("val") is not None else None,
                })
    schema = {"cik": pl.Int64, "tag": pl.Utf8, "uom": pl.Utf8, "ddate": pl.Date,
              "qtrs": pl.Int8, "filed": pl.Date, "value": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def sample_ciks(cfg: Config, n: int, seed: int = 0) -> list[int]:
    """从观测表中随机抽 n 家公司，固定种子保证可复现。"""
    ciks = (pl.scan_parquet(cfg.observations_glob)
              .select("cik").unique()
              .collect(engine="streaming")["cik"].to_list())
    rng = random.Random(seed)
    return sorted(rng.sample(ciks, min(n, len(ciks))))


def run(cfg: Config, n_ciks: int = 100, seed: int = 0,
        write: bool = True) -> dict:
    """抽样比对，返回覆盖率与一致率，并把冲突清单落盘。"""
    cfg.ensure_dirs()
    ciks = sample_ciks(cfg, n_ciks, seed)
    log.info("抽样 %d 家公司（seed=%d）", len(ciks), seed)

    ours = (
        pl.scan_parquet(cfg.observations_glob)
          .filter(pl.col("cik").is_in(ciks))
          .select("cik", "tag", "uom", "ddate", "qtrs", "filed", "value")
          .filter(pl.col("value").is_not_null())
          .collect(engine="streaming")
    )

    theirs = []
    n_fetched = 0
    for cik in ciks:
        payload = fetch_companyfacts(cfg, cik)
        if payload is None:
            continue
        n_fetched += 1
        theirs.append(_companyfacts_rows(payload, cik))
    cf = pl.concat(theirs) if theirs else _companyfacts_rows({}, 0)
    log.info("取到 %d/%d 家的 Company Facts，共 %d 行", n_fetched, len(ciks), cf.height)

    obs_key = ["cik", "tag", "uom", "ddate", "qtrs", "filed"]

    # 两边都必须先在连接键上去重，否则内连接会产生笛卡尔积：
    # 同一天可能有多份申报，各自对同一事实给出不同的值，
    # 直接连接会把 m×n 种组合都算成"分歧"，把一致率压得毫无意义。
    # 与解析层同一条纪律：同键取值矛盾者视为不可判定，单独计数而不计入一致率。
    def _dedup(df: pl.DataFrame, col: str) -> tuple[pl.DataFrame, int]:
        g = df.group_by(obs_key).agg(
            pl.col("value").n_unique().alias("nv"),
            pl.col("value").first().alias(col))
        ambiguous = g.filter(pl.col("nv") > 1).height
        return g.filter(pl.col("nv") == 1).drop("nv"), ambiguous

    ours_u, amb_ours = _dedup(ours, "value")
    cf_u, amb_cf = _dedup(cf, "value_cf")

    joined = ours_u.join(cf_u, on=obs_key, how="inner")
    den = pl.max_horizontal(
        pl.col("value").abs(), pl.col("value_cf").abs(), pl.lit(1.0))
    joined = joined.with_columns(
        ((pl.col("value_cf") - pl.col("value")).abs() / den).alias("rel"))
    conflicts = joined.filter(pl.col("rel") > VALUE_RTOL)

    summary = {
        "seed": seed,
        "n_ciks_sampled": len(ciks),
        "n_ciks_fetched": n_fetched,
        "our_rows": ours.height,
        "our_unique_keys": ours_u.height,
        "our_ambiguous_keys": amb_ours,
        "companyfacts_rows": cf.height,
        "companyfacts_unique_keys": cf_u.height,
        "companyfacts_ambiguous_keys": amb_cf,
        "matched_observations": joined.height,
        "match_rate_of_ours": joined.height / ours_u.height if ours_u.height else 0.0,
        "value_agreement": (1 - conflicts.height / joined.height) if joined.height else 0.0,
        "n_conflicts": conflicts.height,
        "conflict_rel_median": (float(conflicts["rel"].median())
                                if conflicts.height else 0.0),
    }

    if write:
        (cfg.out_dir / "crosscheck_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (conflicts.head(5000)
                  .select(*obs_key, "value", "value_cf", "rel")
                  .write_csv(cfg.out_dir / "crosscheck_conflicts.csv"))
        log.info("crosscheck → %s", cfg.out_dir / "crosscheck_summary.json")
    return summary
