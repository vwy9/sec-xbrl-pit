"""常量、路径与运行配置。"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

# SEC 要求限速 10 req/s，并在请求头声明含联系方式的 User-Agent，见
# https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
# 联系方式不写进代码，运行前由环境变量提供，例如
#   export SECPIT_USER_AGENT="Your Name you@example.com"
DEFAULT_USER_AGENT = os.environ.get("SECPIT_USER_AGENT", "")

FSDS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def sec_headers(user_agent: str, host: str = "www.sec.gov") -> dict[str, str]:
    """按 SEC 官方样例构造请求头。host 需与目标域一致。"""
    if not user_agent:
        raise RuntimeError(
            "SEC 要求请求头声明含联系方式的 User-Agent：请先设置环境变量 SECPIT_USER_AGENT，"
            '例如 export SECPIT_USER_AGENT="Your Name you@example.com"')
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }


# 定期报告及其修正案。排除 20-F/40-F（IFRS taxonomy，tag 体系不同）、
# 8-K（携带 XBRL 但非定期报告）、S-1/S-4（注册书）。
FORM_WHITELIST = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A"})

# 0 时点存量 / 1 单季 / 2 半年累计 / 3 前三季累计 / 4 年度。
# 实测长尾（qtrs 5..124）合计不足 200 行，全部丢弃。
QTRS_WHITELIST = frozenset({0, 1, 2, 3, 4})

# 统计阶段施加的相对幅度阈值档位。0 表示"任何数值差异"。
THRESHOLDS = (0.0, 1e-6, 1e-4, 1e-2)

# 视为单位跃迁（千 / 百万 / 十亿及其倒数）的十进制指数。
# revisions 表存的是实际指数，此集合只在统计阶段用于筛选，故可事后调整。
SCALE_EXPONENTS = frozenset({-9, -6, -3, 3, 6, 9})

# 右截断下限：观测窗口短于此天数的键视为"尚无充分被重述的机会"。
RIGHT_TRUNCATION_DAYS = 365

# 判定"首次观测是比较列"的间隔：期末日早于所在申报的报告期超过此天数，
# 即该数属于更早的期间。FSDS 的 ddate 与 period 都已四舍五入到月末，
# 同期两者相等，上一季度相差约 90 天，31 天足以把两者分开。
COMPARATIVE_GAP_DAYS = 31

# "重述率最高的字段"排行的最小样本量，样本太小时率的噪声过大。
MIN_TAG_SAMPLE = 1000

# 判定单位跃迁时，log10(ratio) 与最近整数的最大允许偏离。
SCALE_LOG_TOL = 1e-6

# 预期表头（取自 2025Q1，与 2021Q3 完全一致）。解析前逐张断言，不一致即报错。
SUB_COLUMNS = (
    "adsh", "cik", "name", "sic", "countryba", "stprba", "cityba", "zipba",
    "bas1", "bas2", "baph", "countryma", "stprma", "cityma", "zipma", "mas1",
    "mas2", "countryinc", "stprinc", "ein", "former", "changed", "afs", "wksi",
    "fye", "form", "period", "fy", "fp", "filed", "accepted", "prevrpt",
    "detail", "instance", "nciks", "aciks",
)

NUM_COLUMNS = (
    "adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg",
    "value", "footnote",
)

TAG_COLUMNS = (
    "tag", "version", "custom", "abstract", "datatype", "iord", "crdr",
    "tlabel", "doc",
)

# 观测事实表的列顺序。
OBSERVATION_COLUMNS = (
    "cik", "adsh", "form", "filed", "accepted", "filed_year",
    "period", "fy", "fp",
    "tag", "version", "ddate", "qtrs", "uom", "value",
)

# 观测表主键：单份 filing 内唯一（实测 157.6 万键 0 重复）。
PRIMARY_KEY = ("adsh", "tag", "ddate", "qtrs", "uom")

# 业务键：一个业务键对应多次观测，该序列即 vintage 历史。
BUSINESS_KEY = ("cik", "tag", "ddate", "qtrs", "uom")

# Λ 演示用的三个概念
TAG_NET_INCOME = "NetIncomeLoss"        # qtrs=4，年度流量
TAG_EQUITY = "StockholdersEquity"       # qtrs=0，时点存量
TAG_ASSETS = "Assets"                   # qtrs=0，时点存量，用于确定样本

DEMO_TAGS = (TAG_NET_INCOME, TAG_EQUITY, TAG_ASSETS)


def parse_quarter(quarter: str) -> tuple[int, int]:
    """`"2023q3"` → `(2023, 3)`。"""
    q = quarter.strip().lower()
    if len(q) != 6 or q[4] != "q" or not q[:4].isdigit() or q[5] not in "1234":
        raise ValueError(f"季度格式应形如 2023q3，收到 {quarter!r}")
    return int(q[:4]), int(q[5])


def quarter_bounds(quarter: str) -> tuple[dt.date, dt.date]:
    """返回该季度的首日与末日。"""
    year, q = parse_quarter(quarter)
    start = dt.date(year, 3 * q - 2, 1)
    end_month = 3 * q
    if end_month == 12:
        end = dt.date(year, 12, 31)
    else:
        end = dt.date(year, end_month + 1, 1) - dt.timedelta(days=1)
    return start, end


def next_quarter(quarter: str) -> str:
    year, q = parse_quarter(quarter)
    return f"{year + 1}q1" if q == 4 else f"{year}q{q + 1}"


def quarters_between(start: str, end: str) -> tuple[str, ...]:
    """闭区间内的季度序列。"""
    if (parse_quarter(start)) > (parse_quarter(end)):
        raise ValueError(f"起始季度 {start} 晚于结束季度 {end}")
    out, cur = [], start.strip().lower()
    end = end.strip().lower()
    while True:
        out.append(cur)
        if cur == end:
            return tuple(out)
        cur = next_quarter(cur)


def quarter_ends(quarters: tuple[str, ...]) -> tuple[dt.date, ...]:
    return tuple(quarter_bounds(q)[1] for q in quarters)


# FSDS 的第一个季度。2009q2 之前没有 XBRL 数据集。
DEFAULT_START = "2009q2"
DEFAULT_END = "2026q2"

# Λ 演示最多取多少个 as-of 日期。逐季度跑是 68 × 2 个世界 × 3 个程序 = 408 次
# 程序调用，每次都在数千万行上查询；均匀抽样到 24 个，曲线形状不受影响。
MAX_ASOF_DATES = 24


@dataclass
class Config:
    """一次运行的全部可配置项。"""

    quarters: tuple[str, ...]
    root: Path
    user_agent: str = DEFAULT_USER_AGENT

    # 请求之间的固定间隔。0.15s ≈ 6.7 req/s，低于 SEC 的 10 req/s 上限。
    rate_limit_sleep: float = 0.15
    max_retries: int = 4

    # Λ 演示的截面样本规模（按可见总资产取前 N 家）。
    universe_size: int = 200
    # Λ_cell 的相对差异容差。
    lambda_tol: float = 1e-6

    # 重述检出与普查的分区数。按 cik 取模切分——重述完全发生在业务键内部，
    # 而业务键含 cik，故分区之间无依赖，结果与不分区逐位相同。
    # 分区只影响峰值内存：全量 68 季度约 6000 万行，不分区排序需要约 14GB。
    key_partitions: int = 16
    right_truncation_days: int = RIGHT_TRUNCATION_DAYS

    # 留空则由窗口派生，见对应 property。
    asof_dates_override: tuple[dt.date, ...] | None = None
    truncation_points_override: tuple[dt.date, ...] | None = None

    # 路径

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def parquet_dir(self) -> Path:
        return self.root / "data" / "parquet"

    @property
    def out_dir(self) -> Path:
        return self.root / "data" / "out"

    @property
    def figures_dir(self) -> Path:
        """图是交付物不是中间数据，落在版本库里而非 data/out。"""
        return self.root / "figures"

    @property
    def observations_dir(self) -> Path:
        return self.parquet_dir / "observations"

    @property
    def observations_glob(self) -> str:
        return str(self.observations_dir / "**" / "*.parquet")

    def bucket_glob(self, bucket: int) -> str:
        """某个 cik 桶的全部分区文件，跨所有年份。"""
        return str(self.observations_dir / "*" / f"cik_bucket={bucket}" / "*.parquet")

    @property
    def manifest_path(self) -> Path:
        return self.raw_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.parquet_dir, self.out_dir,
                  self.observations_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    # 窗口

    @property
    def window_start(self) -> dt.date:
        return quarter_bounds(self.quarters[0])[0]

    @property
    def window_end(self) -> dt.date:
        return quarter_bounds(self.quarters[-1])[1]

    @property
    def left_censor_cutoff(self) -> dt.date:
        """窗口第二个季度的首日。

        最早观测早于此日的业务键标记为左截断——其原始申报可能落在窗口之前，
        vintage 序列是从半截开始的。这只是左截断的一个来源：最早观测在更晚的
        季度，也不能说明它就是原始申报——若它出现在比较列里，所属期间的原始申报
        同样不在库中。后一种情况在普查中另行判定，见 revisions._census_partition。
        """
        if len(self.quarters) < 2:
            return self.window_end
        return quarter_bounds(self.quarters[1])[0]

    @property
    def asof_dates(self) -> tuple[dt.date, ...]:
        """Λ 演示的 as-of 日期序列。

        取窗口内第 3 个季度末起、至倒数第 2 个季度末止的各季度末：
        前面留出季度使 D_PIT 有可查的历史，后面留出季度使重述有机会发生。
        """
        if self.asof_dates_override is not None:
            return self.asof_dates_override
        ends = quarter_ends(self.quarters)
        # 末个季度末必须排除：D_PIT(窗口末) 与 D_rev 内容相同，Λ 平凡为 0，
        # 画进曲线里会造成"泄漏消失了"的错觉。
        if len(ends) <= 3:                    # 开发期短窗口：留到倒数第二个
            return ends[:-1] or ends
        picked = ends[2:len(ends) - 1]
        if len(picked) <= MAX_ASOF_DATES:
            return picked
        step = (len(picked) - 1) / (MAX_ASOF_DATES - 1)
        return tuple(picked[round(i * step)] for i in range(MAX_ASOF_DATES))

    @property
    def truncation_points(self) -> tuple[dt.date, ...]:
        """截断检验的 5 个 ddate 截断点，取自 as-of 日期序列的均匀抽样。"""
        if self.truncation_points_override is not None:
            return self.truncation_points_override
        dates = self.asof_dates
        if len(dates) <= 5:
            return dates
        step = (len(dates) - 1) / 4
        return tuple(dates[round(i * step)] for i in range(5))

    # 构造

    @classmethod
    def default(cls, root: Path | str | None = None) -> "Config":
        return cls(
            quarters=quarters_between(DEFAULT_START, DEFAULT_END),
            root=Path(root) if root else Path(__file__).resolve().parent.parent,
        )

    @classmethod
    def for_quarters(cls, quarters: str | list[str], root: Path | str | None = None) -> "Config":
        """`"2025q1,2025q2"` 或列表 → Config。"""
        if isinstance(quarters, str):
            quarters = [q for q in quarters.replace(" ", "").split(",") if q]
        cfg = cls.default(root)
        cfg.quarters = tuple(q.lower() for q in quarters)
        return cfg
