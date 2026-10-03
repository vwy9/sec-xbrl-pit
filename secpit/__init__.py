"""SEC XBRL point-in-time 重述普查与 vintage 微分演示。"""

import polars as pl

# revisions 把 tag / uom / version 转成 Categorical 压排序内存；Categorical
# 跨 DataFrame join 或比较时两侧必须共享同一份类别映射，所以按 polars 文档
# 的要求，在建立任何 Categorical 之前启用全局字符串缓存。
pl.enable_string_cache()

__version__ = "0.1.0"

from secpit.config import Config  # noqa: E402

__all__ = ["Config", "__version__"]
