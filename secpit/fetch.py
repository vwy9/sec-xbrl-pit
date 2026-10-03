"""下载 SEC 季度 ZIP，维护本地缓存与清单。

每个季度至多发一次请求，请求之间固定间隔，远低于 SEC 的 10 req/s 上限。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
from pathlib import Path

import requests

from secpit.config import FSDS_URL, Config, sec_headers

log = logging.getLogger(__name__)

_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """重试耗尽后仍未取得文件。"""


def load_manifest(cfg: Config) -> dict[str, dict]:
    if not cfg.manifest_path.exists():
        return {}
    with cfg.manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def save_manifest(cfg: Config, manifest: dict[str, dict]) -> None:
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.manifest_path.with_suffix(".json.part")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(cfg.manifest_path)


def zip_path(cfg: Config, quarter: str) -> Path:
    return cfg.raw_dir / f"{quarter}.zip"


def needs_download(cfg: Config, quarter: str, manifest: dict[str, dict]) -> bool:
    """清单无记录、本地文件缺失、或大小与清单不符 → 需要下载。"""
    record = manifest.get(quarter)
    path = zip_path(cfg, quarter)
    if record is None or record.get("status") != "ok":
        return True
    if not path.exists():
        return True
    return path.stat().st_size != record.get("bytes")


def _download_one(cfg: Config, quarter: str) -> dict:
    """下载单个季度。写 .part 再原子改名，使中断不留下半个可信 ZIP。"""
    url = FSDS_URL.format(quarter=quarter)
    dest = zip_path(cfg, quarter)
    part = dest.with_suffix(".zip.part")
    headers = sec_headers(cfg.user_agent)

    last_error: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            log.info("GET %s (attempt %d/%d) UA=%r",
                     url, attempt, cfg.max_retries, cfg.user_agent)
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code != 200:
                    raise DownloadError(f"{quarter}: HTTP {resp.status_code}")
                digest = hashlib.sha256()
                size = 0
                with part.open("wb") as f:
                    for chunk in resp.iter_content(_CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            part.replace(dest)
            log.info("saved %s (%.1f MB)", dest.name, size / 1e6)
            return {
                "url": url,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "status": "ok",
            }
        except Exception as exc:                      # noqa: BLE001 - 统一退避重试
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt == cfg.max_retries:
                break
            backoff = cfg.rate_limit_sleep * (2 ** attempt)
            log.warning("%s 失败（%s），%.2fs 后重试", quarter, exc, backoff)
            time.sleep(backoff)

    raise DownloadError(f"{quarter}: 重试 {cfg.max_retries} 次仍失败") from last_error


def download(cfg: Config, quarters: tuple[str, ...] | None = None) -> dict:
    """下载缺失的季度，返回本次动作摘要。已完整的季度跳过。"""
    cfg.ensure_dirs()
    quarters = quarters or cfg.quarters
    manifest = load_manifest(cfg)
    downloaded, skipped = [], []

    for i, quarter in enumerate(quarters):
        if not needs_download(cfg, quarter, manifest):
            skipped.append(quarter)
            log.info("skip %s（已存在且大小与清单一致）", quarter)
            continue
        if i > 0:
            time.sleep(cfg.rate_limit_sleep)          # 节流：低于 10 req/s
        manifest[quarter] = _download_one(cfg, quarter)
        save_manifest(cfg, manifest)
        downloaded.append(quarter)

    total = sum(manifest[q]["bytes"] for q in quarters if q in manifest)
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "quarters": list(quarters),
        "total_bytes": total,
    }
