#!/usr/bin/env python3
"""试验：从 OpenStreetMap 经 Overpass API 拉取一小片「highway」路网折线。

依赖：仅标准库。运行: python osm_fetch_demo.py

把下面 BBOX 改成你关心的区域（越小越快）。若默认镜像超时，可改 OVERPASS_URL。
"""

from __future__ import annotations

import csv
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Overpass 使用 south, west, north, east（纬度、经度的最小外接矩形）
# 默认：爱丁堡老城附近约 400m×350m，数据量小、易成功
BBOX_SOUTH, BBOX_WEST, BBOX_NORTH, BBOX_EAST = 55.9448, -3.1915, 55.9478, -3.1865

# 主站偶发 504，可多试几个镜像
OVERPASS_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)

OUT_CSV = "osm_sample_segments.csv"


@dataclass(frozen=True)
class Segment:
    lon1: float
    lat1: float
    lon2: float
    lat2: float
    highway: str


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """近似球面距离（米），足够做统计。"""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def build_overpass_query(south: float, west: float, north: float, east: float) -> str:
    # 仅道路中心线；bbox 足够小时不必再筛道路等级
    return f"""[out:json][timeout:45];
(
  way["highway"]({south},{west},{north},{east});
);
out geom;
"""


def fetch_overpass(query: str, urls: Sequence[str] = OVERPASS_URLS) -> Dict[str, Any]:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_err: Optional[BaseException] = None
    for url in urls:
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "User-Agent": "DataStructure-course-demo/1.0 (OSM Overpass test)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def segments_from_osm_json(data: Dict[str, Any]) -> List[Segment]:
    segs: List[Segment] = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry")
        if not isinstance(geom, list) or len(geom) < 2:
            continue
        hw = el.get("tags", {}).get("highway", "")
        for a, b in zip(geom, geom[1:]):
            lon1, lat1 = float(a["lon"]), float(a["lat"])
            lon2, lat2 = float(b["lon"]), float(b["lat"])
            if (lon1, lat1) == (lon2, lat2):
                continue
            segs.append(Segment(lon1, lat1, lon2, lat2, hw))
    return segs


def write_segments_csv(path: str, segments: Sequence[Segment]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon1", "lat1", "lon2", "lat2", "highway", "length_m"])
        for s in segments:
            d = _haversine_m(s.lat1, s.lon1, s.lat2, s.lon2)
            w.writerow([s.lon1, s.lat1, s.lon2, s.lat2, s.highway, f"{d:.2f}"])


def summarize(segments: Iterable[Segment]) -> Tuple[int, float, Dict[str, int]]:
    by_hw: Dict[str, int] = {}
    total_m = 0.0
    n = 0
    for s in segments:
        n += 1
        total_m += _haversine_m(s.lat1, s.lon1, s.lat2, s.lon2)
        by_hw[s.highway] = by_hw.get(s.highway, 0) + 1
    return n, total_m, by_hw


def main() -> int:
    q = build_overpass_query(BBOX_SOUTH, BBOX_WEST, BBOX_NORTH, BBOX_EAST)
    print(f"请求 Overpass（依次尝试）: {', '.join(OVERPASS_URLS)}")
    print(f"BBOX (S,W,N,E) = {BBOX_SOUTH}, {BBOX_WEST}, {BBOX_NORTH}, {BBOX_EAST}")
    try:
        data = fetch_overpass(q)
    except urllib.error.HTTPError as e:
        print(f"HTTP 错误: {e.code} {e.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"网络错误: {e.reason}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}", file=sys.stderr)
        return 1

    segs = segments_from_osm_json(data)
    n, meters, by_hw = summarize(segs)
    print(f"折线段数: {n}，近似总长: {meters:.1f} m")
    if by_hw:
        top = sorted(by_hw.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print("道路类型 Top:", ", ".join(f"{k}={v}" for k, v in top))

    write_segments_csv(OUT_CSV, segs)
    print(f"已写入: {OUT_CSV}（可用 Excel 或 python fleet_osm.py --csv {OUT_CSV}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
