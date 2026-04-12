#!/usr/bin/env python3
"""
OSM 路网共用逻辑：路段结构、Haversine、Overpass 拉取、CSV、RoadGraph（tuple 顶点）。

供 fleet_osm.py 等使用；不包含独立 CLI。
"""

from __future__ import annotations

import csv
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

OVERPASS_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)

# 车行常见 highway=*（弱化人行道等，路网更连贯）
HIGHWAY_REGEX = (
    "^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    "living_street|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"
)


@dataclass(frozen=True)
class Segment:
    lon1: float
    lat1: float
    lon2: float
    lat2: float
    highway: str = ""


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def build_overpass_query(south: float, west: float, north: float, east: float) -> str:
    return f"""[out:json][timeout:60];
(
  way["highway"~"{HIGHWAY_REGEX}"]({south},{west},{north},{east});
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
                "User-Agent": "DataStructure-course-demo/osm_graph",
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


def bbox_from_segments(segments: Sequence[Segment]) -> Tuple[float, float, float, float]:
    """south, west, north, east"""
    lats = [s.lat1 for s in segments] + [s.lat2 for s in segments]
    lons = [s.lon1 for s in segments] + [s.lon2 for s in segments]
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    return south, west, north, east


def load_segments_csv(path: str) -> List[Segment]:
    out: List[Segment] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                Segment(
                    float(row["lon1"]),
                    float(row["lat1"]),
                    float(row["lon2"]),
                    float(row["lat2"]),
                    row.get("highway", ""),
                )
            )
    return out


Node = Tuple[float, float]


def quantize(lon: float, lat: float, ndigits: int = 6) -> Node:
    return (round(lon, ndigits), round(lat, ndigits))


class RoadGraph:
    """无向正权图，邻接表（量化 (lon,lat) 为顶点）。"""

    def __init__(self, segments: Iterable[Segment]) -> None:
        self.adj: DefaultDict[Node, Dict[Node, float]] = defaultdict(dict)
        for s in segments:
            u = quantize(s.lon1, s.lat1)
            v = quantize(s.lon2, s.lat2)
            if u == v:
                continue
            w = haversine_m(s.lat1, s.lon1, s.lat2, s.lon2)
            self._put_min_edge(self.adj, u, v, w)
            self._put_min_edge(self.adj, v, u, w)

    @staticmethod
    def _put_min_edge(adj: DefaultDict[Node, Dict[Node, float]], u: Node, v: Node, w: float) -> None:
        prev = adj[u].get(v)
        if prev is None or w < prev:
            adj[u][v] = w

    def nodes(self) -> List[Node]:
        return list(self.adj.keys())

    def edge_list(self) -> List[Tuple[Node, Node]]:
        seen: Set[Tuple[Node, Node]] = set()
        out: List[Tuple[Node, Node]] = []
        for u, nbrs in self.adj.items():
            for v in nbrs:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in seen:
                    seen.add((a, b))
                    out.append((u, v))
        return out
