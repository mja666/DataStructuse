#!/usr/bin/env python3
"""
OSM 三档规模 × 四种策略：仅控制台批量跑分（较慢）。

可视化请运行: python fleet_osm.py（可加 --csv）
本脚本: python fleet_osm_scores.py（可加 --csv osm_sample_segments.csv）

可复现性：同一 CSV 文件、不加 --seed 时三档内置 seed 为 21/22/23，边速与任务随机流固定，
多次运行跑分一致。联网拉 Overpass 时几何可能随远端数据变化，跑分可能不同。
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from fleet_osm import (
    build_scenario_triples_from_presets,
    osm_presets_for_run,
    run_osm_console_score_batch,
)
from osm_graph import Segment, load_segments_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="OSM：三档×四策略跑分（stdout）")
    ap.add_argument("--csv", metavar="PATH", help="从 CSV 读路网；三档共用此几何")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="三档 SimConfig.seed 设为 N、N+1、N+2；省略则 21/22/23",
    )
    args = ap.parse_args()

    presets = osm_presets_for_run(args.seed)
    segs_override: Optional[List[Segment]] = None
    if args.csv:
        try:
            segs_override = load_segments_csv(args.csv)
        except Exception as e:
            print("读取 CSV 失败:", e, file=sys.stderr)
            return 1

    try:
        scenarios = build_scenario_triples_from_presets(presets, segs_override)
    except Exception as e:
        print("构建路网/场景失败:", e, file=sys.stderr)
        return 1

    run_osm_console_score_batch(scenarios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
