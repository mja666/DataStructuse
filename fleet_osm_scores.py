#!/usr/bin/env python3
"""
OSM 三档规模 × 四种策略：仅控制台批量跑分（较慢）。

可视化请运行: python fleet_osm.py（可加 --csv）
本脚本: python fleet_osm_scores.py（可加 --csv osm_sample_segments.csv）
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from fleet_osm import (
    build_scenario_triples_from_presets,
    preset_osm_map_presets,
    run_osm_console_score_batch,
)
from osm_graph import Segment, load_segments_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="OSM：三档×四策略跑分（stdout）")
    ap.add_argument("--csv", metavar="PATH", help="从 CSV 读路网；三档共用此几何")
    args = ap.parse_args()

    presets = preset_osm_map_presets()
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
