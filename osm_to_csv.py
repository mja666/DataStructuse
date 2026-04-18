#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import xml.etree.ElementTree as ET
from typing import Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import contextily as ctx
except ImportError:
    ctx = None

try:
    from pyproj import Transformer
except ImportError:
    Transformer = None


def parse_osm_to_csv(osm_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    nodes_path = os.path.join(out_dir, "nodes.csv")
    ways_path = os.path.join(out_dir, "ways.csv")
    edges_path = os.path.join(out_dir, "edges.csv")

    # 先收集所有 node（id -> (lat, lon)）
    nodes = {}

    # 写 nodes.csv
    with open(nodes_path, "w", newline="", encoding="utf-8") as f_nodes:
        writer_nodes = csv.writer(f_nodes)
        writer_nodes.writerow(["id", "lat", "lon"])

        # iterparse 节省内存
        for event, elem in ET.iterparse(osm_path, events=("end",)):
            if elem.tag == "node":
                node_id = elem.attrib.get("id")
                lat = elem.attrib.get("lat")
                lon = elem.attrib.get("lon")
                if node_id and lat and lon:
                    nodes[node_id] = (lat, lon)
                    writer_nodes.writerow([node_id, lat, lon])
                elem.clear()

    # 第二遍解析 way，写 ways.csv 和 edges.csv
    with open(ways_path, "w", newline="", encoding="utf-8") as f_ways, \
         open(edges_path, "w", newline="", encoding="utf-8") as f_edges:

        writer_ways = csv.writer(f_ways)
        writer_edges = csv.writer(f_edges)

        writer_ways.writerow([
            "way_id", "highway", "name", "oneway", "maxspeed", "node_ids"
        ])
        writer_edges.writerow([
            "way_id", "u", "v", "highway", "name", "oneway", "maxspeed"
        ])

        for event, elem in ET.iterparse(osm_path, events=("end",)):
            if elem.tag == "way":
                way_id = elem.attrib.get("id", "")
                node_refs = []
                tags = {}

                for child in elem:
                    if child.tag == "nd":
                        ref = child.attrib.get("ref")
                        if ref:
                            node_refs.append(ref)
                    elif child.tag == "tag":
                        k = child.attrib.get("k")
                        v = child.attrib.get("v")
                        if k and v is not None:
                            tags[k] = v

                # 仅导出道路类 way（highway=*）
                highway = tags.get("highway", "")
                if highway:
                    name = tags.get("name", "")
                    oneway = tags.get("oneway", "")
                    maxspeed = tags.get("maxspeed", "")

                    writer_ways.writerow([
                        way_id, highway, name, oneway, maxspeed, ";".join(node_refs)
                    ])

                    # 拆成边：相邻 nd 构成 (u, v)
                    for i in range(len(node_refs) - 1):
                        u = node_refs[i]
                        v = node_refs[i + 1]
                        writer_edges.writerow([
                            way_id, u, v, highway, name, oneway, maxspeed
                        ])

                elem.clear()

    print("转换完成：")
    print(f"- {nodes_path}")
    print(f"- {ways_path}")
    print(f"- {edges_path}")


def render_csv_to_image(
    out_dir: str,
    image_name: str = "road_network.png",
    use_osm_basemap: bool = True
) -> None:
    """
    将 nodes.csv + edges.csv 渲染为道路网络图片（PNG）。
    """
    if plt is None:
        print("未安装 matplotlib，跳过图片生成。")
        print("可安装：pip install matplotlib")
        return

    nodes_path = os.path.join(out_dir, "nodes.csv")
    edges_path = os.path.join(out_dir, "edges.csv")
    image_path = os.path.join(out_dir, image_name)

    if not os.path.isfile(nodes_path) or not os.path.isfile(edges_path):
        print("未找到 nodes.csv 或 edges.csv，跳过图片生成。")
        return

    node_pos = {}
    with open(nodes_path, "r", newline="", encoding="utf-8") as f_nodes:
        reader = csv.DictReader(f_nodes)
        for row in reader:
            try:
                node_pos[row["id"]] = (float(row["lon"]), float(row["lat"]))
            except (KeyError, TypeError, ValueError):
                continue

    if not node_pos:
        print("nodes.csv 中没有可用坐标，跳过图片生成。")
        return

    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    line_count = 0
    use_mercator = bool(use_osm_basemap and ctx is not None and Transformer is not None)

    transformer = None
    if use_mercator:
        # OSM 在线瓦片底图使用 Web Mercator(EPSG:3857)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    with open(edges_path, "r", newline="", encoding="utf-8") as f_edges:
        reader = csv.DictReader(f_edges)
        for row in reader:
            u = row.get("u")
            v = row.get("v")
            if not u or not v:
                continue
            if u not in node_pos or v not in node_pos:
                continue
            x1, y1 = node_pos[u]
            x2, y2 = node_pos[v]
            if transformer is not None:
                x1, y1 = transformer.transform(x1, y1)
                x2, y2 = transformer.transform(x2, y2)
            ax.plot([x1, x2], [y1, y2], color="#1f77b4", linewidth=0.35, alpha=0.8)
            line_count += 1

    if line_count == 0:
        plt.close(fig)
        print("edges.csv 中没有可绘制边，跳过图片生成。")
        return

    if use_osm_basemap and ctx is not None and use_mercator:
        # 加载 OSM 底图；网络不可用时不影响主流程
        try:
            # 使用无文字底图（只保留颜色和地物底色）
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.PositronNoLabels,
                zoom="auto",
                attribution=False
            )
        except Exception as exc:
            print(f"OSM 底图加载失败，使用纯线图：{exc}")
    elif use_osm_basemap and (ctx is None or Transformer is None):
        print("未安装 contextily/pyproj，跳过 OSM 底图。")
        print("可安装：pip install contextily pyproj")

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(image_path)
    plt.close(fig)

    print(f"图片已生成：{image_path}")


def main(osm_file: Optional[str] = None, out_dir: str = "osm_csv_output") -> None:
    """
    在 PyCharm 中可直接运行此脚本：
    - 默认读取脚本同目录下的第一个 .osm 文件
    - 默认输出到当前目录下的 osm_csv_output

    你也可以在代码中直接传入自定义路径：
        main("beijing.osm", "output_dir")
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if osm_file is None:
        osm_candidates = sorted(
            f for f in os.listdir(script_dir) if f.lower().endswith(".osm")
        )
        if not osm_candidates:
            raise FileNotFoundError(
                f"在目录中未找到 .osm 文件: {script_dir}"
            )
        osm_file = os.path.join(script_dir, osm_candidates[0])
    elif not os.path.isabs(osm_file):
        osm_file = os.path.join(script_dir, osm_file)

    if not os.path.isabs(out_dir):
        out_dir = os.path.join(script_dir, out_dir)

    if not os.path.isfile(osm_file):
        raise FileNotFoundError(f"找不到输入文件: {osm_file}")

    parse_osm_to_csv(osm_file, out_dir)
    render_csv_to_image(out_dir)


if __name__ == "__main__":
    main()