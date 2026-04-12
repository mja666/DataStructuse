#!/usr/bin/env python3
"""
真实 OSM 路网 + 原车队模型（多车、电量、充电排队、任务动态到达、重量贪心装批 + 最近邻配送）。

- 边权为 Haversine 米；SimConfig.travel_speed 按 m/s、energy_per_distance 按 每米耗电 解释。
- 可视化：Tk，经纬度投影到画布（非格子），车辆沿最短路顶点线性插值移动。

运行: python fleet_osm.py
离线: python fleet_osm.py --csv osm_sample_segments.csv
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import tkinter as tk
from collections import defaultdict, deque
from dataclasses import dataclass
from tkinter import ttk
from typing import Dict, List, Optional, Sequence, Set, Tuple

from fleet_simulation import (
    ChargingStation,
    FleetSimulator,
    SimConfig,
    Task,
    TaskStatus,
    Vehicle,
    dijkstra,
    summarize,
)
from fleet_visual import _flatten_route_points, _vehicle_battery, _vehicle_xy
from osm_graph import (
    RoadGraph,
    Segment,
    build_overpass_query,
    fetch_overpass,
    haversine_m,
    load_segments_csv,
    segments_from_osm_json,
)


# 与 osm_fetch_demo 默认 bbox 一致的小片区域（可改）
BBOX_SOUTH, BBOX_WEST, BBOX_NORTH, BBOX_EAST = 55.9448, -3.1915, 55.9478, -3.1865


@dataclass(frozen=True)
class PreparedRoad:
    n: int
    adj: List[List[Tuple[int, float]]]
    depot: int
    node_lonlat: List[Tuple[float, float]]


def _largest_component(n: int, adj: List[List[Tuple[int, float]]]) -> List[int]:
    vis = [False] * n
    best: List[int] = []
    for s in range(n):
        if vis[s]:
            continue
        stack = [s]
        vis[s] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adj[u]:
                if not vis[v]:
                    vis[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def roadgraph_to_int_adj(rg: RoadGraph) -> Tuple[int, List[List[Tuple[int, float]]], List[Tuple[float, float]]]:
    keys = sorted(rg.adj.keys())
    idx: Dict[Tuple[float, float], int] = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    lonlat: List[Tuple[float, float]] = [(k[0], k[1]) for k in keys]
    best: Dict[int, Dict[int, float]] = defaultdict(dict)
    for u in keys:
        iu = idx[u]
        for v, w in rg.adj[u].items():
            iv = idx[v]
            if iu == iv:
                continue
            prev = best[iu].get(iv)
            if prev is None or w < prev:
                best[iu][iv] = w
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for iu, mp in best.items():
        for iv, w in mp.items():
            adj[iu].append((iv, w))
    return n, adj, lonlat


def prepare_road_network(segments: Sequence[Segment], rng: random.Random) -> PreparedRoad:
    rg = RoadGraph(segments)
    n_full, adj_full, ll_full = roadgraph_to_int_adj(rg)
    if n_full < 8:
        raise ValueError("路网顶点过少")

    comp = _largest_component(n_full, adj_full)
    if len(comp) < 8:
        raise ValueError("最大连通分量过小")

    old = sorted(comp)
    remap = {o: i for i, o in enumerate(old)}
    n = len(old)
    best: Dict[int, Dict[int, float]] = defaultdict(dict)
    for o in old:
        io = remap[o]
        for v, w in adj_full[o]:
            if v not in remap:
                continue
            iv = remap[v]
            if io == iv:
                continue
            prev = best[io].get(iv)
            if prev is None or w < prev:
                best[io][iv] = w
    adj = [[] for _ in range(n)]
    for io, mp in best.items():
        for iv, w in mp.items():
            adj[io].append((iv, w))

    lonlat = [ll_full[o] for o in old]

    cent_lon = sum(x for x, _ in lonlat) / n
    cent_lat = sum(y for _, y in lonlat) / n
    degs = sorted([(len(adj[i]), i) for i in range(n)], reverse=True)
    cut = max(1, n // 8)
    min_deg = max(2, degs[min(cut, len(degs) - 1)][0])
    cand = [i for d, i in degs if d >= min_deg]
    if not cand:
        cand = list(range(n))

    depot = min(
        cand,
        key=lambda i: (
            haversine_m(lonlat[i][1], lonlat[i][0], cent_lat, cent_lon),
            i,
        ),
    )
    _ = rng  # 预留：可按 seed 固定仓库规则
    return PreparedRoad(n=n, adj=adj, depot=depot, node_lonlat=lonlat)


def default_osm_sim_config() -> SimConfig:
    """米制路网：travel_speed≈m/s，energy_per_distance=每米耗电，时间与网格仿真同为「秒」。"""
    return SimConfig(
        name="OSM_FLEET",
        rows=1,
        cols=1,
        num_vehicles=4,
        num_chargers=3,
        sim_duration=900.0,
        task_spawn_rate=0.28,
        weight_range=(6.0, 45.0),
        deadline_slack_range=(200.0, 420.0),
        battery_capacity=400.0,
        load_capacity=220.0,
        energy_per_distance=0.2,
        travel_speed=10.0,
        charge_power=85.0,
        early_bonus_per_weight=9.0,
        late_penalty_per_time=14.0,
        distance_penalty_coef=0.008,
        obstacle_cover_ratio=0.0,
        seed=11,
    )


class FleetSimulatorRoad(FleetSimulator):
    """在 PreparedRoad 上复用 FleetSimulator 的调度、充电与评分逻辑。"""

    node_lonlat: List[Tuple[float, float]]

    def __init__(self, cfg: SimConfig, prep: PreparedRoad) -> None:
        self.cfg = cfg
        random.seed(cfg.seed)
        self._rng = random.Random(cfg.seed)
        self.n = prep.n
        self.adj = prep.adj
        self.depot = prep.depot
        self.obstacles = set()
        self.node_lonlat = prep.node_lonlat
        self._non_obstacle_nodes = list(range(self.n))
        d0, _ = dijkstra(self.n, self.adj, self.depot)
        self._task_candidate_nodes = [
            i
            for i in self._non_obstacle_nodes
            if i != self.depot and not math.isinf(d0[i])
        ]
        if len(self._task_candidate_nodes) < 2:
            raise ValueError("可达任务点不足")
        self._dist_row_cache: Dict[int, Tuple[List[float], List[int]]] = {}
        self.tasks: Dict[int, Task] = {}
        self._next_tid = 0
        self.vehicles: List[Vehicle] = []
        for i in range(cfg.num_vehicles):
            self.vehicles.append(
                Vehicle(
                    vid=i,
                    node=self.depot,
                    battery=cfg.battery_capacity,
                    load_used=0.0,
                    busy_until=0.0,
                )
            )
        pool = [i for i in self._non_obstacle_nodes if i != self.depot]
        self._rng.shuffle(pool)
        k = min(cfg.num_chargers, len(pool))
        chosen = pool[:k]
        self.chargers = [
            ChargingStation(sid=i, node=node, slots=2) for i, node in enumerate(chosen)
        ]
        self.score = 0.0


def load_osm_segments(use_network: bool) -> List[Segment]:
    if use_network:
        data = fetch_overpass(build_overpass_query(BBOX_SOUTH, BBOX_WEST, BBOX_NORTH, BBOX_EAST))
        return segments_from_osm_json(data)
    raise RuntimeError("use_network=False 时应走 CSV 分支")


class FleetOSMVisualApp:
    def __init__(self, root: tk.Tk, prep: PreparedRoad, cfg: SimConfig) -> None:
        self.root = root
        self.prep = prep
        self.cfg = cfg
        self.sim = FleetSimulatorRoad(cfg, prep)
        self.t = 0.0
        self.dt = 0.5
        self.running = False
        self.steps_per_tick = 1
        self._trails: Dict[int, deque] = defaultdict(deque)

        self._colors = {
            "bg": "#16161e",
            "edge": "#3b4261",
            "depot": "#e0af68",
            "charger": "#73daca",
            "task_pend": "#ff8a4c",
            "task_go": "#bb9af7",
            "route": "#565f89",
            "trail": "#3d4f6f",
            "bar_bg": "#24283b",
            "bar_time": "#3d59a1",
            "bar_pos": "#9ece6a",
            "bar_neg": "#f7768e",
        }
        self._vh = [
            "#7aa2f7",
            "#7dcfff",
            "#f7768e",
            "#e0af68",
            "#bb9af7",
            "#9ece6a",
            "#ff9e64",
            "#c0caf5",
        ]

        self._bounds = self._compute_bounds(prep.node_lonlat)

        top = ttk.Frame(root, padding=4)
        top.pack(fill=tk.X)
        ttk.Button(top, text="▶", width=3, command=self._play).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="⏸", width=3, command=self._pause).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="↻", width=3, command=self._restart).pack(side=tk.LEFT, padx=2)
        self.lbl = ttk.Label(top, text="OSM 车队仿真")
        self.lbl.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(
            root,
            width=1180,
            height=780,
            bg=self._colors["bg"],
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._restart()
        self._tick_loop()

    def _compute_bounds(
        self, lonlat: List[Tuple[float, float]], pad_ratio: float = 0.06
    ) -> Tuple[float, float, float, float]:
        lons = [x for x, _ in lonlat]
        lats = [y for _, y in lonlat]
        w, e = min(lons), max(lons)
        s, n = min(lats), max(lats)
        dx = (e - w) * pad_ratio + 1e-9
        dy = (n - s) * pad_ratio + 1e-9
        return w - dx, s - dy, e + dx, n + dy

    def _project(self, lon: float, lat: float, ox: float, oy: float, pw: float, ph: float) -> Tuple[float, float]:
        w0, s0, e0, n0 = self._bounds
        x = ox + (lon - w0) / (e0 - w0) * pw
        y = oy + (n0 - lat) / (n0 - s0) * ph
        return x, y

    def _cxy(self, node: int) -> Tuple[float, float]:
        lon, lat = self.sim.node_lonlat[node]
        W = int(self.canvas.cget("width"))
        H = int(self.canvas.cget("height"))
        bar_h = 52
        leg_h = 56
        pad = 12
        pw = W - 2 * pad
        ph = H - bar_h - leg_h - 2 * pad
        ox, oy = pad, pad
        return self._project(lon, lat, ox, oy, pw, ph)

    def _play(self) -> None:
        self.running = True

    def _pause(self) -> None:
        self.running = False

    def _restart(self) -> None:
        self._pause()
        self.sim = FleetSimulatorRoad(self.cfg, self.prep)
        self.t = 0.0
        self._trails = defaultdict(deque)

    def _tick_loop(self) -> None:
        cfg = self.cfg
        if self.running and self.t <= cfg.sim_duration:
            for _ in range(self.steps_per_tick):
                if self.t > cfg.sim_duration:
                    break
                self.sim.step(self.t, self.dt)
                self.t += self.dt
            if self.t > cfg.sim_duration:
                self.running = False
        self.lbl.config(text=f"t={self.t:.1f}/{cfg.sim_duration:.0f}  得分 {self.sim.score:.1f}")
        self._draw()
        self.root.after(45, self._tick_loop)

    def _erase_trail_if_idle(self, v: Vehicle) -> None:
        if (
            v.current_task is None
            and not v.carry_batch
            and v.busy_until <= self.t + 1e-9
            and not v.visual_segments
        ):
            self._trails[v.vid].clear()

    def _draw(self) -> None:
        self.canvas.delete("all")
        sim = self.sim
        cfg = self.cfg
        W = int(self.canvas.cget("width"))
        H = int(self.canvas.cget("height"))
        bar_h = 52
        leg_h = 56
        pad = 12
        pw = W - 2 * pad
        ph = H - bar_h - leg_h - 2 * pad
        ox, oy = pad, pad

        w0, s0, e0, n0 = self._bounds

        seen: Set[Tuple[int, int]] = set()
        for u in range(sim.n):
            lon_u, lat_u = sim.node_lonlat[u]
            x0, y0 = self._project(lon_u, lat_u, ox, oy, pw, ph)
            for v, _w in sim.adj[u]:
                if u < v:
                    a, b = u, v
                else:
                    a, b = v, u
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                lon_v, lat_v = sim.node_lonlat[v]
                x1, y1 = self._project(lon_v, lat_v, ox, oy, pw, ph)
                self.canvas.create_line(x0, y0, x1, y1, fill=self._colors["edge"], width=1)

        cx_d, cy_d = self._cxy(sim.depot)
        self.canvas.create_rectangle(
            cx_d - 9,
            cy_d - 9,
            cx_d + 9,
            cy_d + 9,
            fill=self._colors["depot"],
            outline="#1a1b26",
            width=2,
        )

        for csn in sim.chargers:
            cx, cy = self._cxy(csn.node)
            r = 7.0
            self.canvas.create_polygon(
                cx,
                cy - r,
                cx + r,
                cy,
                cx,
                cy + r,
                cx - r,
                cy,
                fill=self._colors["charger"],
                outline="#1a1f2e",
            )

        pending_cnt: Dict[int, int] = defaultdict(int)
        for task in sim.tasks.values():
            if task.status == TaskStatus.PENDING and task.spawn_time <= self.t:
                pending_cnt[task.node] += 1
        for node, k in pending_cnt.items():
            px, py = self._cxy(node)
            for i in range(min(k, 5)):
                ang = (i / max(k, 1)) * 2 * math.pi
                rr = min(3.5 + k * 0.35, 7.5)
                self.canvas.create_oval(
                    px + math.cos(ang) * 5 - rr / 2,
                    py + math.sin(ang) * 5 - rr / 2,
                    px + math.cos(ang) * 5 + rr / 2,
                    py + math.sin(ang) * 5 + rr / 2,
                    fill=self._colors["task_pend"],
                    outline="",
                )

        for task in sim.tasks.values():
            if task.status == TaskStatus.ASSIGNED:
                px, py = self._cxy(task.node)
                self.canvas.create_oval(px - 6, py - 6, px + 6, py + 6, outline=self._colors["task_go"], width=2)

        for v in sim.vehicles:
            self._erase_trail_if_idle(v)

        for v in sim.vehicles:
            if v.visual_segments and self.t <= v.busy_until + 1e-9:
                pts = _flatten_route_points(v.visual_segments, self._cxy)
                if len(pts) >= 2:
                    flat: List[float] = []
                    for p in pts:
                        flat.extend(p)
                    self.canvas.create_line(
                        *flat,
                        fill=self._colors["route"],
                        width=2,
                        dash=(5, 4),
                    )

        for v in sim.vehicles:
            tr = self._trails[v.vid]
            px, py = _vehicle_xy(sim, v, self.t, self._cxy)
            if not tr or math.hypot(tr[-1][0] - px, tr[-1][1] - py) > 0.8:
                tr.append((px, py))
            if len(tr) >= 2:
                flat = []
                for p in tr:
                    flat.extend(p)
                self.canvas.create_line(
                    *flat,
                    fill=self._colors["trail"],
                    width=2,
                    smooth=True,
                    splinesteps=8,
                )

        for v in sim.vehicles:
            vx, vy = _vehicle_xy(sim, v, self.t, self._cxy)
            oxv = 7 * math.cos(v.vid * 2.1)
            oyv = 7 * math.sin(v.vid * 2.1)
            vx += oxv
            vy += oyv
            col = self._vh[v.vid % len(self._vh)]
            self.canvas.create_oval(vx - 7, vy - 7, vx + 7, vy + 7, fill=col, outline="#1a1b26", width=2)
            cap = cfg.battery_capacity
            cur_bat = _vehicle_battery(v, self.t)
            ratio = max(0.0, min(1.0, cur_bat / cap))
            bw, bh = 28.0, 4.0
            self.canvas.create_rectangle(vx - bw / 2, vy + 9, vx + bw / 2, vy + 9 + bh, fill="#24283b", outline="")
            hue = "#a6e3a1" if ratio > 0.35 else "#f9e2af" if ratio > 0.15 else "#f7768e"
            self.canvas.create_rectangle(
                vx - bw / 2,
                vy + 9,
                vx - bw / 2 + bw * ratio,
                vy + 9 + bh,
                fill=hue,
                outline="",
            )

        ly = H - leg_h + 4
        self.canvas.create_rectangle(0, ly, W, H - 2, fill="#1a1b26", outline="#292e42")
        self.canvas.create_text(
            14,
            ly + 18,
            text="■ 仓库  ◆ 充电  · 待接  ○ 配送中  — 规划  灰线 路网",
            fill="#a9b1d6",
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
        )

        bx0, bx1 = pad, W - pad
        by0 = H - bar_h + 6
        by1 = H - 6
        self.canvas.create_rectangle(bx0, by0, bx1, by1, fill=self._colors["bar_bg"], outline="")
        prog = 0.0 if cfg.sim_duration <= 0 else min(1.0, self.t / cfg.sim_duration)
        self.canvas.create_rectangle(bx0, by0, bx0 + (bx1 - bx0) * prog, by1, fill=self._colors["bar_time"], outline="")
        self.canvas.create_text((bx0 + bx1) / 2, (by0 + by1) / 2, text="仿真进度", fill="#565f89", font=("Microsoft YaHei UI", 8))

        sc = max(-8000.0, min(8000.0, sim.score))
        tmid = (bx0 + bx1) / 2
        gw = bx1 - bx0
        smax = 4000.0
        sx = tmid + (sc / smax) * (gw / 2 - 16)
        sx = max(bx0 + 6, min(bx1 - 6, sx))
        self.canvas.create_line(tmid, by0 - 2, tmid, by1 + 2, fill="#414868")
        col = self._colors["bar_pos"] if sc >= 0 else self._colors["bar_neg"]
        self.canvas.create_polygon(sx, by0 - 4, sx - 4, by0 - 11, sx + 4, by0 - 11, fill=col, outline="")


def main() -> int:
    ap = argparse.ArgumentParser(description="OSM 真实路网车队仿真 + 动态可视化")
    ap.add_argument("--csv", metavar="PATH", help="从 CSV 读路网（不联网）")
    ap.add_argument("--no-gui", action="store_true", help="只建图并跑完仿真打印摘要，不打开窗口")
    args = ap.parse_args()

    try:
        if args.csv:
            segs = load_segments_csv(args.csv)
        else:
            segs = load_osm_segments(True)
    except Exception as e:
        print("加载路网失败:", e, file=sys.stderr)
        return 1

    rng = random.Random(0)
    try:
        prep = prepare_road_network(segs, rng)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    cfg = default_osm_sim_config()
    print(
        f"路网顶点 n={prep.n}，仓库节点={prep.depot}，"
        f"车辆={cfg.num_vehicles}，充电站={cfg.num_chargers}"
    )

    if args.no_gui:
        sim = FleetSimulatorRoad(cfg, prep)
        t = 0.0
        dt = 0.5
        while t <= cfg.sim_duration:
            sim.step(t, dt)
            t += dt
        print(summarize(sim))
        return 0

    root = tk.Tk()
    root.title("OSM 车队 · 真实路网")
    root.geometry("1200x860")
    FleetOSMVisualApp(root, prep, cfg)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
