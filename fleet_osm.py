#!/usr/bin/env python3
"""
真实 OSM 路网 + 原车队模型（多车、电量、充电排队、任务动态到达、重量贪心装批 + 最近邻配送）。

- 边权为 Haversine 米；SimConfig.travel_speed 按 m/s、energy_per_distance 按 每米耗电 解释。
- 可视化：Tk，经纬度投影到画布（非格子），车辆沿最短路顶点线性插值移动。

可视化: python fleet_osm.py（可加 --csv）
批量跑分（仅控制台、较慢）: python fleet_osm_scores.py（见该文件说明，可加 --csv）
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
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type

from fleet_metaheuristic import (
    MetaHeuristicFleetSimulator,
    MetaHeuristicNearestFleetSimulator,
)
from fleet_nearest_first import FleetSimulatorNearestFirst
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


@dataclass(frozen=True)
class OSMMapPreset:
    """真实地图一档：经纬度 bbox + 与路网规模配套的仿真参数（无 XL_STRESS）。"""

    name: str
    south: float
    west: float
    north: float
    east: float
    cfg: SimConfig


def _osm_sim_config(
    name: str,
    seed: int,
    num_vehicles: int,
    num_chargers: int,
    sim_duration: float,
    task_spawn_rate: float,
    weight_lo: float,
    weight_hi: float,
    slack_lo: float,
    slack_hi: float,
) -> SimConfig:
    """米制路网共用物理系数；仅规模与到达率等随档位变化。"""
    return SimConfig(
        name=name,
        rows=1,
        cols=1,
        num_vehicles=num_vehicles,
        num_chargers=num_chargers,
        sim_duration=sim_duration,
        task_spawn_rate=task_spawn_rate,
        weight_range=(weight_lo, weight_hi),
        deadline_slack_range=(slack_lo, slack_hi),
        battery_capacity=400.0,
        load_capacity=220.0,
        energy_per_distance=0.2,
        travel_speed=10.0,
        charge_power=85.0,
        early_bonus_per_weight=9.0,
        late_penalty_per_time=14.0,
        distance_penalty_coef=0.008,
        obstacle_cover_ratio=0.0,
        seed=seed,
    )


def preset_osm_map_presets() -> List[OSMMapPreset]:
    """
    三档规模：同一城市片区（爱丁堡老城附近）由小到大 bbox，车辆/充电/时长/任务率递增。
    坐标可与课程 OSM 试验一致；若 Overpass 超时请缩小 LARGE 或换镜像。
    """
    return [
        OSMMapPreset(
            name="OSM_SMALL",
            south=55.9448,
            west=-3.1915,
            north=55.9478,
            east=-3.1865,
            cfg=_osm_sim_config(
                "OSM_SMALL",
                seed=21,
                num_vehicles=6,
                num_chargers=4,
                sim_duration=900.0,
                task_spawn_rate=0.26,
                weight_lo=6.0,
                weight_hi=42.0,
                slack_lo=200.0,
                slack_hi=400.0,
            ),
        ),
        OSMMapPreset(
            name="OSM_MEDIUM",
            south=55.9436,
            west=-3.1935,
            north=55.9490,
            east=-3.1845,
            cfg=_osm_sim_config(
                "OSM_MEDIUM",
                seed=22,
                num_vehicles=8,
                num_chargers=6,
                sim_duration=1200.0,
                task_spawn_rate=0.32,
                weight_lo=6.0,
                weight_hi=55.0,
                slack_lo=160.0,
                slack_hi=380.0,
            ),
        ),
        OSMMapPreset(
            name="OSM_LARGE",
            south=55.9413,
            west=-3.1970,
            north=55.9513,
            east=-3.1810,
            cfg=_osm_sim_config(
                "OSM_LARGE",
                seed=23,
                num_vehicles=10,
                num_chargers=8,
                sim_duration=1500.0,
                task_spawn_rate=0.38,
                weight_lo=8.0,
                weight_hi=70.0,
                slack_lo=130.0,
                slack_hi=340.0,
            ),
        ),
    ]


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


def populate_road_fleet_state(sim: FleetSimulator, cfg: SimConfig, prep: PreparedRoad) -> None:
    """在任意 FleetSimulator 子类实例上写入 OSM 路网与车队初态（不调用网格版 ``__init__``）。"""
    sim.cfg = cfg
    random.seed(cfg.seed)
    sim._rng = random.Random(cfg.seed)
    sim.n = prep.n
    sim.adj = prep.adj
    sim.depot = prep.depot
    sim.obstacles = set()
    sim.node_lonlat = prep.node_lonlat
    sim._non_obstacle_nodes = list(range(sim.n))
    d0, _ = dijkstra(sim.n, sim.adj, sim.depot)
    sim._task_candidate_nodes = [
        i
        for i in sim._non_obstacle_nodes
        if i != sim.depot and not math.isinf(d0[i])
    ]
    if len(sim._task_candidate_nodes) < 2:
        raise ValueError("可达任务点不足")
    sim._dist_row_cache = {}
    sim.tasks = {}
    sim._next_tid = 0
    sim.vehicles = []
    for i in range(cfg.num_vehicles):
        sim.vehicles.append(
            Vehicle(
                vid=i,
                node=sim.depot,
                battery=cfg.battery_capacity,
                load_used=0.0,
                busy_until=0.0,
            )
        )
    pool = [i for i in sim._non_obstacle_nodes if i != sim.depot]
    sim._rng.shuffle(pool)
    k = min(cfg.num_chargers, len(pool))
    chosen = pool[:k]
    sim.chargers = [
        ChargingStation(sid=i, node=node, slots=2) for i, node in enumerate(chosen)
    ]
    sim.score = 0.0


class FleetSimulatorRoad(FleetSimulator):
    """在 PreparedRoad 上复用 FleetSimulator 的调度、充电与评分逻辑。"""

    node_lonlat: List[Tuple[float, float]]

    def __init__(self, cfg: SimConfig, prep: PreparedRoad) -> None:
        populate_road_fleet_state(self, cfg, prep)


class FleetSimulatorNearestFirstRoad(FleetSimulatorNearestFirst):
    """最近任务装批 + OSM 路网。"""

    node_lonlat: List[Tuple[float, float]]

    def __init__(self, cfg: SimConfig, prep: PreparedRoad) -> None:
        populate_road_fleet_state(self, cfg, prep)


class MetaHeuristicFleetSimulatorRoad(MetaHeuristicFleetSimulator):
    """元启发（重量装批）+ OSM 路网。"""

    node_lonlat: List[Tuple[float, float]]

    def __init__(self, cfg: SimConfig, prep: PreparedRoad) -> None:
        populate_road_fleet_state(self, cfg, prep)
        self._meta_rng = random.Random(cfg.seed + 90_210)
        self.stranded_penalty = max(2000.0, 80.0 * cfg.late_penalty_per_time)
        self.stranded_events = 0


class MetaHeuristicNearestFleetSimulatorRoad(MetaHeuristicNearestFleetSimulator):
    """元启发（最近装批）+ OSM 路网。"""

    node_lonlat: List[Tuple[float, float]]

    def __init__(self, cfg: SimConfig, prep: PreparedRoad) -> None:
        populate_road_fleet_state(self, cfg, prep)
        self._meta_rng = random.Random(cfg.seed + 90_210)
        self.stranded_penalty = max(2000.0, 80.0 * cfg.late_penalty_per_time)
        self.stranded_events = 0


OSM_SIM_BUILDERS: Dict[str, Type[FleetSimulator]] = {
    "最大任务": FleetSimulatorRoad,
    "最近任务": FleetSimulatorNearestFirstRoad,
    "元启发·重量": MetaHeuristicFleetSimulatorRoad,
    "元启发·最近": MetaHeuristicNearestFleetSimulatorRoad,
}


def _print_osm_score_matrix(
    scenario_order: Sequence[str],
    strat_keys: Sequence[str],
    scores: Dict[str, Dict[str, float]],
) -> None:
    """控制台：每种规模 × 每种策略的最终 sim.score 对齐表。"""
    c0 = max(len("规模"), *(len(n) for n in scenario_order))
    widths = [max(len(k), 14) for k in strat_keys]
    sep = "  "
    head = "规模".ljust(c0)
    for k, w in zip(strat_keys, widths):
        head += sep + k.ljust(w)
    print(head)
    print("-" * len(head))
    for name in scenario_order:
        line = name.ljust(c0)
        row = scores.get(name, {})
        for k, w in zip(strat_keys, widths):
            v = row.get(k, float("nan"))
            line += sep + f"{v:.2f}".rjust(w)
        print(line)


def load_osm_segments_bbox(south: float, west: float, north: float, east: float) -> List[Segment]:
    data = fetch_overpass(build_overpass_query(south, west, north, east))
    return segments_from_osm_json(data)


def build_scenario_triples_from_presets(
    presets: Sequence[OSMMapPreset],
    segments_override: Optional[List[Segment]],
) -> List[Tuple[str, PreparedRoad, SimConfig]]:
    """
    联网：每档独立拉 bbox 并构图。
    CSV：三档共用同一 segments_override，仅 SimConfig 不同（地图几何相同、负载不同）。
    """
    rng0 = random.Random(0)
    out: List[Tuple[str, PreparedRoad, SimConfig]] = []
    for p in presets:
        if segments_override is not None:
            segs = segments_override
        else:
            segs = load_osm_segments_bbox(p.south, p.west, p.north, p.east)
        prep = prepare_road_network(segs, rng0)
        out.append((p.name, prep, p.cfg))
    return out


class FleetOSMVisualApp:
    def __init__(
        self,
        root: tk.Tk,
        scenarios: List[Tuple[str, PreparedRoad, SimConfig]],
    ) -> None:
        self.root = root
        self.scenarios = scenarios
        self._scenario_idx = 0
        self._builder_key = "最大任务"
        self.prep = scenarios[0][1]
        self.cfg = scenarios[0][2]
        self.sim = self._new_sim()
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

        self._bounds = self._compute_bounds(self.prep.node_lonlat)

        top = ttk.Frame(root, padding=4)
        top.pack(fill=tk.X)
        ttk.Label(top, text="规模").pack(side=tk.LEFT, padx=(0, 4))
        self.combo = ttk.Combobox(
            top,
            state="readonly",
            width=14,
            values=[s[0] for s in scenarios],
        )
        self.combo.current(0)
        self.combo.pack(side=tk.LEFT, padx=4)
        self.combo.bind("<<ComboboxSelected>>", self._on_scenario)

        ttk.Label(top, text="策略").pack(side=tk.LEFT, padx=(12, 4))
        self.strategy_combo = ttk.Combobox(
            top,
            state="readonly",
            width=14,
            values=list(OSM_SIM_BUILDERS.keys()),
        )
        _bk = list(OSM_SIM_BUILDERS.keys())
        self.strategy_combo.current(_bk.index(self._builder_key) if self._builder_key in _bk else 0)
        self.strategy_combo.pack(side=tk.LEFT, padx=4)
        self.strategy_combo.bind("<<ComboboxSelected>>", self._on_strategy)

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

    def _new_sim(self) -> FleetSimulator:
        cls = OSM_SIM_BUILDERS[self._builder_key]
        return cls(self.cfg, self.prep)

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

    def _on_scenario(self, _evt=None) -> None:
        idx = self.combo.current()
        if idx < 0:
            idx = 0
        self._scenario_idx = idx
        self._pause()
        self.prep = self.scenarios[idx][1]
        self.cfg = self.scenarios[idx][2]
        self._bounds = self._compute_bounds(self.prep.node_lonlat)
        self._restart()

    def _on_strategy(self, _evt=None) -> None:
        key = self.strategy_combo.get().strip()
        if key in OSM_SIM_BUILDERS:
            self._builder_key = key
        self._restart()

    def _play(self) -> None:
        self.running = True

    def _pause(self) -> None:
        self.running = False

    def _restart(self) -> None:
        self._pause()
        self.sim = self._new_sim()
        self.t = 0.0
        self._trails = defaultdict(deque)
        name = self.scenarios[self._scenario_idx][0]
        self.root.title(f"OSM 车队 · {name} · {self._builder_key}")

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


def run_osm_console_score_batch(scenarios: List[Tuple[str, PreparedRoad, SimConfig]]) -> None:
    """三档 × 四种策略各跑满时长，向 stdout 打印跑分行、summarize 与汇总矩阵。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    strat_keys = list(OSM_SIM_BUILDERS.keys())
    scenario_order = [n for n, _p, _c in scenarios]
    score_table: Dict[str, Dict[str, float]] = {n: {} for n in scenario_order}
    for name, prep, cfg in scenarios:
        for skey, cls in OSM_SIM_BUILDERS.items():
            sim = cls(cfg, prep)
            t = 0.0
            dt = 0.5
            while t <= cfg.sim_duration:
                sim.step(t, dt)
                t += dt
            score_table[name][skey] = sim.score
            print(f"跑分 | {name} | {skey} | {sim.score:.2f}")
            print(f"=== {name} | {skey} ===")
            print(summarize(sim))
            print()
    print("==== OSM 跑分汇总（每种规模 × 每种策略，列=最终 sim.score）====")
    _print_osm_score_matrix(scenario_order, strat_keys, score_table)
    print()
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="OSM 真实路网车队仿真 + 动态可视化（三档规模）")
    ap.add_argument("--csv", metavar="PATH", help="从 CSV 读路网（不联网）；三档共用此路网几何")
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

    root = tk.Tk()
    root.geometry("1200x860")
    FleetOSMVisualApp(root, scenarios)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
