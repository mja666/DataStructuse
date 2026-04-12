"""
新能源物流车队协同调度 — 图结构道路 + 最短路径 + 最大任务（重量）优先策略

大作业要求要点（本实现覆盖）：
- 图表示道路，Dijkstra 最短路（按源点缓存距离行）
- 车队规模、电量上限、载重上限；任务动态到达（时间、节点、重量随机）
- 评分：越早完成、路径越短越高；超时扣分
- 电量不足时前往充电站；充电站排队与并发槽位（负荷）
- 至少三种不同规模场景（SMALL / MEDIUM / LARGE）

策略：仓库按载重贪心装多票（重量大优先装车），最近邻排配送顺序；同一车次送完再回仓。单任务函数 pick_task_max_weight 保留便于改策略对比。
"""

from __future__ import annotations

import csv
import heapq
import math
import os
import random
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------- 图与最短路 ----------------------------


def dijkstra(
    n: int, adj: List[List[Tuple[int, float]]], src: int
) -> Tuple[List[float], List[int]]:
    """单源最短路；parent[v] 为树边父节点，src 处为 -1。"""
    dist = [math.inf] * n
    parent = [-1] * n
    dist[src] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


def path_from_parent(parent: List[int], src: int, dst: int) -> List[int]:
    if dst < 0 or src < 0 or dst >= len(parent):
        return []
    if parent[dst] == -1 and dst != src:
        return []
    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def build_grid_graph(
    rows: int,
    cols: int,
    blocked_nodes: Optional[Set[int]] = None,
) -> Tuple[int, List[List[Tuple[int, float]]]]:
    """行主序编号，四邻接，边权为欧氏距离；障碍节点无边。"""
    n = rows * cols
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    blocked = blocked_nodes or set()

    def coord(k: int) -> Tuple[int, int]:
        return k // cols, k % cols

    for u in range(n):
        if u in blocked:
            continue
        r, c = coord(u)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = nr * cols + nc
                if v in blocked:
                    continue
                w = math.hypot(dr, dc)
                adj[u].append((v, w))
    return n, adj


# ---------------------------- 领域模型 ----------------------------


class TaskStatus(Enum):
    PENDING = auto()
    ASSIGNED = auto()
    DONE = auto()
    EXPIRED = auto()


@dataclass
class Task:
    tid: int
    spawn_time: float
    node: int
    weight: float
    deadline: float
    status: TaskStatus = TaskStatus.PENDING
    assigned_vehicle: Optional[int] = None
    finish_time: Optional[float] = None
    travel_distance: float = 0.0


@dataclass
class Vehicle:
    vid: int
    node: int
    battery: float
    load_used: float
    busy_until: float = 0.0
    current_task: Optional[int] = None
    # 在仓库一次装车、按序连续配送的订单 id；current_task 与 carry_batch[batch_index] 一致
    carry_batch: List[int] = field(default_factory=list)
    batch_index: int = 0
    # (段开始时刻, 段结束时刻, 路径顶点序列)；单点或两点相同表示在节点等待（如充电）
    visual_segments: List[Tuple[float, float, List[int]]] = field(default_factory=list)
    # (段开始, 段结束, 起始电量, 结束电量)，用于可视化平滑显示电量变化
    battery_segments: List[Tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class ChargingSession:
    vehicle_id: int
    start: float
    until: float


@dataclass
class ChargingStation:
    sid: int
    node: int
    slots: int
    active: List[ChargingSession] = field(default_factory=list)
    total_served: int = 0
    peak_active: int = 0


@dataclass
class SimConfig:
    name: str
    rows: int
    cols: int
    num_vehicles: int
    num_chargers: int
    sim_duration: float
    task_spawn_rate: float
    weight_range: Tuple[float, float]
    deadline_slack_range: Tuple[float, float]
    battery_capacity: float
    load_capacity: float
    energy_per_distance: float
    travel_speed: float
    charge_power: float
    early_bonus_per_weight: float
    late_penalty_per_time: float
    distance_penalty_coef: float
    obstacle_cover_ratio: float = 0.15
    seed: int = 42


# ---------------------------- 策略：最大任务优先 ----------------------------


def pick_task_max_weight(
    pending: Iterable[Task],
    vehicle: Vehicle,
    now: float,
    load_cap: float,
) -> Optional[Task]:
    """在待分配任务中选重量最大且未过期、且车辆还能装下的任务。"""
    best: Optional[Task] = None
    for t in pending:
        if t.status != TaskStatus.PENDING:
            continue
        if t.spawn_time > now:
            continue
        if now > t.deadline:
            continue
        if t.weight + vehicle.load_used > load_cap + 1e-9:
            continue
        if best is None or t.weight > best.weight:
            best = t
    return best


def _eligible_pending(pending: Iterable[Task], now: float) -> List[Task]:
    out: List[Task] = []
    for t in pending:
        if t.status != TaskStatus.PENDING:
            continue
        if t.spawn_time > now:
            continue
        if now > t.deadline:
            continue
        out.append(t)
    return out


def pick_batch_greedy_max_weight(
    pending: Iterable[Task],
    now: float,
    load_cap: float,
    load_already: float = 0.0,
) -> List[Task]:
    """在载重上限内按重量从大到小贪心装入多个任务（假定均在仓库装货）。"""
    cand = sorted(
        _eligible_pending(pending, now),
        key=lambda x: -x.weight,
    )
    batch: List[Task] = []
    wsum = load_already
    for t in cand:
        if wsum + t.weight <= load_cap + 1e-9:
            batch.append(t)
            wsum += t.weight
    return batch


# ---------------------------- 仿真核心 ----------------------------


class FleetSimulator:
    @staticmethod
    def _random_depot_in_center(rows: int, cols: int, rng: random.Random) -> int:
        """在「中间约三分之一 × 三分之一」的矩形区域内随机选一格作为仓库。"""
        r_lo = rows // 3
        r_hi = (2 * rows) // 3
        c_lo = cols // 3
        c_hi = (2 * cols) // 3
        if r_lo >= r_hi:
            r_lo, r_hi = 0, max(0, rows - 1)
        else:
            r_hi = min(r_hi, rows - 1)
        if c_lo >= c_hi:
            c_lo, c_hi = 0, max(0, cols - 1)
        else:
            c_hi = min(c_hi, cols - 1)
        r = rng.randint(r_lo, r_hi)
        c = rng.randint(c_lo, c_hi)
        return r * cols + c

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        self._rng = random.Random(cfg.seed)
        self.depot = self._random_depot_in_center(cfg.rows, cfg.cols, self._rng)
        self.obstacles = self._generate_obstacles()
        self.n, self.adj = build_grid_graph(cfg.rows, cfg.cols, self.obstacles)
        self._non_obstacle_nodes = [i for i in range(self.n) if i not in self.obstacles]
        self._task_candidate_nodes = [i for i in self._non_obstacle_nodes if i != self.depot]
        self._dist_row_cache: Dict[int, Tuple[List[float], List[int]]] = {}

        self.tasks: Dict[int, Task] = {}
        self._pending_tids: Set[int] = set()
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

        self.chargers = self._place_chargers(cfg.num_chargers)
        self.score = 0.0
        # 真实 OSM 路网：每条边独立限速；网格仿真保持 None
        self.edge_speed_mps: Optional[Dict[Tuple[int, int], float]] = None

    def _generate_obstacles(self) -> Set[int]:
        """
        随机矩形障碍：2x2 / 2x3 / 1x4（含旋转）；仓库格必定不在障碍中。
        """
        rows, cols = self.cfg.rows, self.cfg.cols
        n = rows * cols
        target = int(n * max(0.0, min(0.35, self.cfg.obstacle_cover_ratio)))
        if target <= 0:
            return set()

        obstacles: Set[int] = set()
        shapes = [(2, 2), (2, 3), (3, 2), (1, 4), (4, 1)]
        protect = {self.depot}
        attempts = 0
        max_attempts = max(200, n * 6)

        while len(obstacles) < target and attempts < max_attempts:
            attempts += 1
            h, w = self._rng.choice(shapes)
            if h > rows or w > cols:
                continue
            r0 = self._rng.randint(0, rows - h)
            c0 = self._rng.randint(0, cols - w)

            block = set()
            for rr in range(r0, r0 + h):
                for cc in range(c0, c0 + w):
                    block.add(rr * cols + cc)

            if block & protect:
                continue
            obstacles |= block

        return obstacles

    def _ensure_dijkstra(self, src: int) -> Tuple[List[float], List[int]]:
        if src not in self._dist_row_cache:
            self._dist_row_cache[src] = dijkstra(self.n, self.adj, src)
        return self._dist_row_cache[src]

    def dist_uv(self, u: int, v: int) -> float:
        d, _ = self._ensure_dijkstra(u)
        return d[v]

    def _place_chargers(self, k: int) -> List[ChargingStation]:
        """在图中均匀撒点充电站（避开仓库节点）。"""
        nodes = [i for i in self._non_obstacle_nodes if i != self.depot]
        self._rng.shuffle(nodes)
        chosen = nodes[:k] if k <= len(nodes) else nodes
        return [
            ChargingStation(sid=i, node=node, slots=2)
            for i, node in enumerate(chosen)
        ]

    def _spawn_task(self, t: float) -> None:
        slack_lo, slack_hi = self.cfg.deadline_slack_range
        w_lo, w_hi = self.cfg.weight_range
        if not self._task_candidate_nodes:
            return
        node = self._rng.choice(self._task_candidate_nodes)
        weight = self._rng.uniform(w_lo, w_hi)
        deadline = t + self._rng.uniform(slack_lo, slack_hi)
        task = Task(
            tid=self._next_tid,
            spawn_time=t,
            node=node,
            weight=weight,
            deadline=deadline,
        )
        self.tasks[task.tid] = task
        self._pending_tids.add(task.tid)
        self._next_tid += 1

    def _travel_time(self, distance: float) -> float:
        return distance / self.cfg.travel_speed

    def _edge_len(self, u: int, v: int) -> float:
        for nb, w in self.adj[u]:
            if nb == v:
                return w
        for nb, w in self.adj[v]:
            if nb == u:
                return w
        return math.inf

    def _speed_on_edge(self, u: int, v: int) -> float:
        m = getattr(self, "edge_speed_mps", None)
        if not m:
            return self.cfg.travel_speed
        a, b = (u, v) if u < v else (v, u)
        return m.get((a, b), self.cfg.travel_speed)

    def _travel_time_for_path(self, path: List[int]) -> float:
        if len(path) < 2:
            return 0.0
        if getattr(self, "edge_speed_mps", None) is None:
            tot = 0.0
            for i in range(len(path) - 1):
                tot += self._edge_len(path[i], path[i + 1])
            return tot / max(self.cfg.travel_speed, 1e-9)
        t = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            d = self._edge_len(a, b)
            t += d / max(self._speed_on_edge(a, b), 1e-9)
        return t

    def _edge_times_along_path(self, path: List[int]) -> List[float]:
        if len(path) < 2:
            return []
        out: List[float] = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            d = self._edge_len(a, b)
            out.append(d / max(self._speed_on_edge(a, b), 1e-9))
        return out

    def _energy_need(self, distance: float) -> float:
        return distance * self.cfg.energy_per_distance

    def _station_on_node(self, node: int) -> Optional[ChargingStation]:
        for cs in self.chargers:
            if cs.node == node:
                return cs
        return None

    def _charging_sessions_covering(self, cs: ChargingStation, t: float) -> List[ChargingSession]:
        return [s for s in cs.active if s.start - 1e-9 <= t < s.until]

    def _next_charge_start(self, cs: ChargingStation, t_arrive: float) -> float:
        """在 t_arrive 及之后找到第一个“有空闲槽位”的时刻（离散推进，用于排队）。"""
        t = t_arrive
        for _ in range(500):
            busy = self._charging_sessions_covering(cs, t)
            if len(busy) < cs.slots:
                return t
            t = min(s.until for s in busy)
        return t_arrive

    def _reserve_charge(
        self, cs: ChargingStation, vid: int, t_arrive: float, battery_before: float
    ) -> Tuple[float, float]:
        """预约充电：从到达充电站时刻起排队，返回 (charge_start, charge_end)。"""
        missing = max(0.0, self.cfg.battery_capacity - battery_before)
        if missing <= 1e-9:
            return (t_arrive, t_arrive)
        duration = missing / self.cfg.charge_power
        start = self._next_charge_start(cs, t_arrive)
        end = start + duration
        cs.active.append(ChargingSession(vehicle_id=vid, start=start, until=end))
        cs.peak_active = max(
            cs.peak_active,
            len(self._charging_sessions_covering(cs, start + 1e-9)),
        )
        cs.total_served += 1
        return (start, end)

    def _cancel_last_session(self, cs: ChargingStation) -> None:
        if cs.active:
            cs.active.pop()
        cs.total_served = max(0, cs.total_served - 1)

    def _tick_chargers(self, now: float) -> None:
        for cs in self.chargers:
            cs.active = [s for s in cs.active if s.until > now]

    def _path_nodes(self, u: int, v: int) -> List[int]:
        _, parent = self._ensure_dijkstra(u)
        return path_from_parent(parent, u, v)

    def _order_tasks_nn(self, start_node: int, tasks: List[Task]) -> List[Task]:
        """从 start_node 出发，最近邻次序排列配送点。"""
        remaining = list(tasks)
        ordered: List[Task] = []
        cur = start_node
        while remaining:
            best_t = min(remaining, key=lambda t: (self.dist_uv(cur, t.node), t.tid))
            ordered.append(best_t)
            remaining.remove(best_t)
            cur = best_t.node
        return ordered

    def _tour_distance_with_return(self, stop_nodes: List[int]) -> float:
        """仓库 → 各配送点（按 stop_nodes 顺序）→ 回仓库 的总路程。"""
        if not stop_nodes:
            return 0.0
        seq = [self.depot] + list(stop_nodes) + [self.depot]
        s = 0.0
        for i in range(len(seq) - 1):
            d = self.dist_uv(seq[i], seq[i + 1])
            if math.isinf(d):
                return math.inf
            s += d
        return s

    def _begin_leg_from_to(
        self,
        v: Vehicle,
        now: float,
        from_node: int,
        to_node: int,
        record_tid: Optional[int],
    ) -> bool:
        """执行一段 from_node→to_node 的行驶（可含充电）。成功则更新 busy_until、电量、位置、可视化。"""
        d_direct = self.dist_uv(from_node, to_node)
        if math.isinf(d_direct):
            return False

        need_direct = self._energy_need(d_direct)
        if need_direct <= v.battery + 1e-9:
            p = self._path_nodes(from_node, to_node)
            travel_t = self._travel_time_for_path(p)
            b0 = v.battery
            b1 = b0 - need_direct
            v.visual_segments = [(now, now + travel_t, p)] if p else []
            v.battery_segments = [(now, now + travel_t, b0, b1)]
            v.battery = b1
            v.node = to_node
            v.busy_until = now + travel_t
            if record_tid is not None:
                self.tasks[record_tid].travel_distance = d_direct
            return True

        best = self._best_charger_plan(from_node, to_node, now, v.battery)
        if best is None:
            return False
        station, cnode, d1, d2 = best
        e1 = self._energy_need(d1)

        p1 = self._path_nodes(from_node, cnode)
        p2 = self._path_nodes(cnode, to_node)
        t_arrive_c = now + self._travel_time_for_path(p1)
        b0 = v.battery
        bat_after_leg1 = b0 - e1
        charge_start, charge_end = self._reserve_charge(
            station, v.vid, t_arrive_c, bat_after_leg1
        )
        bat_after = self.cfg.battery_capacity
        e2 = self._energy_need(d2)
        if e2 > bat_after + 1e-9:
            self._cancel_last_session(station)
            return False

        travel2 = self._travel_time_for_path(p2)
        arrive_task = charge_end + travel2
        v.visual_segments = []
        if p1:
            v.visual_segments.append((now, t_arrive_c, p1))
        v.visual_segments.append((t_arrive_c, charge_end, [cnode, cnode]))
        if p2:
            v.visual_segments.append((charge_end, arrive_task, p2))
        b_leg1 = bat_after_leg1
        b_leg2_start = bat_after
        b_final = bat_after - e2
        bsegs: List[Tuple[float, float, float, float]] = [(now, t_arrive_c, b0, b_leg1)]
        if charge_start > t_arrive_c + 1e-9:
            bsegs.append((t_arrive_c, charge_start, b_leg1, b_leg1))
        if charge_end > charge_start + 1e-9:
            bsegs.append((charge_start, charge_end, b_leg1, b_leg2_start))
        bsegs.append((charge_end, arrive_task, b_leg2_start, b_final))
        v.battery_segments = bsegs
        v.battery = b_final
        v.node = to_node
        v.busy_until = arrive_task
        if record_tid is not None:
            self.tasks[record_tid].travel_distance = d1 + d2
        return True

    def _best_charger_plan(
        self,
        from_node: int,
        to_node: int,
        now: float,
        battery_now: float,
    ) -> Optional[Tuple[ChargingStation, int, float, float]]:
        """
        选择使“到达目的地时刻”最小的充电站，代价包含：
        行驶到站 + 排队等待 + 充电时间 + 充后行驶到目标。
        """
        best: Optional[Tuple[ChargingStation, int, float, float]] = None
        best_arrival = math.inf

        for cs in self.chargers:
            cnode = cs.node
            d1 = self.dist_uv(from_node, cnode)
            d2 = self.dist_uv(cnode, to_node)
            if math.isinf(d1) or math.isinf(d2):
                continue

            e1 = self._energy_need(d1)
            if e1 > battery_now + 1e-9:
                continue

            p1 = self._path_nodes(from_node, cnode)
            p2 = self._path_nodes(cnode, to_node)
            t_arrive_c = now + self._travel_time_for_path(p1)
            battery_after_leg1 = battery_now - e1
            missing = max(0.0, self.cfg.battery_capacity - battery_after_leg1)
            charge_duration = missing / self.cfg.charge_power
            charge_start = self._next_charge_start(cs, t_arrive_c)
            charge_end = charge_start + charge_duration

            e2 = self._energy_need(d2)
            if e2 > self.cfg.battery_capacity + 1e-9:
                continue

            arrival = charge_end + self._travel_time_for_path(p2)
            if arrival < best_arrival:
                best_arrival = arrival
                best = (cs, cnode, d1, d2)

        return best

    def _select_charger_for_depot_topup(self, now: float) -> Optional[ChargingStation]:
        """从仓库出发：兼顾路程与当前排队，选更合适的充电站。"""
        if not self.chargers:
            return None

        def score(cs: ChargingStation) -> Tuple[float, int]:
            d = self.dist_uv(self.depot, cs.node)
            q = len(self._charging_sessions_covering(cs, now))
            return (d + 30.0 * q, cs.sid)

        return min(self.chargers, key=score)

    def _try_proactive_depot_charge(self, v: Vehicle, now: float) -> bool:
        """
        在仓库、无车次时：电量偏低则执行「仓库→充电站（充满）→仓库」，
        为后续派单留余量，减少途中/回仓断电。
        """
        if v.node != self.depot or v.carry_batch or v.current_task is not None:
            return False
        cfg = self.cfg
        cap = cfg.battery_capacity
        if v.battery >= cap * 0.46:
            return False
        station = self._select_charger_for_depot_topup(now)
        if station is None:
            return False
        cnode = station.node
        d1 = self.dist_uv(self.depot, cnode)
        d2 = self.dist_uv(cnode, self.depot)
        if math.isinf(d1) or math.isinf(d2):
            return False
        e1 = self._energy_need(d1)
        if e1 > v.battery + 1e-9:
            return False
        e2 = self._energy_need(d2)
        if e2 > cap + 1e-9:
            return False
        p1 = self._path_nodes(self.depot, cnode)
        t_arrive_c = now + self._travel_time_for_path(p1)
        b0 = v.battery
        bat_after_leg1 = b0 - e1
        charge_start, charge_end = self._reserve_charge(
            station, v.vid, t_arrive_c, bat_after_leg1
        )
        bat_after = cap
        if e2 > bat_after + 1e-9:
            self._cancel_last_session(station)
            return False
        p2 = self._path_nodes(cnode, self.depot)
        t_end = charge_end + self._travel_time_for_path(p2)
        v.visual_segments = []
        if p1:
            v.visual_segments.append((now, t_arrive_c, p1))
        v.visual_segments.append((t_arrive_c, charge_end, [cnode, cnode]))
        if p2:
            v.visual_segments.append((charge_end, t_end, p2))
        b_leg1 = bat_after_leg1
        b_leg2_start = bat_after
        b_final = bat_after - e2
        bsegs: List[Tuple[float, float, float, float]] = [(now, t_arrive_c, b0, b_leg1)]
        if charge_start > t_arrive_c + 1e-9:
            bsegs.append((t_arrive_c, charge_start, b_leg1, b_leg1))
        if charge_end > charge_start + 1e-9:
            bsegs.append((charge_start, charge_end, b_leg1, b_leg2_start))
        bsegs.append((charge_end, t_end, b_leg2_start, b_final))
        v.battery_segments = bsegs
        v.battery = b_final
        v.node = self.depot
        v.busy_until = t_end
        return True

    def _rollback_batch(self, v: Vehicle, tids: List[int]) -> None:
        for tid in tids:
            self.tasks[tid].status = TaskStatus.PENDING
            self.tasks[tid].assigned_vehicle = None
            self._pending_tids.add(tid)
        v.carry_batch = []
        v.batch_index = 0
        v.current_task = None
        v.load_used = 0.0
        v.visual_segments = []
        v.battery_segments = []

    def _assign_vehicle(self, v: Vehicle, now: float) -> None:
        if v.node != self.depot:
            return
        if v.carry_batch:
            return
        if self._try_proactive_depot_charge(v, now):
            return
        pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        raw = pick_batch_greedy_max_weight(
            pending, now, self.cfg.load_capacity, load_already=0.0
        )
        if not raw:
            return

        ordered = self._order_tasks_nn(self.depot, raw)
        while len(ordered) > 1:
            tour_d = self._tour_distance_with_return([t.node for t in ordered])
            if self._energy_need(tour_d) <= v.battery + 1e-9:
                break
            drop = min(ordered, key=lambda t: (t.weight, t.tid))
            ordered.remove(drop)
            ordered = self._order_tasks_nn(self.depot, ordered)
        if not ordered:
            return

        tour_d = self._tour_distance_with_return([t.node for t in ordered])
        need_tour = self._energy_need(tour_d)
        if need_tour > v.battery + 1e-9:
            if self._try_proactive_depot_charge(v, now):
                return
            return

        tids = [t.tid for t in ordered]
        for t in ordered:
            t.status = TaskStatus.ASSIGNED
            t.assigned_vehicle = v.vid
            self._pending_tids.discard(t.tid)

        v.carry_batch = tids
        v.batch_index = 0
        v.current_task = tids[0]
        v.load_used = sum(self.tasks[tid].weight for tid in tids)

        first = self.tasks[tids[0]]
        if not self._begin_leg_from_to(v, now, v.node, first.node, first.tid):
            self._rollback_batch(v, tids)

    def _complete_task_if_due(self, v: Vehicle, now: float) -> None:
        if v.current_task is None:
            return
        if now + 1e-9 < v.busy_until:
            return
        task = self.tasks[v.current_task]
        task.finish_time = now
        task.status = TaskStatus.DONE
        v.load_used -= task.weight
        cfg = self.cfg
        if now <= task.deadline:
            self.score += cfg.early_bonus_per_weight * task.weight
        else:
            self.score -= cfg.late_penalty_per_time * (now - task.deadline)
        self.score -= cfg.distance_penalty_coef * task.travel_distance

        v.batch_index += 1
        if v.batch_index < len(v.carry_batch):
            next_tid = v.carry_batch[v.batch_index]
            v.current_task = next_tid
            nt = self.tasks[next_tid]
            if not self._begin_leg_from_to(v, now, v.node, nt.node, next_tid):
                for j in range(v.batch_index, len(v.carry_batch)):
                    tj = self.tasks[v.carry_batch[j]]
                    if tj.status == TaskStatus.ASSIGNED:
                        tj.status = TaskStatus.PENDING
                        tj.assigned_vehicle = None
                        self._pending_tids.add(tj.tid)
                        v.load_used -= tj.weight
                v.carry_batch = []
                v.batch_index = 0
                v.current_task = None
                v.visual_segments = []
                v.battery_segments = []
        else:
            v.carry_batch = []
            v.batch_index = 0
            v.current_task = None
            v.visual_segments = []
            v.battery_segments = []

    def _return_depot(self, v: Vehicle, now: float) -> None:
        if v.node == self.depot:
            return
        d_back = self.dist_uv(v.node, self.depot)
        if math.isinf(d_back):
            return
        need = self._energy_need(d_back)
        if need <= v.battery + 1e-9:
            p = self._path_nodes(v.node, self.depot)
            tr = self._travel_time_for_path(p)
            b0 = v.battery
            b1 = b0 - need
            v.visual_segments = [(now, now + tr, p)] if p else []
            v.battery_segments = [(now, now + tr, b0, b1)]
            v.battery = b1
            v.node = self.depot
            v.busy_until = now + tr
            return

        best = self._best_charger_plan(v.node, self.depot, now, v.battery)
        if best is None:
            # 仍无法经站回仓：仅极低电量时做小幅应急（优先靠主动补能与规划）
            cfg = self.cfg
            b0 = v.battery
            if b0 >= cfg.battery_capacity * 0.12:
                return
            need_back = self._energy_need(d_back)
            gain = max(cfg.battery_capacity * 0.32, need_back * 0.45 + 50.0)
            b1 = min(cfg.battery_capacity, b0 + gain)
            if b1 <= b0 + 1e-6:
                return
            dur = 12.0
            v.visual_segments = [(now, now + dur, [v.node, v.node])]
            v.battery_segments = [(now, now + dur, b0, b1)]
            v.battery = b1
            v.busy_until = now + dur
            self.score -= max(12.0, 0.025 * cfg.late_penalty_per_time * 100.0)
            return
        station, cnode, d1, d2 = best
        e1 = self._energy_need(d1)
        p1 = self._path_nodes(v.node, cnode)
        p2 = self._path_nodes(cnode, self.depot)
        t_arrive_c = now + self._travel_time_for_path(p1)
        b0 = v.battery
        bat1 = v.battery - e1
        charge_start, charge_end = self._reserve_charge(station, v.vid, t_arrive_c, bat1)
        bat_after = self.cfg.battery_capacity
        e2 = self._energy_need(d2)
        if e2 > bat_after + 1e-9:
            self._cancel_last_session(station)
            return
        t_end = charge_end + self._travel_time_for_path(p2)
        v.visual_segments = []
        if p1:
            v.visual_segments.append((now, t_arrive_c, p1))
        v.visual_segments.append((t_arrive_c, charge_end, [cnode, cnode]))
        if p2:
            v.visual_segments.append((charge_end, t_end, p2))
        b_leg1 = bat1
        b_leg2_start = bat_after
        b_final = bat_after - e2
        bsegs: List[Tuple[float, float, float, float]] = [(now, t_arrive_c, b0, b_leg1)]
        if charge_start > t_arrive_c + 1e-9:
            bsegs.append((t_arrive_c, charge_start, b_leg1, b_leg1))
        if charge_end > charge_start + 1e-9:
            bsegs.append((charge_start, charge_end, b_leg1, b_leg2_start))
        bsegs.append((charge_end, t_end, b_leg2_start, b_final))
        v.battery_segments = bsegs
        v.battery = b_final
        v.node = self.depot
        v.busy_until = t_end

    def step(self, t: float, dt: float) -> None:
        """推进一个仿真时间步（供可视化逐步调用）。"""
        cfg = self.cfg
        if self._rng.random() < cfg.task_spawn_rate * dt:
            self._spawn_task(t)

        self._tick_chargers(t)

        for v in self.vehicles:
            self._complete_task_if_due(v, t)
            if v.busy_until <= t + 1e-9 and v.current_task is None:
                self._return_depot(v, t)

        for v in self.vehicles:
            if v.busy_until <= t + 1e-9 and v.current_task is None:
                self._assign_vehicle(v, t)

        for tid in list(self._pending_tids):
            task = self.tasks.get(tid)
            if task is None or task.status != TaskStatus.PENDING:
                self._pending_tids.discard(tid)
                continue
            if t > task.deadline:
                task.status = TaskStatus.EXPIRED
                self._pending_tids.discard(tid)
                self.score -= cfg.late_penalty_per_time * (t - task.deadline)

        for v in self.vehicles:
            if (
                v.node == self.depot
                and v.busy_until <= t + 1e-9
                and v.current_task is None
            ):
                v.visual_segments = []
                v.battery_segments = []

    def run(self) -> None:
        t = 0.0
        dt = 0.5
        while t <= self.cfg.sim_duration:
            self.step(t, dt)
            t += dt


def nearest_charger_node(
    chargers: Sequence[ChargingStation],
    sim: FleetSimulator,
    from_node: int,
) -> Optional[int]:
    best_n: Optional[int] = None
    best_d = math.inf
    for cs in chargers:
        d = sim.dist_uv(from_node, cs.node)
        if d < best_d:
            best_d = d
            best_n = cs.node
    return best_n


def preset_scenarios() -> List[SimConfig]:
    """三种以上不同规模（网格边长及配套参数相对初版 ×5）。"""
    base = dict(
        battery_capacity=500.0,
        load_capacity=250.0,
        energy_per_distance=0.8,
        travel_speed=1.0,
        charge_power=75.0,
        early_bonus_per_weight=10.0,
        late_penalty_per_time=15.0,
        distance_penalty_coef=0.01,
    )
    return [
        SimConfig(
            name="SMALL",
            rows=25,
            cols=25,
            num_vehicles=10,
            num_chargers=10,
            sim_duration=1000.0,
            task_spawn_rate=0.40,
            weight_range=(5.0, 40.0),
            deadline_slack_range=(125.0, 275.0),
            obstacle_cover_ratio=0.14,
            **base,
            seed=1,
        ),
        SimConfig(
            name="MEDIUM",
            rows=50,
            cols=50,
            num_vehicles=20,
            num_chargers=20,
            sim_duration=1500.0,
            task_spawn_rate=0.60,
            weight_range=(5.0, 75.0),
            deadline_slack_range=(100.0, 300.0),
            obstacle_cover_ratio=0.16,
            **base,
            seed=2,
        ),
        SimConfig(
            name="LARGE",
            rows=80,
            cols=80,
            num_vehicles=40,
            num_chargers=40,
            sim_duration=2000.0,
            task_spawn_rate=0.75,
            weight_range=(10.0, 125.0),
            deadline_slack_range=(90.0, 250.0),
            obstacle_cover_ratio=0.18,
            **base,
            seed=3,
        ),
        SimConfig(
            name="XL_STRESS",
            rows=100,
            cols=100,
            num_vehicles=30,
            num_chargers=25,
            sim_duration=1750.0,
            task_spawn_rate=1.10,
            weight_range=(15.0, 150.0),
            deadline_slack_range=(60.0, 175.0),
            obstacle_cover_ratio=0.20,
            **base,
            seed=4,
        ),
    ]


def node_to_grid_xy(node: int, cols: int) -> Tuple[int, int]:
    """节点编号 → (行, 列)，与 build_grid_graph 行主序一致。"""
    return node // cols, node % cols


_TASK_CSV_FIELDNAMES = [
    "scenario",
    "seed",
    "grid_rows",
    "grid_cols",
    "task_id",
    "spawn_time",
    "row",
    "col",
    "node_id",
    "weight",
    "deadline",
]

_META_CSV_FIELDNAMES = [
    "scenario",
    "seed",
    "grid_rows",
    "grid_cols",
    "num_vehicles",
    "num_chargers",
    "depot_row",
    "depot_col",
    "depot_node_id",
    "obstacle_count",
]

_OBSTACLE_CSV_FIELDNAMES = [
    "scenario",
    "seed",
    "grid_rows",
    "grid_cols",
    "obstacle_id",
    "row",
    "col",
    "node_id",
]


def write_export_readme_txt(output_dir: str) -> str:
    """
    在导出目录写入 README.txt，说明各 CSV 文件及列含义。
    返回写入文件的绝对路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "README.txt")
    content = """CSV 导出说明
================

目录中的文件按场景命名（例如 SMALL / MEDIUM / LARGE），每个场景包含三类 CSV：

1) <SCENARIO>_tasks.csv
   任务级明细，一行一个任务。
   - scenario: 场景名称
   - seed: 随机种子
   - grid_rows: 网格行数
   - grid_cols: 网格列数
   - task_id: 任务编号
   - spawn_time: 任务生成时间
   - row: 任务位置行坐标
   - col: 任务位置列坐标
   - node_id: 任务节点 id（行主序：row * grid_cols + col）
   - weight: 任务重量
   - deadline: 任务截止时间

2) <SCENARIO>_meta.csv
   场景级元信息，一般仅一行。
   - scenario: 场景名称
   - seed: 随机种子
   - grid_rows: 网格行数
   - grid_cols: 网格列数
   - num_vehicles: 车辆数量
   - num_chargers: 充电站数量
   - depot_row: 仓库位置行坐标
   - depot_col: 仓库位置列坐标
   - depot_node_id: 仓库节点 id（行主序）
   - obstacle_count: 障碍节点总数

3) <SCENARIO>_obstacles.csv
   障碍物位置明细，一行一个障碍节点。
   - scenario: 场景名称
   - seed: 随机种子
   - grid_rows: 网格行数
   - grid_cols: 网格列数
   - obstacle_id: 障碍顺序编号（按 node_id 升序后从 0 开始）
   - row: 障碍位置行坐标
   - col: 障碍位置列坐标
   - node_id: 障碍节点 id（行主序）
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.abspath(path)


def write_scenario_tasks_csv(sim: FleetSimulator, filepath: str) -> None:
    """将一次已运行结束的仿真中的全部任务写入单个 CSV。"""
    cfg = sim.cfg
    cols = cfg.cols
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_TASK_CSV_FIELDNAMES)
        w.writeheader()
        for tid in sorted(sim.tasks.keys()):
            task = sim.tasks[tid]
            r, c = node_to_grid_xy(task.node, cols)
            w.writerow(
                {
                    "scenario": cfg.name,
                    "seed": cfg.seed,
                    "grid_rows": cfg.rows,
                    "grid_cols": cfg.cols,
                    "task_id": task.tid,
                    "spawn_time": task.spawn_time,
                    "row": r,
                    "col": c,
                    "node_id": task.node,
                    "weight": task.weight,
                    "deadline": task.deadline,
                }
            )


def write_scenario_meta_csv(sim: FleetSimulator, filepath: str) -> None:
    """导出场景级元信息：仓库位置、车辆数量、障碍数量等。"""
    cfg = sim.cfg
    depot_r, depot_c = node_to_grid_xy(sim.depot, cfg.cols)
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_META_CSV_FIELDNAMES)
        w.writeheader()
        w.writerow(
            {
                "scenario": cfg.name,
                "seed": cfg.seed,
                "grid_rows": cfg.rows,
                "grid_cols": cfg.cols,
                "num_vehicles": cfg.num_vehicles,
                "num_chargers": cfg.num_chargers,
                "depot_row": depot_r,
                "depot_col": depot_c,
                "depot_node_id": sim.depot,
                "obstacle_count": len(sim.obstacles),
            }
        )


def write_scenario_obstacles_csv(sim: FleetSimulator, filepath: str) -> None:
    """导出障碍物位置明细（行/列/节点 id）。"""
    cfg = sim.cfg
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_OBSTACLE_CSV_FIELDNAMES)
        w.writeheader()
        for oid, node in enumerate(sorted(sim.obstacles)):
            r, c = node_to_grid_xy(node, cfg.cols)
            w.writerow(
                {
                    "scenario": cfg.name,
                    "seed": cfg.seed,
                    "grid_rows": cfg.rows,
                    "grid_cols": cfg.cols,
                    "obstacle_id": oid,
                    "row": r,
                    "col": c,
                    "node_id": node,
                }
            )


def export_three_scenarios_tasks_csv(output_dir: str) -> List[str]:
    """
    仅导出三种规模：各跑一遍完整仿真，在 output_dir 下各写一个 CSV。
    （与直接运行 main 不同，会重复仿真；一般改用默认 main 即可。）
    """
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []
    readme_path = write_export_readme_txt(output_dir)
    written.append(readme_path)
    for cfg in preset_scenarios()[:3]:
        sim = FleetSimulator(cfg)
        sim.run()
        safe_name = cfg.name.replace(os.sep, "_").replace("/", "_")
        path_tasks = os.path.join(output_dir, f"{safe_name}_tasks.csv")
        path_meta = os.path.join(output_dir, f"{safe_name}_meta.csv")
        path_obstacles = os.path.join(output_dir, f"{safe_name}_obstacles.csv")
        write_scenario_tasks_csv(sim, path_tasks)
        write_scenario_meta_csv(sim, path_meta)
        write_scenario_obstacles_csv(sim, path_obstacles)
        written.extend([path_tasks, path_meta, path_obstacles])
    return written


DEFAULT_TASK_EXPORT_DIR = "task_spawns_export"


def summarize(sim: FleetSimulator) -> str:
    done = sum(1 for t in sim.tasks.values() if t.status == TaskStatus.DONE)
    expired = sum(1 for t in sim.tasks.values() if t.status == TaskStatus.EXPIRED)
    pending = sum(1 for t in sim.tasks.values() if t.status == TaskStatus.PENDING)
    lines = [
        f"[{sim.cfg.name}] 最终得分: {sim.score:.2f}",
        f"  任务: 完成 {done} / 超时未接 {expired} / 仍待分配 {pending} / 总生成 {len(sim.tasks)}",
        f"  充电站: "
        + ", ".join(
            f"#{cs.sid}@节点{cs.node} 服务{cs.total_served}次 峰值占用{cs.peak_active}"
            for cs in sim.chargers
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    os.makedirs(DEFAULT_TASK_EXPORT_DIR, exist_ok=True)
    csv_written: List[str] = []
    readme_path = write_export_readme_txt(DEFAULT_TASK_EXPORT_DIR)
    csv_written.append(readme_path)
    for idx, cfg in enumerate(preset_scenarios()):
        sim = FleetSimulator(cfg)
        sim.run()
        print(summarize(sim))
        print()
        if idx < 3:
            safe_name = cfg.name.replace(os.sep, "_").replace("/", "_")
            path_tasks = os.path.join(DEFAULT_TASK_EXPORT_DIR, f"{safe_name}_tasks.csv")
            path_meta = os.path.join(DEFAULT_TASK_EXPORT_DIR, f"{safe_name}_meta.csv")
            path_obstacles = os.path.join(
                DEFAULT_TASK_EXPORT_DIR, f"{safe_name}_obstacles.csv"
            )
            write_scenario_tasks_csv(sim, path_tasks)
            write_scenario_meta_csv(sim, path_meta)
            write_scenario_obstacles_csv(sim, path_obstacles)
            csv_written.extend(
                [
                    os.path.abspath(path_tasks),
                    os.path.abspath(path_meta),
                    os.path.abspath(path_obstacles),
                ]
            )
    print("---")
    print(f"已写入 SMALL / MEDIUM / LARGE 场景 CSV（共 {len(csv_written)} 个文件）:")
    for p in csv_written:
        print(f"  {p}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--export-tasks", "-e"):
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "task_spawns_export"
        paths = export_three_scenarios_tasks_csv(out_dir)
        print(f"已写入目录: {out_dir}")
        for p in paths:
            print(f"  {p}")
    else:
        main()
