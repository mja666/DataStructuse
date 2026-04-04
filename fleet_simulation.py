"""
新能源物流车队协同调度 — 图结构道路 + 最短路径 + 最大任务（重量）优先策略

大作业要求要点（本实现覆盖）：
- 图表示道路，Dijkstra 寻路
- 车队规模、电量上限、载重上限；任务动态到达（时间、节点、重量随机）
- 评分：越早完成、路径越短越高；超时扣分
- 电量不足时前往充电站；充电站排队与并发槽位（负荷）
- 至少三种不同规模场景（SMALL / MEDIUM / LARGE）

策略：MAX_WEIGHT_FIRST（最大任务优先）。最近任务优先可另写策略函数对比。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------- 图与最短路 ----------------------------


def all_pairs_shortest_paths(
    n: int, adj: List[List[Tuple[int, float]]]
) -> Tuple[List[List[float]], List[List[int]]]:
    """Floyd–Warshall，返回 dist 与 nxt（重构路径用，-1 表示无后继）。"""
    dist = [[math.inf] * n for _ in range(n)]
    nxt = [[-1] * n for _ in range(n)]
    for u in range(n):
        dist[u][u] = 0.0
    for u in range(n):
        for v, w in adj[u]:
            if w < dist[u][v]:
                dist[u][v] = w
                nxt[u][v] = v
    for k in range(n):
        for i in range(n):
            if dist[i][k] == math.inf:
                continue
            for j in range(n):
                if dist[k][j] == math.inf:
                    continue
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    nxt[i][j] = nxt[i][k]
    return dist, nxt


def build_grid_graph(rows: int, cols: int) -> Tuple[int, List[List[Tuple[int, float]]]]:
    """行主序编号，四邻接，边权为欧氏距离。"""
    n = rows * cols
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    def coord(k: int) -> Tuple[int, int]:
        return k // cols, k % cols

    for u in range(n):
        r, c = coord(u)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = nr * cols + nc
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


def nearest_charger_node(
    chargers: Sequence[ChargingStation],
    dist: Sequence[Sequence[float]],
    from_node: int,
) -> Optional[int]:
    best_n: Optional[int] = None
    best_d = math.inf
    for cs in chargers:
        d = dist[from_node][cs.node]
        if d < best_d:
            best_d = d
            best_n = cs.node
    return best_n


# ---------------------------- 仿真核心 ----------------------------


class FleetSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        self.n, self.adj = build_grid_graph(cfg.rows, cfg.cols)
        self.depot = 0
        self.dist, self.nxt = all_pairs_shortest_paths(self.n, self.adj)

        self.tasks: Dict[int, Task] = {}
        self._next_tid = 0
        self._rng = random.Random(cfg.seed)
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

    def _place_chargers(self, k: int) -> List[ChargingStation]:
        """在图中均匀撒点充电站（避开 depot）。"""
        nodes = list(range(1, self.n))
        self._rng.shuffle(nodes)
        chosen = nodes[:k] if k <= len(nodes) else nodes
        return [
            ChargingStation(sid=i, node=node, slots=2)
            for i, node in enumerate(chosen)
        ]

    def _spawn_task(self, t: float) -> None:
        slack_lo, slack_hi = self.cfg.deadline_slack_range
        w_lo, w_hi = self.cfg.weight_range
        node = self._rng.randrange(self.n)
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
        self._next_tid += 1

    def _travel_time(self, distance: float) -> float:
        return distance / self.cfg.travel_speed

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

    def _assign_vehicle(self, v: Vehicle, now: float) -> None:
        if v.node != self.depot:
            return
        pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        task = pick_task_max_weight(pending, v, now, self.cfg.load_capacity)
        if task is None:
            return

        d_direct = self.dist[v.node][task.node]
        if math.isinf(d_direct):
            return

        need_direct = self._energy_need(d_direct)
        if need_direct <= v.battery + 1e-9:
            travel_t = self._travel_time(d_direct)
            v.battery -= need_direct
            v.node = task.node
            v.load_used += task.weight
            v.busy_until = now + travel_t
            task.status = TaskStatus.ASSIGNED
            task.assigned_vehicle = v.vid
            task.travel_distance = d_direct
            v.current_task = task.tid
            return

        cnode = nearest_charger_node(self.chargers, self.dist, v.node)
        if cnode is None:
            return
        d1 = self.dist[v.node][cnode]
        d2 = self.dist[cnode][task.node]
        if math.isinf(d1) or math.isinf(d2):
            return
        e1 = self._energy_need(d1)
        if e1 > v.battery + 1e-9:
            return

        station = self._station_on_node(cnode)
        if station is None:
            return

        t_arrive_c = now + self._travel_time(d1)
        bat_after_leg1 = v.battery - e1
        _, charge_end = self._reserve_charge(station, v.vid, t_arrive_c, bat_after_leg1)
        bat_after = self.cfg.battery_capacity
        e2 = self._energy_need(d2)
        if e2 > bat_after + 1e-9:
            self._cancel_last_session(station)
            return

        travel2 = self._travel_time(d2)
        arrive_task = charge_end + travel2
        v.battery = bat_after - e2
        v.node = task.node
        v.load_used += task.weight
        v.busy_until = arrive_task
        task.status = TaskStatus.ASSIGNED
        task.assigned_vehicle = v.vid
        task.travel_distance = d1 + d2
        v.current_task = task.tid

    def _complete_task_if_due(self, v: Vehicle, now: float) -> None:
        if v.current_task is None:
            return
        if now + 1e-9 < v.busy_until:
            return
        task = self.tasks[v.current_task]
        task.finish_time = now
        task.status = TaskStatus.DONE
        v.load_used -= task.weight
        v.current_task = None
        cfg = self.cfg
        if now <= task.deadline:
            self.score += cfg.early_bonus_per_weight * task.weight
        else:
            self.score -= cfg.late_penalty_per_time * (now - task.deadline)
        self.score -= cfg.distance_penalty_coef * task.travel_distance

    def _return_depot(self, v: Vehicle, now: float) -> None:
        if v.node == self.depot:
            return
        d_back = self.dist[v.node][self.depot]
        if math.isinf(d_back):
            return
        need = self._energy_need(d_back)
        if need <= v.battery + 1e-9:
            v.battery -= need
            v.node = self.depot
            v.busy_until = now + self._travel_time(d_back)
            return

        cnode = nearest_charger_node(self.chargers, self.dist, v.node)
        if cnode is None:
            return
        d1 = self.dist[v.node][cnode]
        d2 = self.dist[cnode][self.depot]
        if math.isinf(d1) or math.isinf(d2):
            return
        e1 = self._energy_need(d1)
        if e1 > v.battery + 1e-9:
            return
        station = self._station_on_node(cnode)
        if station is None:
            return
        t_arrive_c = now + self._travel_time(d1)
        bat1 = v.battery - e1
        _, charge_end = self._reserve_charge(station, v.vid, t_arrive_c, bat1)
        bat_after = self.cfg.battery_capacity
        e2 = self._energy_need(d2)
        if e2 > bat_after + 1e-9:
            self._cancel_last_session(station)
            return
        v.battery = bat_after - e2
        v.node = self.depot
        v.busy_until = charge_end + self._travel_time(d2)

    def run(self) -> None:
        cfg = self.cfg
        t = 0.0
        dt = 0.5
        while t <= cfg.sim_duration:
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

            for task in self.tasks.values():
                if task.status == TaskStatus.PENDING and t > task.deadline:
                    task.status = TaskStatus.EXPIRED
                    self.score -= cfg.late_penalty_per_time * (t - task.deadline)

            t += dt


def preset_scenarios() -> List[SimConfig]:
    """三种以上不同规模。"""
    base = dict(
        battery_capacity=100.0,
        load_capacity=50.0,
        energy_per_distance=0.8,
        travel_speed=1.0,
        charge_power=15.0,
        early_bonus_per_weight=2.0,
        late_penalty_per_time=3.0,
        distance_penalty_coef=0.05,
    )
    return [
        SimConfig(
            name="SMALL",
            rows=5,
            cols=5,
            num_vehicles=2,
            num_chargers=2,
            sim_duration=200.0,
            task_spawn_rate=0.08,
            weight_range=(1.0, 8.0),
            deadline_slack_range=(25.0, 55.0),
            **base,
            seed=1,
        ),
        SimConfig(
            name="MEDIUM",
            rows=10,
            cols=10,
            num_vehicles=4,
            num_chargers=4,
            sim_duration=300.0,
            task_spawn_rate=0.12,
            weight_range=(1.0, 15.0),
            deadline_slack_range=(20.0, 60.0),
            **base,
            seed=2,
        ),
        SimConfig(
            name="LARGE",
            rows=16,
            cols=16,
            num_vehicles=8,
            num_chargers=8,
            sim_duration=400.0,
            task_spawn_rate=0.15,
            weight_range=(2.0, 25.0),
            deadline_slack_range=(18.0, 50.0),
            **base,
            seed=3,
        ),
        SimConfig(
            name="XL_STRESS",
            rows=20,
            cols=20,
            num_vehicles=6,
            num_chargers=5,
            sim_duration=350.0,
            task_spawn_rate=0.22,
            weight_range=(3.0, 30.0),
            deadline_slack_range=(12.0, 35.0),
            **base,
            seed=4,
        ),
    ]


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
    for cfg in preset_scenarios():
        sim = FleetSimulator(cfg)
        sim.run()
        print(summarize(sim))
        print()


if __name__ == "__main__":
    main()
