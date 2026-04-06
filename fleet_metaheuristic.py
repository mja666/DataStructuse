#!/usr/bin/env python3
"""
元启发式对比实验：在「相同配置、相同随机种子、相同任务生成流」下，
基线仍为 fleet_simulation.FleetSimulator（最近邻 + 按重量装批）。

本模块子类改进：
1) 代价与评分对齐：路程项 × distance_penalty_coef + 迟到时间 × late_penalty_per_time；
2) 多初解（最近邻 / EDD）、n≤6 精确枚举、SA + 交换 / 2-opt 段反序 / Or-opt 插入、轻量禁忌与降温重启；
3) 装批：仍按重量贪心装满，与同重任务按截止 EDD 排序（不改变重量优先结构）。

运行: python fleet_metaheuristic.py
"""

from __future__ import annotations

import math
import random
import tkinter as tk
from collections import deque
from itertools import permutations
from typing import Deque, List, Optional, Set, Tuple

from fleet_simulation import (
    FleetSimulator,
    SimConfig,
    Task,
    TaskStatus,
    Vehicle,
    _eligible_pending,
    preset_scenarios,
    summarize,
)
from fleet_visual import FleetVisualApp


def pick_batch_weight_then_edd(
    pending,
    now: float,
    load_cap: float,
    load_already: float = 0.0,
) -> List[Task]:
    """
    与基线相同的「重量降序 + 容量贪心」，但在同重量下按截止时刻（EDD）与 tid 打破平局，
    引入紧迫度而不改变重量主导结构，避免中大规模下装批顺序剧变拉低得分。
    """
    cand = sorted(
        _eligible_pending(pending, now),
        key=lambda t: (-t.weight, t.deadline, t.tid),
    )
    batch: List[Task] = []
    wsum = load_already
    for t in cand:
        if wsum + t.weight <= load_cap + 1e-9:
            batch.append(t)
            wsum += t.weight
    return batch


class MetaHeuristicFleetSimulator(FleetSimulator):
    """
    继承原仿真器：覆盖装批与批次内序；任务到达与 self._rng 消费顺序不变。
    元启发式随机数仅用 _meta_rng。
    """

    def __init__(self, cfg: SimConfig) -> None:
        super().__init__(cfg)
        self._meta_rng = random.Random(cfg.seed + 90_210)

    def _route_opt_t0(self) -> float:
        return float(getattr(self, "_route_opt_now", 0.0))

    def _meta_route_cost(self, start_node: int, ordered: List[Task], t0: float) -> float:
        """与 SimConfig 扣分系数对齐的代理目标：路长 + 送达迟到。"""
        if not ordered:
            return 0.0
        dist_tour = self._tour_distance_with_return([t.node for t in ordered])
        if math.isinf(dist_tour):
            return float("inf")
        cum = t0
        node = start_node
        late_sum = 0.0
        for t in ordered:
            cum += self._travel_time(self.dist_uv(node, t.node))
            late_sum += max(0.0, cum - t.deadline)
            node = t.node
        cum += self._travel_time(self.dist_uv(node, self.depot))
        dc = self.cfg.distance_penalty_coef
        lp = self.cfg.late_penalty_per_time
        return dc * dist_tour + lp * late_sum

    def _best_initial_route(self, start_node: int, ts: List[Task]) -> List[Task]:
        """在多种构造解中取代理代价最小者（调用父类最近邻，避免递归）。"""
        t0 = self._route_opt_t0()
        nn = FleetSimulator._order_tasks_nn(self, start_node, ts)
        edd = sorted(ts, key=lambda t: (t.deadline, t.tid))
        candidates = [nn, edd]
        return min(candidates, key=lambda perm: self._meta_route_cost(start_node, perm, t0))

    def _brute_optimal_route(self, start_node: int, ts: List[Task]) -> List[Task]:
        t0 = self._route_opt_t0()
        best: Optional[List[Task]] = None
        best_c = float("inf")
        for perm in permutations(ts):
            lst = list(perm)
            c = self._meta_route_cost(start_node, lst, t0)
            if c < best_c:
                best_c = c
                best = lst
        return best if best is not None else ts[:]

    def _neighbor_route(
        self, current: List[Task], n: int
    ) -> Tuple[List[Task], Optional[Tuple[int, int]]]:
        """
        返回邻域解；若产生交换边则返回 (i,j) 供禁忌记录（仅 swap 记禁忌）。
        """
        nb = current[:]
        r = self._meta_rng.random()
        tabu_key: Optional[Tuple[int, int]] = None
        if r < 0.34:
            i, j = self._meta_rng.sample(range(n), 2)
            nb[i], nb[j] = nb[j], nb[i]
            tabu_key = (min(i, j), max(i, j))
        elif r < 0.68:
            i, j = self._meta_rng.sample(range(n), 2)
            if i > j:
                i, j = j, i
            nb[i : j + 1] = list(reversed(nb[i : j + 1]))
        else:
            i = self._meta_rng.randrange(n)
            j = self._meta_rng.randrange(n - 1)
            if j >= i:
                j += 1
            x = nb.pop(i)
            nb.insert(j, x)
        return nb, tabu_key

    def _simulated_annealing_route_order(
        self, start_node: int, tasks: List[Task]
    ) -> List[Task]:
        ts = list(tasks)
        n = len(ts)
        if n <= 1:
            return ts

        t0 = self._route_opt_t0()

        if n <= 6:
            return self._brute_optimal_route(start_node, ts)

        current = self._best_initial_route(start_node, ts)
        c_cur = self._meta_route_cost(start_node, current, t0)
        if math.isinf(c_cur):
            return FleetSimulator._order_tasks_nn(self, start_node, ts)

        best = current[:]
        c_best = c_cur

        T0 = max(1.0, min(c_cur * 0.1, 600.0))
        T = T0
        alpha = 0.987
        max_iter = min(2000, 100 * n + 320)
        tabu_len = 28
        tabu: Deque[Tuple[int, int]] = deque()
        tabu_set: Set[Tuple[int, int]] = set()
        stagnation = 0
        no_accept_cap = max(120, 35 * n)

        for _ in range(max_iter):
            nb, tabu_key = self._neighbor_route(current, n)
            c_nb = self._meta_route_cost(start_node, nb, t0)
            if math.isinf(c_nb):
                T *= alpha
                if T < 1e-5:
                    break
                continue

            # 禁忌非改进移动；优于历史最优时视为特赦（仍可接受）
            tabu_block = (
                tabu_key is not None
                and tabu_key in tabu_set
                and c_nb >= c_best - 1e-9
            )
            if tabu_block:
                stagnation += 1
            else:
                dE = c_nb - c_cur
                accepted = dE < 0 or self._meta_rng.random() < math.exp(
                    -dE / max(T, 1e-9)
                )
                if accepted:
                    current = nb
                    c_cur = c_nb
                    stagnation = 0
                    if tabu_key is not None:
                        tabu.append(tabu_key)
                        tabu_set.add(tabu_key)
                        while len(tabu) > tabu_len:
                            tabu_set.discard(tabu.popleft())
                    if c_cur < c_best:
                        best = current[:]
                        c_best = c_cur
                else:
                    stagnation += 1

            if stagnation >= no_accept_cap:
                T = min(T0 * 0.85, max(T * 2.2, 15.0))
                stagnation = 0

            T *= alpha
            if T < 1e-5:
                break
        return best

    def _order_tasks_nn(self, start_node: int, tasks: List[Task]) -> List[Task]:
        return self._simulated_annealing_route_order(start_node, tasks)

    def _assign_vehicle(self, v: Vehicle, now: float) -> None:
        if v.node != self.depot:
            return
        if v.carry_batch:
            return
        pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        raw = pick_batch_weight_then_edd(
            pending, now, self.cfg.load_capacity, load_already=0.0
        )
        if not raw:
            return

        self._route_opt_now = now
        try:
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

            tids = [t.tid for t in ordered]
            for t in ordered:
                t.status = TaskStatus.ASSIGNED
                t.assigned_vehicle = v.vid

            v.carry_batch = tids
            v.batch_index = 0
            v.current_task = tids[0]
            v.load_used = sum(self.tasks[tid].weight for tid in tids)

            first = self.tasks[tids[0]]
            if not self._begin_leg_from_to(v, now, v.node, first.node, first.tid):
                self._rollback_batch(v, tids)
        finally:
            if hasattr(self, "_route_opt_now"):
                delattr(self, "_route_opt_now")


def _task_stream_signature(sim: FleetSimulator) -> Tuple[int, Tuple[Tuple[float, int, float], ...]]:
    items = []
    for tid in sorted(sim.tasks.keys()):
        t = sim.tasks[tid]
        items.append((t.spawn_time, t.node, t.weight))
    return (len(items), tuple(items))


def _obstacle_signature(sim: FleetSimulator) -> Tuple[int, Tuple[int, ...]]:
    obs = tuple(sorted(getattr(sim, "obstacles", set())))
    return (len(obs), obs)


def run_controlled_comparison() -> None:
    print("=" * 60)
    print("控制变量对比：基线(重量装批+最近邻) vs 元启发式(同重EDD装批+SA/精确序)")
    print("相同项：SimConfig、随机种子、任务生成/截止/重量、仓库与充电站、车辆与充电模型")
    print("不同项：同重装批 EDD 平局规则 + 批次内 SA/精确访问序")
    print("=" * 60)

    for cfg in preset_scenarios():
        print(f"\n>>> 规模 [{cfg.name}]  seed={cfg.seed}")
        base = FleetSimulator(cfg)
        base.run()
        meta = MetaHeuristicFleetSimulator(cfg)
        meta.run()

        sig_b = _task_stream_signature(base)
        sig_m = _task_stream_signature(meta)
        obs_b = _obstacle_signature(base)
        obs_m = _obstacle_signature(meta)
        if obs_b != obs_m:
            print("  [警告] 障碍物布局不一致（不应发生），请检查随机数消费顺序。")
            print(f"    基线: n={obs_b[0]}  元启发: n={obs_m[0]}")
        else:
            print(f"  [控制变量] 障碍物布局一致: 共 {obs_b[0]} 个障碍节点")
        if sig_b != sig_m:
            print("  [警告] 任务流签名不一致（不应发生），请检查随机数是否被元启发式污染。")
            print(f"    基线: n={sig_b[0]}  元启发: n={sig_m[0]}")
        else:
            print(f"  [控制变量] 任务流一致: 共 {sig_b[0]} 条任务 (spawn_time, node, weight 序列相同)")

        print("  --- 基线 (FleetSimulator / 最近邻) ---")
        for line in summarize(base).split("\n"):
            print("   ", line)
        print("  --- 元启发式 ---")
        for line in summarize(meta).split("\n"):
            print("   ", line)

        ds = meta.score - base.score
        print(f"  >>> 得分差 (元启发 - 基线): {ds:+.2f}")


def run_meta_visual() -> None:
    root = tk.Tk()
    FleetVisualApp(
        root,
        sim_builders={"元启发式": MetaHeuristicFleetSimulator},
        default_builder="元启发式",
    )
    root.mainloop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("--visual", "-v"):
        run_meta_visual()
    else:
        run_controlled_comparison()
