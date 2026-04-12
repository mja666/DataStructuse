"""
最近任务优先策略 — 与 fleet_simulation.py 并列使用，不修改原文件。

与当前「最大任务优先」仿真骨架一致（多票同车、最近邻排程、电量不足丢最轻单等）：
- 装车阶段：在载重上限内，从仓库出发反复选**图最短路意义下最近**且仍可容纳的任务装入同一批次；
- 装批完成后仍用基类 `_order_tasks_nn` 对配送点重排，并按 `_tour_distance_with_return` 与基类相同逻辑
  在电量不足时逐次丢弃最轻任务直至全程可执行；
- 行驶与充电、回仓、得分、可视化由基类 `_begin_leg_from_to` / `step` 等完成。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

from fleet_simulation import (
    FleetSimulator,
    Task,
    TaskStatus,
    Vehicle,
    _eligible_pending,
    preset_scenarios,
    summarize,
)


def pick_batch_greedy_nearest(
    pending: Iterable[Task],
    now: float,
    load_cap: float,
    sim: FleetSimulator,
    start_node: int,
    load_already: float = 0.0,
) -> List[Task]:
    """
    在载重上限内贪心装批：从 start_node 起，每次在剩余可选任务中选 dist(当前位置, 任务点) 最小者；
    装入后当前位置更新为该任务节点，直至无法再装或没有可达任务。
    """
    remaining = list(_eligible_pending(pending, now))
    batch: List[Task] = []
    wsum = load_already
    cur = start_node
    while remaining:
        best: Optional[Task] = None
        best_d = math.inf
        for t in remaining:
            if wsum + t.weight > load_cap + 1e-9:
                continue
            d = sim.dist_uv(cur, t.node)
            if math.isinf(d):
                continue
            if best is None or d < best_d or (d == best_d and t.tid < best.tid):
                best = t
                best_d = d
        if best is None:
            break
        batch.append(best)
        remaining.remove(best)
        wsum += best.weight
        cur = best.node
    return batch


class FleetSimulatorNearestFirst(FleetSimulator):
    def _assign_vehicle(self, v: Vehicle, now: float) -> None:
        if v.node != self.depot:
            return
        if v.carry_batch:
            return
        if self._try_proactive_depot_charge(v, now):
            return
        pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        raw = pick_batch_greedy_nearest(
            pending, now, self.cfg.load_capacity, self, self.depot, load_already=0.0
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


def main() -> None:
    for cfg in preset_scenarios():
        sim = FleetSimulatorNearestFirst(cfg)
        sim.run()
        print(summarize(sim))
        print()


if __name__ == "__main__":
    main()
