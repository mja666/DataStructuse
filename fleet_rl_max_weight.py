#!/usr/bin/env python3
"""
强化学习（策略梯度 REINFORCE）+ 最大任务装批策略

- 装车：与基线相同，使用 fleet_simulation.pick_batch_greedy_max_weight（重量贪心）。
- 学习对象：同一批次内的**配送顺序**（指针策略：每步在剩余任务上做 softmax 选下一单）。
- 回报：与 fleet_metaheuristic 对齐的代理代价的相反数
  R = -(distance_penalty_coef * 路程 + late_penalty_per_time * 迟到时间总和)，
  在决策时刻 t0 下沿当前顺序模拟送达时刻（不含充电，与元启发式路由代理一致）。

依赖：仅标准库 + fleet_simulation。

训练默认权重：python fleet_rl_max_weight.py --train
对比实验：python fleet_rl_max_weight.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from fleet_simulation import (
    FleetSimulator,
    SimConfig,
    Task,
    TaskStatus,
    Vehicle,
    pick_batch_greedy_max_weight,
    preset_scenarios,
    summarize,
)

# 与 MetaHeuristicFleetSimulator._meta_route_cost 一致（不含充电细模）
def _route_proxy_cost(
    sim: FleetSimulator,
    start_node: int,
    ordered: Sequence[Task],
    t0: float,
) -> float:
    if not ordered:
        return 0.0
    dist_tour = sim._tour_distance_with_return([t.node for t in ordered])
    if math.isinf(dist_tour):
        return float("inf")
    cum = t0
    node = start_node
    late_sum = 0.0
    for t in ordered:
        cum += sim._travel_time(sim.dist_uv(node, t.node))
        late_sum += max(0.0, cum - t.deadline)
        node = t.node
    cum += sim._travel_time(sim.dist_uv(node, sim.depot))
    dc = sim.cfg.distance_penalty_coef
    lp = sim.cfg.late_penalty_per_time
    return dc * dist_tour + lp * late_sum


def _feature_row(
    sim: FleetSimulator,
    current_node: int,
    t0: float,
    cand: Task,
) -> List[float]:
    d1 = sim.dist_uv(current_node, cand.node)
    d2 = sim.dist_uv(sim.depot, cand.node)
    if math.isinf(d1):
        d1 = 1e12
    if math.isinf(d2):
        d2 = 1e12
    scale = max(1.0, float(sim.cfg.rows + sim.cfg.cols))
    slack_hi = max(1.0, sim.cfg.deadline_slack_range[1])
    slack = cand.deadline - t0
    lc = sim.cfg.load_capacity
    wn = cand.weight / lc if lc > 1e-9 else 0.0
    return [
        d1 / scale,
        d2 / scale,
        wn,
        max(0.0, slack) / slack_hi,
        min(5.0, d1 / max(1.0, slack + 1.0)),
        1.0,
    ]


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits) if logits else 0.0
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


class PointerPolicy:
    """线性打分 + softmax，每步选下一任务（全体 logits 加同一常数不改变分布，故无偏置项）。"""

    def __init__(self, dim: int = 6, rng: Optional[random.Random] = None) -> None:
        self.dim = dim
        r = rng or random.Random()
        self.w = [r.gauss(0, 0.02) for _ in range(dim)]
        self.w[0] -= 2.0

    def logits(
        self,
        sim: FleetSimulator,
        current_node: int,
        t0: float,
        remaining: List[Task],
    ) -> List[float]:
        out: List[float] = []
        for t in remaining:
            phi = _feature_row(sim, current_node, t0, t)
            out.append(sum(self.w[i] * phi[i] for i in range(self.dim)))
        return out

    def sample_order_with_trace(
        self,
        sim: FleetSimulator,
        start_node: int,
        tasks: List[Task],
        t0: float,
        rng: random.Random,
    ) -> Tuple[List[Task], List[Tuple[List[List[float]], List[float], int]]]:
        """
        随机序 + REINFORCE 用轨迹：每步 (phis, 概率分布 pr, 选中下标 idx)。
        """
        remaining = list(tasks)
        cur = start_node
        order: List[Task] = []
        trace: List[Tuple[List[List[float]], List[float], int]] = []
        while remaining:
            logits = self.logits(sim, cur, t0, remaining)
            pr = _softmax(logits)
            phis = [_feature_row(sim, cur, t0, t) for t in remaining]
            u = rng.random()
            acc = 0.0
            idx = 0
            for i, p in enumerate(pr):
                acc += p
                if u <= acc or i == len(pr) - 1:
                    idx = i
                    break
            trace.append((phis, pr, idx))
            nxt = remaining.pop(idx)
            order.append(nxt)
            cur = nxt.node
        return order, trace

    def apply_reinforce_trace(
        self,
        trace: List[Tuple[List[List[float]], List[float], int]],
        advantage: float,
        lr: float,
    ) -> None:
        for phis, pr, idx in trace:
            phi_sel = phis[idx]
            exp_phi = [0.0] * self.dim
            for j, p in enumerate(pr):
                for d in range(self.dim):
                    exp_phi[d] += p * phis[j][d]
            for d in range(self.dim):
                self.w[d] += lr * advantage * (phi_sel[d] - exp_phi[d])

    def greedy_order(
        self,
        sim: FleetSimulator,
        start_node: int,
        tasks: List[Task],
        t0: float,
    ) -> List[Task]:
        remaining = list(tasks)
        cur = start_node
        order: List[Task] = []
        while remaining:
            logits = self.logits(sim, cur, t0, remaining)
            best_i = max(range(len(remaining)), key=lambda i: logits[i])
            nxt = remaining.pop(best_i)
            order.append(nxt)
            cur = nxt.node
        return order

    def to_json(self) -> str:
        return json.dumps({"w": self.w, "b": 0.0, "dim": self.dim})

    @classmethod
    def from_json(cls, s: str) -> PointerPolicy:
        d = json.loads(s)
        p = cls(dim=int(d.get("dim", 6)))
        p.w = [float(x) for x in d["w"]]
        return p


_POLICY_PATH = Path(__file__).resolve().parent / "fleet_rl_policy.json"
_cached_policy: Optional[PointerPolicy] = None


def get_policy(path: Optional[Path] = None, reload: bool = False) -> PointerPolicy:
    global _cached_policy
    pth = path or _POLICY_PATH
    if _cached_policy is not None and not reload:
        return _cached_policy
    if pth.is_file():
        try:
            _cached_policy = PointerPolicy.from_json(pth.read_text(encoding="utf-8"))
            return _cached_policy
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    _cached_policy = PointerPolicy()
    return _cached_policy


def save_policy(policy: PointerPolicy, path: Optional[Path] = None) -> None:
    pth = path or _POLICY_PATH
    pth.write_text(policy.to_json(), encoding="utf-8")


def _train_template_sim(rng: random.Random) -> FleetSimulator:
    cfg = SimConfig(
        name="RL_TRAIN",
        rows=16,
        cols=16,
        num_vehicles=1,
        num_chargers=4,
        sim_duration=80.0,
        task_spawn_rate=0.05,
        weight_range=(5.0, 60.0),
        deadline_slack_range=(35.0, 140.0),
        battery_capacity=500.0,
        load_capacity=220.0,
        energy_per_distance=0.8,
        travel_speed=1.0,
        charge_power=75.0,
        early_bonus_per_weight=1.0,
        late_penalty_per_time=1.0,
        distance_penalty_coef=0.01,
        obstacle_cover_ratio=0.12,
        seed=rng.randint(0, 10_000_000),
    )
    return FleetSimulator(cfg)


def train_policy(
    episodes: int = 800,
    lr: float = 0.15,
    seed: int = 42_199,
    save_path: Optional[Path] = None,
) -> PointerPolicy:
    rng = random.Random(seed)
    policy = PointerPolicy()
    baseline = 0.0
    beta = 0.08
    k_max = 7

    for ep in range(episodes):
        sim = _train_template_sim(rng)
        if not sim._task_candidate_nodes:
            continue
        k = rng.randint(2, min(k_max, len(sim._task_candidate_nodes)))
        batch: List[Task] = []
        for i in range(k):
            node = rng.choice(sim._task_candidate_nodes)
            w = rng.uniform(*sim.cfg.weight_range)
            dl = rng.uniform(20.0, 180.0) + rng.uniform(*sim.cfg.deadline_slack_range) * 0.3
            batch.append(
                Task(
                    tid=i,
                    spawn_time=0.0,
                    node=node,
                    weight=w,
                    deadline=dl,
                )
            )
        t0 = rng.uniform(5.0, 120.0)
        order, trace = policy.sample_order_with_trace(sim, sim.depot, batch, t0, rng)
        cost = _route_proxy_cost(sim, sim.depot, order, t0)
        if math.isinf(cost):
            continue
        reward = -cost
        adv = reward - baseline
        baseline = (1.0 - beta) * baseline + beta * reward
        policy.apply_reinforce_trace(trace, adv, lr)

    save_policy(policy, save_path)
    global _cached_policy
    _cached_policy = policy
    return policy


class RLMaxWeightFleetSimulator(FleetSimulator):
    """
    最大任务（重量贪心装批）+ 强化学习排程：批次内顺序由 PointerPolicy 贪心决策。
    """

    def __init__(
        self,
        cfg: SimConfig,
        *,
        policy: Optional[PointerPolicy] = None,
        policy_path: Optional[Path] = None,
    ) -> None:
        super().__init__(cfg)
        self._rl_policy = policy or get_policy(policy_path)

    def _order_tasks_nn(self, start_node: int, tasks: List[Task]) -> List[Task]:
        if len(tasks) <= 1:
            return list(tasks)
        t0 = float(getattr(self, "_route_opt_now", 0.0))
        return self._rl_policy.greedy_order(self, start_node, tasks, t0)

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


def run_rl_comparison() -> None:
    print("=" * 60)
    print("强化学习 vs 基线（均为重量贪心装批；基线批次内最近邻，RL 为策略梯度指针序）")
    print("=" * 60)
    pol = get_policy()
    for cfg in preset_scenarios():
        print(f"\n>>> [{cfg.name}] seed={cfg.seed}")
        base = FleetSimulator(cfg)
        base.run()
        rl = RLMaxWeightFleetSimulator(cfg, policy=pol)
        rl.run()

        sig_b = _task_stream_signature(base)
        sig_r = _task_stream_signature(rl)
        if sig_b != sig_r:
            print("  [警告] 任务流不一致")
        else:
            print(f"  [控制变量] 任务流一致: {sig_b[0]} 条")

        print("  --- 基线 (最近邻序) ---")
        for line in summarize(base).split("\n"):
            print("   ", line)
        print("  --- 强化学习序 ---")
        for line in summarize(rl).split("\n"):
            print("   ", line)
        print(f"  >>> 得分差 (RL - 基线): {rl.score - base.score:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RL 最大任务装批 + 顺序策略")
    ap.add_argument(
        "--train",
        action="store_true",
        help="训练策略并写入 fleet_rl_policy.json",
    )
    ap.add_argument("--episodes", type=int, default=900, help="训练回合数")
    ap.add_argument("--lr", type=float, default=0.12, help="REINFORCE 学习率")
    args = ap.parse_args()
    if args.train:
        print(f"训练中 ({args.episodes} episodes)…")
        train_policy(episodes=args.episodes, lr=args.lr)
        print(f"已保存: {_POLICY_PATH}")
    else:
        run_rl_comparison()


if __name__ == "__main__":
    main()
