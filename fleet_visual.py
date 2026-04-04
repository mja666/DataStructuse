#!/usr/bin/env python3
"""图形界面：沿最短路插值移动、规划虚线、轨迹尾迹、图例。运行: python fleet_visual.py"""

from __future__ import annotations

import math
import tkinter as tk
from collections import defaultdict, deque
from tkinter import ttk

from fleet_simulation import FleetSimulator, SimConfig, TaskStatus, preset_scenarios


def _interp_on_path(
    t0: float,
    t1: float,
    path: list[int],
    now: float,
    dist: list[list[float]],
    cxy,
) -> tuple[float, float]:
    if not path:
        return cxy(0)
    if t1 <= t0 + 1e-12:
        return cxy(path[-1])
    if now <= t0:
        return cxy(path[0])
    if now >= t1:
        return cxy(path[-1])
    if len(path) == 1 or path[0] == path[-1]:
        return cxy(path[0])
    total_d = sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))
    if total_d < 1e-9:
        return cxy(path[0])
    frac = (now - t0) / (t1 - t0)
    target_d = frac * total_d
    acc = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        d = dist[a][b]
        if acc + d >= target_d - 1e-9:
            alpha = (target_d - acc) / d if d > 1e-9 else 0.0
            x0, y0 = cxy(a)
            x1, y1 = cxy(b)
            return (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0))
        acc += d
    return cxy(path[-1])


def _vehicle_xy(
    sim: FleetSimulator,
    v,
    now: float,
    cxy,
) -> tuple[float, float]:
    dist = sim.dist
    for t0, t1, path in v.visual_segments:
        if t0 - 1e-9 <= now <= t1 + 1e-9:
            return _interp_on_path(t0, t1, path, now, dist, cxy)
    return cxy(v.node)


def _flatten_route_points(segments: list[tuple[float, float, list[int]]], cxy) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for _t0, _t1, path in segments:
        if not path:
            continue
        for n in path:
            p = cxy(n)
            if not pts or (abs(pts[-1][0] - p[0]) > 0.5 or abs(pts[-1][1] - p[1]) > 0.5):
                pts.append(p)
            else:
                pts[-1] = p
    return pts


class FleetVisualApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fleet")
        self.scenarios = preset_scenarios()
        self.sim: FleetSimulator | None = None
        self.cfg: SimConfig | None = None
        self.t = 0.0
        self.dt = 0.5
        self.running = False
        self.steps_per_tick = 2
        self._trail_len = 140
        self._trails: dict[int, deque[tuple[float, float]]] = {}

        self._colors = {
            "bg": "#16161e",
            "grid": "#292e42",
            "cell": "#1f2233",
            "depot": "#e0af68",
            "charger": "#73daca",
            "task_pend": "#ff8a4c",
            "task_go": "#bb9af7",
            "bar_bg": "#24283b",
            "bar_time": "#3d59a1",
            "bar_pos": "#9ece6a",
            "bar_neg": "#f7768e",
            "route": "#565f89",
            "trail": "#3d4f6f",
            "legend_bg": "#1a1b26",
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

        top = ttk.Frame(root, padding=4)
        top.pack(fill=tk.X)
        self.combo = ttk.Combobox(
            top,
            state="readonly",
            width=14,
            values=[c.name for c in self.scenarios],
        )
        self.combo.current(0)
        self.combo.pack(side=tk.LEFT, padx=4)
        self.combo.bind("<<ComboboxSelected>>", self._on_scenario)

        ttk.Button(top, text="▶", width=3, command=self._play).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="⏸", width=3, command=self._pause).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="↻", width=3, command=self._restart).pack(side=tk.LEFT, padx=2)

        self.canvas = tk.Canvas(
            root,
            width=960,
            height=700,
            bg=self._colors["bg"],
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._restart()
        self._tick_loop()

    def _on_scenario(self, _evt=None) -> None:
        self._restart()

    def _play(self) -> None:
        self.running = True

    def _pause(self) -> None:
        self.running = False

    def _restart(self) -> None:
        self._pause()
        idx = self.combo.current()
        if idx < 0:
            idx = 0
        self.cfg = self.scenarios[idx]
        self.sim = FleetSimulator(self.cfg)
        self.t = 0.0
        self._trails = {}

    def _tick_loop(self) -> None:
        if self.sim and self.cfg and self.running and self.t <= self.cfg.sim_duration:
            for _ in range(self.steps_per_tick):
                if self.t > self.cfg.sim_duration:
                    break
                self.sim.step(self.t, self.dt)
                self.t += self.dt
            if self.t > self.cfg.sim_duration:
                self.running = False
        self._draw()
        self.root.after(42, self._tick_loop)

    def _node_rc(self, node: int) -> tuple[int, int]:
        assert self.cfg is not None
        return node // self.cfg.cols, node % self.cfg.cols

    def _draw(self) -> None:
        self.canvas.delete("all")
        if not self.sim or not self.cfg:
            return

        cfg = self.cfg
        sim = self.sim
        rows, cols = cfg.rows, cfg.cols
        W, H = int(self.canvas.cget("width")), int(self.canvas.cget("height"))
        bar_h = 26
        legend_h = 78
        pad = 14
        map_bottom = H - bar_h - legend_h - 10
        inner_w = W - pad * 2
        inner_h = map_bottom - pad * 2
        cs = max(10.0, min(40.0, min(inner_w / cols, inner_h / rows)))

        gw, gh = cols * cs, rows * cs
        ox = pad + (inner_w - gw) / 2
        oy = pad + (inner_h - gh) / 2

        def center(node: int) -> tuple[float, float]:
            r, c = self._node_rc(node)
            x = ox + c * cs + cs / 2
            y = oy + r * cs + cs / 2
            return x, y

        charger_nodes = {csn.node for csn in sim.chargers}

        for r in range(rows + 1):
            y = oy + r * cs
            self.canvas.create_line(ox, y, ox + gw, y, fill=self._colors["grid"], width=1)
        for c in range(cols + 1):
            x = ox + c * cs
            self.canvas.create_line(x, oy, x, oy + gh, fill=self._colors["grid"], width=1)

        for node in range(rows * cols):
            x0 = ox + (node % cols) * cs + 1
            y0 = oy + (node // cols) * cs + 1
            x1 = x0 + cs - 2
            y1 = y0 + cs - 2
            fill = self._colors["cell"]
            if node == sim.depot:
                fill = self._colors["depot"]
            elif node in charger_nodes:
                fill = "#1a3d36"
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")

        if sim.depot not in charger_nodes:
            cx, cy = center(sim.depot)
            self.canvas.create_rectangle(
                cx - cs * 0.28,
                cy - cs * 0.28,
                cx + cs * 0.28,
                cy + cs * 0.28,
                fill="",
                outline=self._colors["depot"],
                width=3,
            )

        for csn in sim.chargers:
            cx, cy = center(csn.node)
            r = cs * 0.22
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
                width=1,
            )

        pending_cnt: dict[int, int] = defaultdict(int)
        for task in sim.tasks.values():
            if task.status == TaskStatus.PENDING and task.spawn_time <= self.t:
                pending_cnt[task.node] += 1
        for node, k in pending_cnt.items():
            cx, cy = center(node)
            for i in range(min(k, 5)):
                ang = (i / max(k, 1)) * 2 * math.pi
                rr = min(3.0 + k * 0.35, 7.0)
                self.canvas.create_oval(
                    cx + math.cos(ang) * 4 - rr / 2,
                    cy + math.sin(ang) * 4 - rr / 2,
                    cx + math.cos(ang) * 4 + rr / 2,
                    cy + math.sin(ang) * 4 + rr / 2,
                    fill=self._colors["task_pend"],
                    outline="",
                )

        for task in sim.tasks.values():
            if task.status == TaskStatus.ASSIGNED:
                cx, cy = center(task.node)
                self.canvas.create_oval(
                    cx - 5,
                    cy - 5,
                    cx + 5,
                    cy + 5,
                    outline=self._colors["task_go"],
                    width=2,
                )

        for v in sim.vehicles:
            if v.visual_segments and self.t <= v.busy_until + 1e-9:
                pts = _flatten_route_points(v.visual_segments, center)
                if len(pts) >= 2:
                    flat = []
                    for p in pts:
                        flat.extend(p)
                    self.canvas.create_line(
                        *flat,
                        fill=self._colors["route"],
                        width=2,
                        dash=(5, 5),
                    )

        for v in sim.vehicles:
            if v.vid not in self._trails:
                self._trails[v.vid] = deque(maxlen=self._trail_len)
            px, py = _vehicle_xy(sim, v, self.t, center)
            tr = self._trails[v.vid]
            if not tr or math.hypot(tr[-1][0] - px, tr[-1][1] - py) > 0.7:
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
            vx, vy = _vehicle_xy(sim, v, self.t, center)
            oxv = 6 * math.cos(v.vid * 2.17)
            oyv = 6 * math.sin(v.vid * 2.17)
            vx += oxv
            vy += oyv
            col = self._vh[v.vid % len(self._vh)]
            r = max(5.0, cs * 0.15)
            self.canvas.create_oval(
                vx - r,
                vy - r,
                vx + r,
                vy + r,
                fill=col,
                outline="#1a1b26",
                width=2,
            )
            cap = cfg.battery_capacity
            ratio = max(0.0, min(1.0, v.battery / cap))
            bw = cs * 0.5
            bh = 3.5
            self.canvas.create_rectangle(
                vx - bw / 2,
                vy + r + 2,
                vx + bw / 2,
                vy + r + 2 + bh,
                fill=self._colors["bar_bg"],
                outline="",
            )
            hue = "#a6e3a1" if ratio > 0.35 else "#f9e2af" if ratio > 0.15 else "#f7768e"
            self.canvas.create_rectangle(
                vx - bw / 2,
                vy + r + 2,
                vx - bw / 2 + bw * ratio,
                vy + r + 2 + bh,
                fill=hue,
                outline="",
            )

        ly0 = map_bottom + 4
        cy_leg = ly0 + legend_h / 2 - 4
        self.canvas.create_rectangle(
            0,
            ly0,
            W,
            ly0 + legend_h - 2,
            fill=self._colors["legend_bg"],
            outline="#292e42",
            width=1,
        )

        def leg_item(x: float, draw_fn, text: str) -> None:
            draw_fn(x, cy_leg)
            self.canvas.create_text(
                x + 22,
                cy_leg,
                text=text,
                fill="#a9b1d6",
                font=("Microsoft YaHei UI", 9),
                anchor=tk.W,
            )

        xi = 12.0

        def d_depot(x: float, cy: float) -> None:
            self.canvas.create_rectangle(
                x,
                cy - 8,
                x + 16,
                cy + 8,
                fill=self._colors["depot"],
                outline="#1a1b26",
            )

        leg_item(xi, d_depot, "仓库")
        xi += 88

        def d_chg(x: float, cy: float) -> None:
            self.canvas.create_polygon(
                x + 8,
                cy - 7,
                x + 15,
                cy,
                x + 8,
                cy + 7,
                x + 1,
                cy,
                fill=self._colors["charger"],
                outline="#1a1b26",
            )

        leg_item(xi, d_chg, "充电站")
        xi += 100

        def d_pend(x: float, cy: float) -> None:
            self.canvas.create_oval(x + 4, cy - 4, x + 12, cy + 4, fill=self._colors["task_pend"], outline="")

        leg_item(xi, d_pend, "待接任务")
        xi += 110

        def d_go(x: float, cy: float) -> None:
            self.canvas.create_oval(x + 2, cy - 5, x + 14, cy + 5, outline=self._colors["task_go"], width=2)

        leg_item(xi, d_go, "配送中订单")
        xi += 130

        def d_car(x: float, cy: float) -> None:
            self.canvas.create_oval(x + 2, cy - 6, x + 14, cy + 6, fill=self._vh[0], outline="#1a1b26", width=2)
            self.canvas.create_rectangle(x + 3, cy + 7, x + 13, cy + 10, fill="#a6e3a1", outline="")

        leg_item(xi, d_car, "车辆·下为电量")
        xi += 150

        self.canvas.create_line(
            xi,
            cy_leg - 6,
            xi + 28,
            cy_leg + 6,
            fill=self._colors["route"],
            width=2,
            dash=(4, 3),
        )
        self.canvas.create_text(
            xi + 36,
            cy_leg,
            text="规划路径",
            fill="#a9b1d6",
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
        )
        xi += 110

        self.canvas.create_line(
            xi,
            cy_leg,
            xi + 30,
            cy_leg,
            fill=self._colors["trail"],
            width=3,
            smooth=True,
        )
        self.canvas.create_text(
            xi + 38,
            cy_leg,
            text="已走轨迹",
            fill="#a9b1d6",
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
        )

        bx0, bx1 = ox, ox + gw
        by0 = H - bar_h - 6
        by1 = H - 8
        self.canvas.create_rectangle(bx0, by0, bx1, by1, fill=self._colors["bar_bg"], outline="")
        prog = 0.0 if cfg.sim_duration <= 0 else min(1.0, self.t / cfg.sim_duration)
        self.canvas.create_rectangle(
            bx0,
            by0,
            bx0 + (bx1 - bx0) * prog,
            by1,
            fill=self._colors["bar_time"],
            outline="",
        )
        self.canvas.create_text(
            (bx0 + bx1) / 2,
            (by0 + by1) / 2,
            text="仿真进度",
            fill="#565f89",
            font=("Microsoft YaHei UI", 8),
        )

        smin, smax = -600.0, 600.0
        sc = max(smin, min(smax, sim.score))
        tmid = (bx0 + bx1) / 2
        sx = tmid + (sc / smax) * (gw / 2 - 8)
        sx = max(bx0 + 4, min(bx1 - 4, sx))
        self.canvas.create_line(tmid, by0 - 3, tmid, by1 + 3, fill="#414868", width=1)
        col = self._colors["bar_pos"] if sc >= 0 else self._colors["bar_neg"]
        self.canvas.create_polygon(
            sx,
            by0 - 5,
            sx - 4,
            by0 - 12,
            sx + 4,
            by0 - 12,
            fill=col,
            outline="",
        )
        self.canvas.create_text(
            bx1 - 4,
            by0 - 14,
            text="得分",
            fill="#565f89",
            font=("Microsoft YaHei UI", 8),
            anchor=tk.E,
        )


def main() -> None:
    root = tk.Tk()
    FleetVisualApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
