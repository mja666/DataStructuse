#!/usr/bin/env python3
"""
OSM real-map fleet scheduling animation + Gradio web UI.

Goals:
- Reuse the OSM simulation logic from `fleet_osm.py` (same step/update behavior).
- Render on real OSM road geometry (not grid blocks).
- Provide a Gradio webpage for selecting strategy and map scale.

Run:
    python osm_gradio_visual.py

Optional quick test (renders a tiny GIF without launching Gradio):
    python osm_gradio_visual.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.font_manager as font_manager
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

try:
    import contextily as ctx  # type: ignore
except Exception:
    ctx = None

try:
    from pyproj import Transformer  # type: ignore
except Exception:
    Transformer = None

from fleet_visual import _flatten_route_points, _vehicle_battery, _vehicle_xy


WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "tmp_outputs"
DEFAULT_EXTERNAL_FLEET_OSM = Path("D:/ds/fleet_osm.py")
BASEMAP_CACHE_DIR = WORKSPACE_DIR / "basemap_cache"
MAX_ANIMATION_DURATION_S = 1800.0


def _get_cjk_font_properties() -> Optional[font_manager.FontProperties]:
    cached = getattr(_get_cjk_font_properties, "_cache", None)
    if cached is not None:
        return cached

    # Prefer common Windows CJK fonts, then broader cross-platform choices.
    candidates = [
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "SimSun",
        "PingFang SC",
        "Heiti SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
    ]
    chosen = None
    for name in candidates:
        prop = font_manager.FontProperties(family=name)
        try:
            font_path = font_manager.findfont(prop, fallback_to_default=False)
        except Exception:
            continue
        if font_path and Path(font_path).is_file():
            chosen = font_manager.FontProperties(fname=font_path)
            break

    setattr(_get_cjk_font_properties, "_cache", chosen)
    return chosen


def _load_fleet_osm_module(explicit_path: Optional[str] = None):
    """Load fleet_osm module from import path or explicit file path."""
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.is_file():
            return _load_module_from_file(candidate)
        raise FileNotFoundError(f"fleet_osm.py not found: {candidate}")

    try:
        import fleet_osm  # type: ignore

        return fleet_osm
    except Exception:
        pass

    candidates = [
        WORKSPACE_DIR / "fleet_osm.py",
        DEFAULT_EXTERNAL_FLEET_OSM,
    ]
    for c in candidates:
        if c.is_file():
            return _load_module_from_file(c)

    raise RuntimeError(
        "Cannot import fleet_osm.py. Use --fleet-osm-path to specify its absolute path."
    )


def _load_module_from_file(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("fleet_osm_dynamic", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/type lookups that rely on sys.modules work.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_scenarios(
    mod: Any,
    seed: Optional[int],
    segments_override: Optional[Sequence[Any]] = None,
) -> Dict[str, Tuple[Any, Any]]:
    presets = mod.osm_presets_for_run(seed)
    export_csv_root = WORKSPACE_DIR / "osm_export_csv"

    if segments_override is not None:
        kwargs = {}
        if export_csv_root.is_dir():
            kwargs["export_csv_root"] = export_csv_root
        scenarios = mod.build_scenario_triples_from_presets(
            presets,
            list(segments_override),
            **kwargs,
        )
    else:
        use_local_direct = False
        if export_csv_root.is_dir():
            use_local_direct = all(
                (export_csv_root / f"{p.name}_map_nodes.csv").is_file()
                and (export_csv_root / f"{p.name}_map_edges.csv").is_file()
                for p in presets
            )

        if use_local_direct:
            scenarios = []
            for p in presets:
                segs = mod._load_local_segments_for_preset(p, export_csv_root=export_csv_root)
                prep = mod.prepare_road_network(
                    segs,
                    mod.random.Random(p.cfg.seed + 17_017),
                    base_speed_mps=p.cfg.travel_speed,
                )
                scenarios.append((p.name, prep, p.cfg))
        else:
            kwargs = {}
            if export_csv_root.is_dir():
                kwargs["export_csv_root"] = export_csv_root
            scenarios = mod.build_scenario_triples_from_presets(
                presets,
                segments_override=None,
                **kwargs,
            )

    out: Dict[str, Tuple[Any, Any]] = {}
    for name, prep, cfg in scenarios:
        out[name] = (prep, cfg)
    return out


def _project_factory(
    lonlat: Sequence[Tuple[float, float]],
    use_osm_basemap: bool,
):
    """
    Return:
      project(lon,lat)->(x,y), transformer_used(bool), bounds(xmin,xmax,ymin,ymax)
    """
    use_mercator = bool(use_osm_basemap and Transformer is not None)
    transformer = None
    if use_mercator:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    xs: List[float] = []
    ys: List[float] = []

    def project(lon: float, lat: float) -> Tuple[float, float]:
        if transformer is not None:
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        return float(x), float(y)

    for lon, lat in lonlat:
        x, y = project(lon, lat)
        xs.append(x)
        ys.append(y)

    pad_x = (max(xs) - min(xs)) * 0.04 + 1e-9
    pad_y = (max(ys) - min(ys)) * 0.04 + 1e-9
    bounds = (min(xs) - pad_x, max(xs) + pad_x, min(ys) - pad_y, max(ys) + pad_y)
    return project, bool(transformer is not None), bounds


def _unique_edges(adj: List[List[Tuple[int, float]]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for u, nbrs in enumerate(adj):
        for v, _w in nbrs:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
    return out


def _fallback_edge_congest_visual_level(
    dense: float,
    t: float,
    period: float,
    u: int,
    v: int,
) -> float:
    a, b = (u, v) if u < v else (v, u)
    phase = ((a * 1103515245 + b * 12345) & 0xFFFFFF) / float(0x1000000) * 2.0 * math.pi
    per = max(period, 1e-3)
    wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / per + phase)
    return max(0.0, min(1.0, dense * (0.22 + 0.78 * wave)))


def _fallback_edge_color_for_congest_level(level: float) -> str:
    lo = (0x3B, 0x42, 0x61)
    hi = (0xF7, 0x66, 0x6E)
    t = max(0.0, min(1.0, level))
    r = int(lo[0] + (hi[0] - lo[0]) * t + 0.5)
    g = int(lo[1] + (hi[1] - lo[1]) * t + 0.5)
    b = int(lo[2] + (hi[2] - lo[2]) * t + 0.5)
    return f"#{r:02x}{g:02x}{b:02x}"


def _ascii_safe(text: str) -> str:
    parts: List[str] = []
    for ch in text:
        if ch.isalnum() and ord(ch) < 128:
            parts.append(ch)
        else:
            parts.append(f"u{ord(ch):x}")
    cleaned = "_".join(parts)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "item"


def _basemap_cache_file(
    bounds: Tuple[float, float, float, float],
    fig: matplotlib.figure.Figure,
    provider_id: str,
    cache_tag: str,
) -> Path:
    xmin, xmax, ymin, ymax = bounds
    provider_slug = _ascii_safe(provider_id)[:60]
    tag_slug = _ascii_safe(cache_tag)[:40]
    key_payload = {
        "provider": provider_id,
        "bounds": [round(xmin, 3), round(xmax, 3), round(ymin, 3), round(ymax, 3)],
        "fig": [
            int(round(float(fig.get_figwidth()) * float(fig.dpi))),
            int(round(float(fig.get_figheight()) * float(fig.dpi))),
            int(fig.dpi),
        ],
        "tag": cache_tag,
        "crs": "EPSG:3857",
        "cache_version": 2,
    }
    digest = hashlib.sha1(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return BASEMAP_CACHE_DIR / f"basemap_{tag_slug}_{provider_slug}_{digest}.npz"


def _load_cached_basemap(ax, cache_file: Path) -> bool:
    if not cache_file.is_file():
        return False
    try:
        with np.load(str(cache_file)) as npz:
            img = np.asarray(npz["image"])
            ext = [float(v) for v in np.asarray(npz["extent"]).tolist()]
        ax.imshow(img, extent=ext, interpolation="bilinear", zorder=0, origin="upper")
        return True
    except Exception:
        return False


def _save_basemap_from_axis(ax, cache_file: Path) -> None:
    if not ax.images:
        return
    img_artist = ax.images[-1]
    arr = np.asarray(img_artist.get_array())
    ext = np.asarray(img_artist.get_extent(), dtype=float)
    BASEMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_file), image=arr, extent=ext)


def _add_osm_basemap_with_cache(
    ax,
    bounds: Tuple[float, float, float, float],
    provider,
    cache_tag: str,
) -> bool:
    provider_id = str(provider)
    cache_file = _basemap_cache_file(bounds, ax.figure, provider_id, cache_tag)
    if _load_cached_basemap(ax, cache_file):
        return True
    ctx.add_basemap(
        ax,
        source=provider,
        zoom="auto",
        attribution="",
    )
    _save_basemap_from_axis(ax, cache_file)
    return False


def _save_animation_fast(
    ani: FuncAnimation,
    out_dir: Path,
    file_stem: str,
    fps: int,
) -> Tuple[Path, str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fps_eff = max(1, int(fps))
    mp4_path = out_dir / f"{file_stem}.mp4"
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is unavailable; cannot export MP4")
    writer = FFMpegWriter(
        fps=fps_eff,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart"],
    )
    ani.save(str(mp4_path), writer=writer)
    return mp4_path, "mp4", "ffmpeg"


def _build_animation_runtime(
    fleet_osm_mod: Any,
    strategy: str,
    scale_name: str,
    seed: Optional[int],
    dt: float,
    duration_s: float,
    steps_per_frame: int,
    fps: int,
    use_osm_basemap: bool,
    show_battery: bool,
    segments_override: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    scenarios = _prepare_scenarios(
        fleet_osm_mod,
        seed,
        segments_override=segments_override,
    )
    if scale_name not in scenarios:
        raise ValueError(f"Unknown scale: {scale_name}")
    if strategy not in fleet_osm_mod.OSM_SIM_BUILDERS:
        raise ValueError(f"Unknown strategy: {strategy}")

    prep, cfg = scenarios[scale_name]
    sim_cls = fleet_osm_mod.OSM_SIM_BUILDERS[strategy]
    sim = sim_cls(cfg, prep)
    # Keep live duration aligned with UI slider max instead of preset-specific sim_duration.
    duration_limit = max(0.5, min(float(duration_s), MAX_ANIMATION_DURATION_S))
    sim_t = 0.0

    project, use_mercator, bounds = _project_factory(sim.node_lonlat, use_osm_basemap)
    xnodes: List[float] = []
    ynodes: List[float] = []
    for lon, lat in sim.node_lonlat:
        x, y = project(lon, lat)
        xnodes.append(x)
        ynodes.append(y)

    def cxy(node: int) -> Tuple[float, float]:
        return xnodes[node], ynodes[node]

    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=120)
    ax.set_facecolor("#0f1220")
    fig.patch.set_facecolor("#0f1220")

    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    basemap_cache_hit = False
    if use_osm_basemap and use_mercator and ctx is not None:
        try:
            basemap_cache_hit = _add_osm_basemap_with_cache(
                ax,
                bounds,
                ctx.providers.CartoDB.PositronNoLabels,
                cache_tag=scale_name,
            )
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        except Exception:
            pass

    edge_pairs = _unique_edges(sim.adj)
    edge_segments = [[cxy(u), cxy(v)] for u, v in edge_pairs]

    edge_congest_base = getattr(sim, "edge_congest_base", {}) or {}
    edge_level_fn = getattr(fleet_osm_mod, "_edge_congest_visual_level", None)
    edge_color_fn = getattr(fleet_osm_mod, "_edge_color_for_congest_level", None)
    if edge_level_fn is None:
        edge_level_fn = _fallback_edge_congest_visual_level
    if edge_color_fn is None:
        edge_color_fn = _fallback_edge_color_for_congest_level
    congest_period = float(max(80.0, min(220.0, float(cfg.sim_duration) / 5.5)))

    edge_colors = []
    for u, v in edge_pairs:
        dense = float(edge_congest_base.get((u, v), 0.35))
        lvl = edge_level_fn(dense, 0.0, congest_period, u, v)
        edge_colors.append(edge_color_fn(lvl))

    lc = LineCollection(edge_segments, colors=edge_colors, linewidths=0.8, alpha=0.8, zorder=2)
    ax.add_collection(lc)

    dx = xmax - xmin
    dy = ymax - ymin
    mark_size = max(40.0, 8000.0 / max(1.0, len(sim.node_lonlat)))

    depot_x, depot_y = cxy(sim.depot)
    ax.scatter([depot_x], [depot_y], s=mark_size * 2.1, c="#e0af68", marker="s", zorder=7)

    charger_x = [cxy(cs.node)[0] for cs in sim.chargers]
    charger_y = [cxy(cs.node)[1] for cs in sim.chargers]
    ax.scatter(charger_x, charger_y, s=mark_size * 1.6, c="#73daca", marker="D", zorder=7)

    pending_sc = ax.scatter([], [], s=22, c="#ff8a4c", alpha=0.9, zorder=8)
    assigned_sc = ax.scatter([], [], s=46, facecolors="none", edgecolors="#bb9af7", linewidths=1.6, zorder=8)

    vehicle_palette = [
        "#7aa2f7",
        "#7dcfff",
        "#f7768e",
        "#e0af68",
        "#bb9af7",
        "#9ece6a",
        "#ff9e64",
        "#c0caf5",
    ]
    v_colors = [vehicle_palette[v.vid % len(vehicle_palette)] for v in sim.vehicles]
    v_init = [cxy(v.node) for v in sim.vehicles]
    vehicle_sc = ax.scatter(
        [p[0] for p in v_init],
        [p[1] for p in v_init],
        s=64,
        c=v_colors,
        edgecolors="#1a1b26",
        linewidths=0.8,
        zorder=9,
    )

    bar_w = max(dx, dy) * 0.028
    bar_h = max(dx, dy) * 0.004
    bar_offset = max(dx, dy) * 0.012
    battery_bg_rects: List[Rectangle] = []
    battery_fg_rects: List[Rectangle] = []
    for x, y in v_init:
        bg = Rectangle(
            (x - bar_w / 2.0, y + bar_offset),
            bar_w,
            bar_h,
            facecolor="#24283b",
            edgecolor="none",
            zorder=10,
            visible=bool(show_battery),
        )
        fg = Rectangle(
            (x - bar_w / 2.0, y + bar_offset),
            bar_w,
            bar_h,
            facecolor="#5fbf7a",
            edgecolor="none",
            zorder=11,
            visible=bool(show_battery),
        )
        ax.add_patch(bg)
        ax.add_patch(fg)
        battery_bg_rects.append(bg)
        battery_fg_rects.append(fg)

    trail_lines = []
    route_lines = []
    for _ in sim.vehicles:
        trail_line, = ax.plot([], [], color="#5b6b96", linewidth=1.3, alpha=0.9, zorder=5)
        route_line, = ax.plot([], [], color="#6c79a6", linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.9, zorder=6)
        trail_lines.append(trail_line)
        route_lines.append(route_line)

    trails: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    trail_sample = max(dx, dy) * 0.0025

    n_frames = max(2, int(math.ceil(duration_limit / (dt * max(1, steps_per_frame)))) + 1)

    def _task_points() -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        pending_pts: List[Tuple[float, float]] = []
        assigned_pts: List[Tuple[float, float]] = []
        TaskStatus = fleet_osm_mod.TaskStatus
        for task in sim.tasks.values():
            st = task.status
            if st == TaskStatus.PENDING and task.spawn_time <= sim_t:
                pending_pts.append(cxy(task.node))
            elif st == TaskStatus.ASSIGNED:
                assigned_pts.append(cxy(task.node))
        return pending_pts, assigned_pts

    def update(_frame_idx: int):
        nonlocal sim_t

        for _ in range(max(1, steps_per_frame)):
            if sim_t > duration_limit + 1e-9:
                break
            sim.step(sim_t, dt)
            sim_t += dt

        pending_pts, assigned_pts = _task_points()
        if pending_pts:
            pending_sc.set_offsets(pending_pts)
        else:
            pending_sc.set_offsets([[float("nan"), float("nan")]])

        if assigned_pts:
            assigned_sc.set_offsets(assigned_pts)
        else:
            assigned_sc.set_offsets([[float("nan"), float("nan")]])

        # Road color changes with simulated congestion wave over time.
        dyn_edge_colors = []
        for u, v in edge_pairs:
            dense = float(edge_congest_base.get((u, v), 0.35))
            lvl = edge_level_fn(dense, sim_t, congest_period, u, v)
            dyn_edge_colors.append(edge_color_fn(lvl))
        lc.set_color(dyn_edge_colors)

        vxy = []
        for v in sim.vehicles:
            x, y = _vehicle_xy(sim, v, sim_t, cxy)
            vxy.append((x, y))
            tr = trails[v.vid]
            if not tr or math.hypot(tr[-1][0] - x, tr[-1][1] - y) > trail_sample:
                tr.append((x, y))
            if len(tr) > 420:
                del tr[: len(tr) - 420]

        vehicle_sc.set_offsets(vxy)

        if show_battery:
            cap = float(getattr(cfg, "battery_capacity", 0.0) or 0.0)
            for i, v in enumerate(sim.vehicles):
                vx, vy = vxy[i]
                bg = battery_bg_rects[i]
                fg = battery_fg_rects[i]
                x0 = vx - bar_w / 2.0
                y0 = vy + bar_offset
                bg.set_visible(True)
                fg.set_visible(True)
                bg.set_xy((x0, y0))
                bg.set_width(bar_w)
                bg.set_height(bar_h)

                cur_bat = float(_vehicle_battery(v, sim_t))
                ratio = max(0.0, min(1.0, cur_bat / cap)) if cap > 1e-9 else 0.0
                hue = "#5fbf7a" if ratio > 0.35 else "#d2ad48" if ratio > 0.15 else "#d15866"
                fg.set_xy((x0, y0))
                fg.set_width(bar_w * ratio)
                fg.set_height(bar_h)
                fg.set_facecolor(hue)
        else:
            for bg in battery_bg_rects:
                bg.set_visible(False)
            for fg in battery_fg_rects:
                fg.set_visible(False)

        for i, v in enumerate(sim.vehicles):
            tr = trails[v.vid]
            if len(tr) >= 2:
                trail_lines[i].set_data([p[0] for p in tr], [p[1] for p in tr])
            else:
                trail_lines[i].set_data([], [])

            if v.visual_segments and sim_t <= v.busy_until + 1e-9:
                pts = _flatten_route_points(v.visual_segments, cxy)
                if len(pts) >= 2:
                    route_lines[i].set_data([p[0] for p in pts], [p[1] for p in pts])
                else:
                    route_lines[i].set_data([], [])
            else:
                route_lines[i].set_data([], [])

        artists = [lc, pending_sc, assigned_sc, vehicle_sc]
        artists.extend(trail_lines)
        artists.extend(route_lines)
        artists.extend(battery_bg_rects)
        artists.extend(battery_fg_rects)
        return artists

    return {
        "fig": fig,
        "update": update,
        "n_frames": n_frames,
        "fps": fps,
        "summary": {
            "strategy": strategy,
            "scale_name": scale_name,
            "dt": dt,
            "steps_per_frame": steps_per_frame,
            "duration_limit": duration_limit,
            "sim": sim,
            "basemap_cache_hit": basemap_cache_hit,
        },
    }


def _summary_text(base: Dict[str, Any], n_frames: int, media: str, writer_used: str) -> str:
    sim = base["sim"]
    return (
        f"Done. strategy={base['strategy']}, scale={base['scale_name']}, dt={base['dt']}, "
        f"steps/frame={base['steps_per_frame']}, fps={int(base.get('fps', 0) or 0)}, "
        f"duration={base['duration_limit']:.1f}s, score={sim.score:.2f}, frames={n_frames}, "
        f"media={media}, writer={writer_used}, basemap_cache={'hit' if base['basemap_cache_hit'] else 'miss'}"
    )


def _render_score_curve_image(times: Sequence[float], scores: Sequence[float]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5.2, 3.1), dpi=120)
    fig.patch.set_facecolor("#0f1220")
    ax.set_facecolor("#151a2e")
    ax.plot(times, scores, color="#7aa2f7", linewidth=2.0)
    ax.set_xlabel("t", color="#a9b1d6")
    ax.set_ylabel("score", color="#a9b1d6")
    ax.tick_params(colors="#a9b1d6", labelsize=8)
    ax.grid(True, color="#2a3150", linewidth=0.6, alpha=0.8)
    for s in ax.spines.values():
        s.set_color("#3b4261")
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    img = np.asarray(rgba[..., :3]).copy()
    plt.close(fig)
    return img


def _render_final_score_frame(frame: np.ndarray, final_score: float) -> np.ndarray:
    h = int(frame.shape[0])
    w = int(frame.shape[1])
    dpi = 100
    fig, ax = plt.subplots(figsize=(max(1, w) / dpi, max(1, h) / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()
    cjk_font = _get_cjk_font_properties()
    title_size = max(34, int(min(w, h) * 0.10))
    score_size = max(46, int(min(w, h) * 0.16))
    ax.text(
        0.5,
        0.62,
        "最终分数",
        ha="center",
        va="center",
        color="white",
        fontsize=title_size,
        fontweight="bold",
        fontproperties=cjk_font,
    )
    ax.text(0.5, 0.42, f"{final_score:.2f}", ha="center", va="center", color="#7dcfff", fontsize=score_size, fontweight="bold")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    img = np.asarray(rgba[..., :3]).copy()
    plt.close(fig)
    return img


def stream_live_canvas_frames(
    fleet_osm_mod: Any,
    strategy: str,
    scale_name: str,
    seed: Optional[int],
    dt: float,
    duration_s: float,
    steps_per_frame: int,
    fps: int,
    use_osm_basemap: bool,
    show_battery: bool,
    segments_override: Optional[Sequence[Any]] = None,
):
    rt = _build_animation_runtime(
        fleet_osm_mod,
        strategy,
        scale_name,
        seed,
        dt,
        duration_s,
        steps_per_frame,
        fps,
        use_osm_basemap,
        show_battery,
        segments_override,
    )
    fig = rt["fig"]
    update = rt["update"]
    n_frames = int(rt["n_frames"])
    base = dict(rt["summary"])
    base["fps"] = fps
    canvas = FigureCanvasAgg(fig)
    score_times: List[float] = [0.0]
    score_values: List[float] = [float(base["sim"].score)]
    last = None
    for i in range(n_frames):
        update(i)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frame = np.asarray(rgba[..., :3]).copy()
        last = frame
        score_times.append(float((i + 1) * dt * max(1, steps_per_frame)))
        score_values.append(float(base["sim"].score))
        score_img = _render_score_curve_image(score_times, score_values)
        yield frame, score_img, _summary_text(base, i + 1, media="live", writer_used="canvas")
    plt.close(fig)
    if last is not None:
        score_img = _render_score_curve_image(score_times, score_values)
        final_score = float(base["sim"].score)
        final_frame = _render_final_score_frame(last, final_score)
        yield final_frame, score_img, _summary_text(base, n_frames, media="live", writer_used="canvas")


def render_animation(
    fleet_osm_mod: Any,
    strategy: str,
    scale_name: str,
    seed: Optional[int],
    dt: float,
    duration_s: float,
    steps_per_frame: int,
    fps: int,
    use_osm_basemap: bool,
    show_battery: bool,
    out_dir: Path,
    segments_override: Optional[Sequence[Any]] = None,
) -> Tuple[str, str, str]:
    strategy_slug = _ascii_safe(strategy)
    rt = _build_animation_runtime(
        fleet_osm_mod,
        strategy,
        scale_name,
        seed,
        dt,
        duration_s,
        steps_per_frame,
        fps,
        use_osm_basemap,
        show_battery,
        segments_override,
    )
    fig = rt["fig"]
    update = rt["update"]
    n_frames = int(rt["n_frames"])
    base = dict(rt["summary"])
    base["fps"] = fps
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / max(1, fps), blit=False, repeat=False)

    out_stem = f"osm_anim_{scale_name}_{strategy_slug}_{int(time.time())}"
    out_path, media_fmt, writer_used = _save_animation_fast(ani, out_dir, out_stem, fps)
    plt.close(fig)
    summary = _summary_text(base, n_frames, media=media_fmt, writer_used=writer_used)
    return str(out_path), media_fmt, summary


def _segments_from_osm_xml_file(mod: Any, path: Path) -> List[Any]:
    tree = ET.parse(str(path))
    root = tree.getroot()
    segment_cls = getattr(mod, "Segment", None)
    if segment_cls is None:
        raise RuntimeError("fleet_osm module missing Segment class")

    nodes: Dict[str, Tuple[float, float]] = {}
    for n in root.findall("node"):
        nid = n.attrib.get("id")
        lon = n.attrib.get("lon")
        lat = n.attrib.get("lat")
        if nid is None or lon is None or lat is None:
            continue
        nodes[nid] = (float(lon), float(lat))

    out: List[Any] = []
    for way in root.findall("way"):
        tags = {
            t.attrib.get("k", ""): t.attrib.get("v", "")
            for t in way.findall("tag")
            if "k" in t.attrib
        }
        hw = str(tags.get("highway", "") or "")
        if not hw:
            continue
        refs = [nd.attrib.get("ref") for nd in way.findall("nd")]
        refs = [r for r in refs if r]
        for a, b in zip(refs, refs[1:]):
            pa = nodes.get(str(a))
            pb = nodes.get(str(b))
            if pa is None or pb is None or pa == pb:
                continue
            out.append(segment_cls(pa[0], pa[1], pb[0], pb[1], hw))
    return out


def _load_uploaded_segments(mod: Any, upload_path: Optional[str]) -> Optional[List[Any]]:
    if not upload_path:
        return None
    p = Path(upload_path)
    ext = p.suffix.lower()
    if ext == ".csv":
        return mod.load_segments_csv(str(p))
    if ext == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return mod.segments_from_osm_json(data)
    if ext in {".osm", ".xml"}:
        return _segments_from_osm_xml_file(mod, p)
    raise ValueError("Unsupported map file; use .csv, .json, .osm, or .xml")


def build_gradio_app(fleet_osm_mod: Any, fleet_osm_path: Optional[str]):
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "gradio is not installed. Please run: pip install gradio"
        ) from exc

    strategies = list(fleet_osm_mod.OSM_SIM_BUILDERS.keys())
    scales = [p.name for p in fleet_osm_mod.osm_presets_for_run(None)]
    run_controls: Dict[str, Dict[str, Any]] = {}

    def _ensure_control(session_key: str) -> Tuple[str, Dict[str, Any]]:
        key = str(session_key or "").strip() or f"sess_{time.time_ns()}"
        ctl = run_controls.get(key)
        if ctl is None:
            ctl = {
                "run_token": 0,
                "paused": False,
                "restart": False,
            }
            run_controls[key] = ctl
        return key, ctl

    def _run(
        strategy: str,
        scale_name: str,
        seed_val: int,
        dt: float,
        duration_s: float,
        steps_per_frame: int,
        fps: int,
        use_osm_basemap: bool,
        show_battery: bool,
        export_mp4: bool,
        uploaded_map_file: Optional[str],
        session_key: str,
    ):
        session_key, ctl = _ensure_control(session_key)
        ctl["run_token"] = int(ctl.get("run_token", 0)) + 1
        token = int(ctl["run_token"])
        ctl["paused"] = False
        ctl["restart"] = False

        seed = None if seed_val < 0 else int(seed_val)
        extra = f"fleet_osm source: {fleet_osm_path or 'import path/auto-detected'}"

        try:
            segments_override = _load_uploaded_segments(fleet_osm_mod, uploaded_map_file)
        except Exception as exc:
            err = f"上传地图读取失败: {exc}\n{extra}"
            yield None, None, None, err, session_key
            return

        if segments_override is not None:
            extra = f"{extra}\ncustom map: {Path(uploaded_map_file).name} ({len(segments_override)} segments)"

        def _new_stream():
            return stream_live_canvas_frames(
                fleet_osm_mod=fleet_osm_mod,
                strategy=strategy,
                scale_name=scale_name,
                seed=seed,
                dt=float(dt),
                duration_s=float(duration_s),
                steps_per_frame=int(steps_per_frame),
                fps=int(fps),
                use_osm_basemap=bool(use_osm_basemap),
                show_battery=bool(show_battery),
                segments_override=segments_override,
            )

        last_frame = None
        last_score_img = None
        live_summary = ""
        frame_stream = _new_stream()

        while True:
            if int(ctl.get("run_token", -1)) != token:
                break
            if bool(ctl.get("restart", False)):
                ctl["restart"] = False
                frame_stream = _new_stream()
                live_summary = "Restart requested: simulation restarted from t=0"
            if bool(ctl.get("paused", False)):
                pause_status = live_summary or "Paused"
                if last_frame is not None and last_score_img is not None:
                    yield (
                        last_frame,
                        last_score_img,
                        None,
                        pause_status + "\nPaused...\n" + extra,
                        session_key,
                    )
                time.sleep(0.15)
                continue

            try:
                frame, score_img, status = next(frame_stream)
            except StopIteration:
                break

            last_frame = frame
            last_score_img = score_img
            live_summary = status
            yield last_frame, last_score_img, None, status + "\n" + extra, session_key

        if int(ctl.get("run_token", -1)) != token:
            return

        if bool(export_mp4):
            try:
                media_path, _media_fmt, save_summary = render_animation(
                    fleet_osm_mod=fleet_osm_mod,
                    strategy=strategy,
                    scale_name=scale_name,
                    seed=seed,
                    dt=float(dt),
                    duration_s=float(duration_s),
                    steps_per_frame=int(steps_per_frame),
                    fps=int(fps),
                    use_osm_basemap=bool(use_osm_basemap),
                    show_battery=bool(show_battery),
                    out_dir=DEFAULT_OUTPUT_DIR,
                    segments_override=segments_override,
                )
                fin = live_summary + "\n" + save_summary + "\n" + extra
                yield last_frame, last_score_img, media_path, fin, session_key
            except Exception as exc:
                err = live_summary + f"\nMP4 export failed: {exc}" + "\n" + extra
                yield last_frame, last_score_img, None, err, session_key
        else:
            fin = live_summary + "\n实时演示完成（未导出MP4）" + "\n" + extra
            yield last_frame, last_score_img, None, fin, session_key

    def _toggle_pause(session_key: str):
        session_key, ctl = _ensure_control(session_key)
        ctl["paused"] = not bool(ctl.get("paused", False))
        state_text = "已暂停" if ctl["paused"] else "已继续播放"
        return f"{state_text}（当前运行ID: {ctl.get('run_token', 0)}）", session_key

    def _restart_live(session_key: str):
        session_key, ctl = _ensure_control(session_key)
        ctl["restart"] = True
        ctl["paused"] = False
        return f"已请求重新开始（当前运行ID: {ctl.get('run_token', 0)}）", session_key

    with gr.Blocks(title="智慧物流运输模拟") as demo:
        gr.Markdown("## 智慧物流运输模拟")
        gr.Markdown(
            "选择策略与规模后点击生成。动画使用 OSM 路网几何与 `fleet_osm.py` 同步的仿真步进逻辑。"
        )

        with gr.Row():
            strategy = gr.Dropdown(choices=strategies, value=strategies[0], label="策略")
            scale_name = gr.Dropdown(choices=scales, value=scales[0], label="规模")
            seed_val = gr.Number(value=-1, precision=0, label="主种子(-1=使用内置21/22/23)")

        with gr.Row():
            dt = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="仿真步长 dt")
            duration_s = gr.Slider(10, 1800, value=240, step=10, label="动画仿真总时长(秒)")
            steps_per_frame = gr.Slider(1, 8, value=2, step=1, label="每帧推进步数")
            fps = gr.Slider(4, 24, value=10, step=1, label="输出 FPS")

        use_osm_basemap = gr.Checkbox(value=False, label="尝试加载在线OSM底图(contextily)")
        show_battery = gr.Checkbox(value=False, label="显示车辆电量条")
        export_mp4 = gr.Checkbox(value=False, label="演示完成后导出MP4")
        uploaded_map_file = gr.File(
            label="拖拽上传OSM地图文件（可选：.csv/.json/.osm/.xml）",
            type="filepath",
            file_types=[".csv", ".json", ".osm", ".xml"],
        )

        with gr.Row():
            btn = gr.Button("生成动画", variant="primary")
            pause_btn = gr.Button("暂停/继续")
            restart_btn = gr.Button("重新开始")

        with gr.Row():
            preview_canvas = gr.Image(type="numpy", label="实时画布")
            score_curve = gr.Image(type="numpy", label="实时分数折线图")

        download = gr.File(label="下载文件")

        summary = gr.Textbox(label="运行信息", lines=4, visible=False)
        session_key = gr.State(value="")

        btn.click(
            _run,
            inputs=[
                strategy,
                scale_name,
                seed_val,
                dt,
                duration_s,
                steps_per_frame,
                fps,
                use_osm_basemap,
                show_battery,
                export_mp4,
                uploaded_map_file,
                session_key,
            ],
            outputs=[preview_canvas, score_curve, download, summary, session_key],
        )

        pause_btn.click(
            _toggle_pause,
            inputs=[session_key],
            outputs=[summary, session_key],
            queue=False,
        )

        restart_btn.click(
            _restart_live,
            inputs=[session_key],
            outputs=[summary, session_key],
            queue=False,
        )

    return demo


def _self_test(fleet_osm_mod: Any) -> str:
    media_path, _media_fmt, summary = render_animation(
        fleet_osm_mod=fleet_osm_mod,
        strategy=list(fleet_osm_mod.OSM_SIM_BUILDERS.keys())[0],
        scale_name=fleet_osm_mod.osm_presets_for_run(None)[0].name,
        seed=None,
        dt=0.5,
        duration_s=20.0,
        steps_per_frame=3,
        fps=8,
        use_osm_basemap=False,
        show_battery=False,
        out_dir=DEFAULT_OUTPUT_DIR,
    )
    return f"{summary}\nfile={media_path}"


def main() -> int:
    ap = argparse.ArgumentParser(description="OSM Fleet Gradio animation")
    ap.add_argument("--fleet-osm-path", default=None, help="Absolute path to fleet_osm.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="Render a tiny GIF and exit")
    args = ap.parse_args()

    fleet_osm_mod = _load_fleet_osm_module(args.fleet_osm_path)

    if args.self_test:
        print(_self_test(fleet_osm_mod))
        return 0

    demo = build_gradio_app(fleet_osm_mod, args.fleet_osm_path)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







