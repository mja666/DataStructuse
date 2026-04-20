#!/usr/bin/env python3
"""Dependency checker and one-click installer for osm_gradio_visual.py.

Usage examples:
  python dependency_doctor.py check
  python dependency_doctor.py install --include-optional
  python dependency_doctor.py fix --include-optional --check-ffmpeg
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore


@dataclass(frozen=True)
class DepSpec:
    import_name: str
    pip_name: str
    required: bool
    note: str


DEPENDENCIES: List[DepSpec] = [
    DepSpec("numpy", "numpy", True, "core plotting/math"),
    DepSpec("matplotlib", "matplotlib", True, "animation and rendering"),
    DepSpec("gradio", "gradio", True, "web UI"),
    DepSpec("contextily", "contextily", False, "online basemap tiles (optional)"),
    DepSpec("pyproj", "pyproj", False, "lat/lon -> EPSG:3857 projection (optional)"),
]


@dataclass
class DepStatus:
    spec: DepSpec
    ok: bool
    version: str
    error: str


def _check_one(spec: DepSpec) -> DepStatus:
    try:
        found = importlib.util.find_spec(spec.import_name) is not None
    except Exception as exc:
        return DepStatus(spec=spec, ok=False, version="", error=str(exc))

    if not found:
        return DepStatus(spec=spec, ok=False, version="", error="module not found")

    version = "unknown"
    try:
        version = str(importlib_metadata.version(spec.pip_name))
    except Exception:
        try:
            version = str(importlib_metadata.version(spec.import_name))
        except Exception:
            version = "unknown"

    return DepStatus(spec=spec, ok=True, version=version, error="")


def check_dependencies(include_optional: bool) -> List[DepStatus]:
    results: List[DepStatus] = []
    for spec in DEPENDENCIES:
        if not include_optional and not spec.required:
            continue
        results.append(_check_one(spec))
    return results


def print_report(statuses: Sequence[DepStatus], check_ffmpeg: bool) -> None:
    print("=" * 78)
    print("Dependency Check Report")
    print("=" * 78)
    for st in statuses:
        level = "REQUIRED" if st.spec.required else "OPTIONAL"
        state = "OK" if st.ok else "MISSING"
        print(f"[{state:<7}] [{level}] {st.spec.import_name:<12} ({st.spec.note})")
        if st.ok:
            print(f"          version: {st.version}")
        else:
            print(f"          error:   {st.error}")

    if check_ffmpeg:
        try:
            from matplotlib.animation import writers  # imported lazily

            ff_ok = writers.is_available("ffmpeg")
        except Exception as exc:
            ff_ok = False
            print(f"[MISSING] [OPTIONAL] ffmpeg       (mp4 export)\n          error:   {exc}")
        else:
            state = "OK" if ff_ok else "MISSING"
            tail = "found" if ff_ok else "not found in PATH"
            print(f"[{state:<7}] [OPTIONAL] ffmpeg       (mp4 export)")
            print(f"          detail:  {tail}")

    print("-" * 78)
    required_missing = [s for s in statuses if s.spec.required and not s.ok]
    optional_missing = [s for s in statuses if (not s.spec.required) and (not s.ok)]
    print(
        f"Required missing: {len(required_missing)} | Optional missing: {len(optional_missing)}"
    )


def install_missing(
    statuses: Sequence[DepStatus],
    include_optional: bool,
    python_exec: Optional[str],
    dry_run: bool,
    upgrade: bool,
) -> int:
    missing_required = [s.spec.pip_name for s in statuses if s.spec.required and not s.ok]
    missing_optional = [
        s.spec.pip_name
        for s in statuses
        if (not s.spec.required) and (not s.ok) and include_optional
    ]
    to_install = missing_required + missing_optional

    # Keep order while deduplicating.
    seen = set()
    ordered_install: List[str] = []
    for pkg in to_install:
        if pkg in seen:
            continue
        seen.add(pkg)
        ordered_install.append(pkg)

    if not ordered_install:
        print("No missing packages to install.")
        return 0

    py = python_exec or sys.executable
    cmd = [py, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(ordered_install)

    print("Installing packages:", ", ".join(ordered_install))
    print("Command:", " ".join(cmd))

    if dry_run:
        print("Dry run enabled; installation not executed.")
        return 0

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(f"pip install failed with code {proc.returncode}")
        return proc.returncode
    print("Installation completed.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check project dependencies and optionally install missing packages."
    )
    sub = ap.add_subparsers(dest="action", required=False)

    ap.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional libraries (contextily/pyproj) in check/install.",
    )
    ap.add_argument(
        "--check-ffmpeg",
        action="store_true",
        help="Also report whether ffmpeg is available for MP4 export.",
    )

    p_install = sub.add_parser("install", help="Install missing packages.")
    p_install.add_argument("--dry-run", action="store_true", help="Print pip command only.")
    p_install.add_argument(
        "--python",
        default=None,
        help="Python executable used for pip install (default: current interpreter).",
    )
    p_install.add_argument("--upgrade", action="store_true", help="Install with --upgrade.")

    p_fix = sub.add_parser("fix", help="Check, install missing, then re-check.")
    p_fix.add_argument("--dry-run", action="store_true", help="Print pip command only.")
    p_fix.add_argument(
        "--python",
        default=None,
        help="Python executable used for pip install (default: current interpreter).",
    )
    p_fix.add_argument("--upgrade", action="store_true", help="Install with --upgrade.")

    sub.add_parser("check", help="Check dependencies only (default).")

    args = ap.parse_args()
    action = args.action or "check"

    statuses = check_dependencies(include_optional=bool(args.include_optional))
    print_report(statuses, check_ffmpeg=bool(args.check_ffmpeg))

    required_missing = any((not s.ok) and s.spec.required for s in statuses)

    if action == "check":
        return 1 if required_missing else 0

    if action == "install":
        return install_missing(
            statuses=statuses,
            include_optional=bool(args.include_optional),
            python_exec=getattr(args, "python", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            upgrade=bool(getattr(args, "upgrade", False)),
        )

    if action == "fix":
        code = install_missing(
            statuses=statuses,
            include_optional=bool(args.include_optional),
            python_exec=getattr(args, "python", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            upgrade=bool(getattr(args, "upgrade", False)),
        )
        if code != 0:
            return code

        print("\nRe-check after installation:\n")
        statuses2 = check_dependencies(include_optional=bool(args.include_optional))
        print_report(statuses2, check_ffmpeg=bool(args.check_ffmpeg))
        required_missing_2 = any((not s.ok) and s.spec.required for s in statuses2)
        return 1 if required_missing_2 else 0

    print(f"Unknown action: {action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
