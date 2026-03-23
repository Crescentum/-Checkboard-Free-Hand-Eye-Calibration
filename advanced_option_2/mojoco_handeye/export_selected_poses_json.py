#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import List


def _read_T_4x4_txt(path: Path) -> List[List[float]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != 4:
        raise ValueError(f"Expected 4 non-empty lines in {path}, got {len(lines)}")

    mat: List[List[float]] = []
    for row_i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) != 4:
            raise ValueError(f"Expected 4 columns in {path} line {row_i+1}, got {len(parts)}")
        mat.append([float(x) for x in parts])

    return mat


def _parse_select(select: str) -> List[str]:
    if not select or not select.strip():
        raise ValueError("--select is required")

    out: List[str] = []
    seen = set()
    for raw in select.split(","):
        token = raw.strip()
        if not token:
            continue

        if token.endswith(".txt"):
            token = token[:-4]

        if "-" in token:
            a, b = token.split("-", 1)
            a = a.strip()
            b = b.strip()
            if a.isdigit() and b.isdigit():
                start = int(a)
                end = int(b)
                step = 1 if end >= start else -1
                for idx in range(start, end + step, step):
                    stem = f"{idx:06d}"
                    if stem in seen:
                        raise ValueError(f"Duplicate selection: {stem}")
                    out.append(stem)
                    seen.add(stem)
                continue

        if token.isdigit():
            stem = f"{int(token):06d}"
        else:
            stem = token

        if stem in seen:
            raise ValueError(f"Duplicate selection: {stem}")
        out.append(stem)
        seen.add(stem)

    if not out:
        raise ValueError("Empty selection after parsing --select")

    return out


def _list_txt_stems(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        raise SystemExit(f"Directory does not exist: {dir_path}")
    stems = [p.stem for p in dir_path.glob("*.txt") if p.is_file()]
    if not stems:
        raise SystemExit(f"No *.txt files found in: {dir_path}")
    return stems


def _sort_frame_ids(frame_ids: List[str]) -> List[str]:
    def key(stem: str):
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    return sorted(frame_ids, key=key)


def _select_evenly(frame_ids: List[str], n: int) -> List[str]:
    if n <= 0:
        raise ValueError("--select must be a positive integer when using count mode")
    if not frame_ids:
        raise ValueError("No available frames to select from")
    m = len(frame_ids)
    if n > m:
        raise SystemExit(f"Requested --select={n} frames, but only {m} matched frames are available")
    if n == 1:
        return [frame_ids[(m - 1) // 2]]

    out: List[str] = []
    for k in range(n):
        idx = (k * (m - 1)) // (n - 1)
        out.append(frame_ids[idx])
    return out


def _is_explicit_list_select(select: str) -> bool:
    s = select.strip()
    return ("," in s) or ("-" in s) or (".txt" in s)


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Select N frames from debug/ob_in_cam and export two JSON files: "
            "one in the same schema as camera_poses_scaled.json, and the matching ee poses "
            "in the same schema as robot_poses.json. Matching is by frame id (filename stem)."
        )
    )
    ap.add_argument(
        "--ob_dir",
        type=Path,
        default=Path("debug/ob_in_cam"),
        help="Directory of *.txt (e.g., debug/ob_in_cam) containing 4x4 transforms",
    )
    ap.add_argument(
        "--ee_dir",
        type=Path,
        default=Path("foundation_pose_data/ee_pose"),
        help="Directory of *.txt (e.g., foundation_pose_data/ee_pose) containing 4x4 transforms",
    )
    ap.add_argument(
        "--select",
        type=str,
        default="15",
        help=(
            "Selection. If a single integer N is provided, select N pose pairs evenly spaced "
            "from the dataset intersection of --ob_dir and --ee_dir. If an explicit list is provided, "
            "use comma-separated ids/ranges, e.g. '0,5,10' or '000010,000020' or '10-24'. "
            "To select a single specific id, use '000015.txt' (or include a comma)."
        ),
    )
    ap.add_argument(
        "--require",
        type=int,
        default=15,
        # default=310,
        help="Require exactly this many selected frames (default: 15). Set to 0 to disable.",
    )
    ap.add_argument(
        "--out_camera",
        type=Path,
        default=Path("camera_poses_scaled_selected.json"),
        help="Output JSON path for selected poses (camera_poses_scaled schema)",
    )
    ap.add_argument(
        "--out_robot",
        type=Path,
        default=Path("robot_poses_selected.json"),
        help="Output JSON path for selected ee poses (robot_poses schema)",
    )

    args = ap.parse_args()

    select_str = args.select.strip()
    if _is_explicit_list_select(select_str):
        frame_ids = _parse_select(select_str)
    else:
        if not re.fullmatch(r"\d+", select_str):
            raise SystemExit(
                "Invalid --select. Provide either a single integer N (count mode), or an explicit list/range."
            )
        n = int(select_str)
        ob_stems = set(_list_txt_stems(args.ob_dir))
        ee_stems = set(_list_txt_stems(args.ee_dir))
        common = _sort_frame_ids(list(ob_stems & ee_stems))
        if not common:
            raise SystemExit(
                f"No common frame ids found between {args.ob_dir} and {args.ee_dir} (by *.txt stem)"
            )
        frame_ids = _select_evenly(common, n)

    if args.require and args.require > 0 and len(frame_ids) != args.require:
        raise SystemExit(f"Selection has {len(frame_ids)} frames, but --require={args.require}")

    cam_poses: List[List[List[float]]] = []
    ee_poses: List[List[List[float]]] = []

    for stem in frame_ids:
        cam_path = args.ob_dir / f"{stem}.txt"
        ee_path = args.ee_dir / f"{stem}.txt"
        if not cam_path.exists():
            raise SystemExit(f"Missing camera pose file: {cam_path}")
        if not ee_path.exists():
            raise SystemExit(f"Missing ee pose file: {ee_path}")

        cam_poses.append(_read_T_4x4_txt(cam_path))
        ee_poses.append(_read_T_4x4_txt(ee_path))

    camera_payload = {
        "description": "Selected poses exported from debug/ob_in_cam",
        "data_type": "INPUT",
        "source": str(args.ob_dir),
        "coordinate_frame": "camera",
        "format": "4x4 homogeneous transformation matrices",
        "count": len(cam_poses),
        "usage": "Pairs with robot poses by index order",
        "frame_ids": frame_ids,
        "poses": cam_poses,
    }

    robot_payload = {
        "description": "Selected robot end-effector poses exported from foundation_pose_data/ee_pose",
        "data_type": "INPUT",
        "coordinate_frame": "robot_base",
        "format": "4x4 homogeneous transformation matrices",
        "count": len(ee_poses),
        "usage": "Input A for hand-eye calibration; pairs with camera poses by index order",
        "frame_ids": frame_ids,
        "poses": ee_poses,
    }

    _dump_json(args.out_camera, camera_payload)
    _dump_json(args.out_robot, robot_payload)

    print(f"Wrote: {args.out_camera} ({len(cam_poses)} poses)")
    print(f"Wrote: {args.out_robot} ({len(ee_poses)} poses)")


if __name__ == "__main__":
    main()
