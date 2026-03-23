#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _read_txt_matrix(path: Path) -> np.ndarray:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            data.append([float(x) for x in parts])
    arr = np.array(data, dtype=np.float64)
    return arr


def read_K(path: Path) -> np.ndarray:
    K = _read_txt_matrix(path)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K in {path}, got {K.shape}")
    return K


def read_T_cam_obj(path: Path) -> np.ndarray:
    T = _read_txt_matrix(path)
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :] = T
        return T4
    raise ValueError(f"Expected 4x4 (or 3x4) pose matrix in {path}, got {T.shape}")


def project_points(K: np.ndarray, X_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X_cam.ndim != 2 or X_cam.shape[1] != 3:
        raise ValueError(f"X_cam must be Nx3, got {X_cam.shape}")

    z = X_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((X_cam.shape[0], 2), np.nan, dtype=np.float64)
    if not np.any(valid):
        return uv, valid

    x = X_cam[valid, 0] / z[valid]
    y = X_cam[valid, 1] / z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x + cx
    v = fy * y + cy
    uv[valid, 0] = u
    uv[valid, 1] = v
    return uv, valid


def _maybe_load_cv2():
    try:
        import cv2

        return cv2
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) is required. Install with: pip install opencv-python"
        ) from e


def draw_axes(img: np.ndarray, K: np.ndarray, T_cam_obj: np.ndarray, axis_len: float, thickness: int) -> np.ndarray:
    cv2 = _maybe_load_cv2()

    R = T_cam_obj[:3, :3]
    t = T_cam_obj[:3, 3]

    X_obj = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float64,
    )

    X_cam = (R @ X_obj.T).T + t[None, :]
    uv, valid = project_points(K, X_cam)

    out = img.copy()

    if not valid[0]:
        return out

    o = tuple(np.round(uv[0]).astype(int).tolist())

    def _line(i: int, color_bgr: Tuple[int, int, int]):
        if not valid[i]:
            return
        p = tuple(np.round(uv[i]).astype(int).tolist())
        cv2.line(out, o, p, color_bgr, thickness, lineType=cv2.LINE_AA)

    _line(1, (0, 0, 255))
    _line(2, (0, 255, 0))
    _line(3, (255, 0, 0))

    cv2.circle(out, o, max(2, thickness + 1), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    return out


def find_frame_ids(pose_dir: Path, rgb_dir: Path) -> list[str]:
    pose_ids = {p.stem for p in pose_dir.glob("*.txt")}
    rgb_ids = {p.stem for p in rgb_dir.glob("*.png")}
    ids = sorted(pose_ids & rgb_ids)
    return ids


def main():
    ap = argparse.ArgumentParser(
        description="Overlay FoundationPose (object-in-camera) poses on RGB frames by drawing 3D axes."
    )
    ap.add_argument("--pose_dir", type=Path, default=Path("debug/ob_in_cam"))
    ap.add_argument("--rgb_dir", type=Path, default=Path("foundation_pose_data/rgb"))
    ap.add_argument("--K", type=Path, default=Path("foundation_pose_data/cam_K.txt"))
    ap.add_argument("--axis_len", type=float, default=0.05, help="Axis length in the same units as translation (usually meters).")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N frames.")
    ap.add_argument("--delay", type=int, default=1, help="Delay (ms) between frames. Use 0 to step frame-by-frame.")
    ap.add_argument("--scale", type=float, default=1.5, help="Window size scale factor relative to the image size.")
    ap.add_argument("--win_w", type=int, default=0, help="Optional window width override (pixels).")
    ap.add_argument("--win_h", type=int, default=0, help="Optional window height override (pixels).")

    args = ap.parse_args()

    cv2 = _maybe_load_cv2()

    pose_dir: Path = args.pose_dir
    rgb_dir: Path = args.rgb_dir

    K = read_K(args.K)

    ids = find_frame_ids(pose_dir, rgb_dir)
    if not ids:
        raise RuntimeError(
            f"No matching frames found. pose_dir has {len(list(pose_dir.glob('*.txt')))} txt, "
            f"rgb_dir has {len(list(rgb_dir.glob('*.png')))} png."
        )

    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    window = "pose_on_rgb"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    window_initialized = False

    for stem in ids:
        pose_path = pose_dir / f"{stem}.txt"
        img_path = rgb_dir / f"{stem}.png"

        T = read_T_cam_obj(pose_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        if not window_initialized:
            h, w = img.shape[:2]
            if args.win_w > 0 and args.win_h > 0:
                cv2.resizeWindow(window, int(args.win_w), int(args.win_h))
            else:
                scale = float(args.scale) if args.scale and args.scale > 0 else 1.0
                cv2.resizeWindow(window, int(w * scale), int(h * scale))
            window_initialized = True

        vis = draw_axes(img, K, T, axis_len=args.axis_len, thickness=args.thickness)
        cv2.imshow(window, vis)

        key = cv2.waitKey(int(args.delay)) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
