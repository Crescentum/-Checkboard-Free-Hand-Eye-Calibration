#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def rotation_geodesic_distance(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_a = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    R_rel = R_a @ R_b.T
    cos_theta = (np.trace(R_rel) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _maybe_load_cv2():
    try:
        import cv2 

        return cv2
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) is required for calibrateHandEye. Install with: pip install opencv-python"
        ) from e


def load_T_dir(directory: Path) -> Dict[str, np.ndarray]:
    mats: Dict[str, np.ndarray] = {}
    for p in sorted(directory.glob("*.txt")):
        T = np.loadtxt(p).reshape(4, 4).astype(np.float64)
        mats[p.stem] = T
    return mats


def load_T_json(filepath: Path) -> Dict[str, np.ndarray]:
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "poses" not in data:
        raise RuntimeError(f"JSON missing key 'poses': {filepath}")

    poses = data["poses"]
    if not isinstance(poses, list) or not poses:
        raise RuntimeError(f"JSON 'poses' must be a non-empty list: {filepath}")

    mats: Dict[str, np.ndarray] = {}
    for i, pose in enumerate(poses):
        T = np.asarray(pose, dtype=np.float64)
        if T.shape != (4, 4):
            raise RuntimeError(f"Pose {i} has shape {T.shape}, expected (4,4) in {filepath}")
        mats[f"{i:06d}"] = T
    return mats


def intersect_and_sort_keys(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> List[str]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        raise RuntimeError("No matching frame ids between the two directories.")
    return keys


def split_R_t(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64).reshape(3, 1)
    return R, t


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Hand-eye calibration via OpenCV calibrateHandEye. "
            "Default inputs are JSON: robot_poses.json provides T_base_ee; "
            "camera_poses_scaled.json provides T_base_cam and is inverted to target2cam=T_cam_base (treating target=base). "
            "(You can still use the legacy .txt directories via --ee_pose_dir/--cam_pose_dir.)"
        )
    )
    ap.add_argument("--robot_poses", type=Path, default=Path("robot_poses_selected.json"), help="JSON with key 'poses' of T_base_ee")
    ap.add_argument(
        "--camera_poses",
        type=Path,
        default=Path("camera_poses_scaled_selected.json"),
        help="JSON with key 'poses' of T_base_cam (will be inverted to T_cam_base)",
    )
    ap.add_argument(
        "--ee_pose_dir",
        type=Path,
        default=Path("foundation_pose_data/ee_pose"),
        help="Legacy: directory of *.txt containing T_base_ee",
    )
    ap.add_argument(
        "--cam_pose_dir",
        type=Path,
        default=Path("debug/ob_in_cam"),
        help="Legacy: directory of *.txt containing T_cam_obj (target/object -> camera)",
    )
    ap.add_argument("--method", type=str, default="ANDREFF", help="TSAI|PARK|HORAUD|ANDREFF|DANIILIDIS")
    ap.add_argument("--limit", type=int, default=0, help="If >0, use only first N matched frames.")
    args = ap.parse_args()

    cv2 = _maybe_load_cv2()

    if args.robot_poses.exists() and args.camera_poses.exists():
        ee_mats = load_T_json(args.robot_poses)  
        cam_base_mats = load_T_json(args.camera_poses) 
        cam_mats = {k: T_base_cam for k, T_base_cam in cam_base_mats.items()} 
    else:
        ee_mats = load_T_dir(args.ee_pose_dir)  
        cam_mats = load_T_dir(args.cam_pose_dir)  

    keys = intersect_and_sort_keys(ee_mats, cam_mats)
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    R_gripper2base: List[np.ndarray] = []
    t_gripper2base: List[np.ndarray] = []
    R_target2cam: List[np.ndarray] = []
    t_target2cam: List[np.ndarray] = []

    for k in keys:
        T_base_ee = ee_mats[k]
        T_target2cam = cam_mats[k]

        Rg, tg = split_R_t(T_base_ee)
        Rt, tt = split_R_t(T_target2cam)

        R_gripper2base.append(Rg)
        t_gripper2base.append(tg)
        R_target2cam.append(Rt)
        t_target2cam.append(tt)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_ANDREFF,
    )

    T_cam_ee = make_T(R_cam2gripper, t_cam2gripper)  
    T_ee_cam = inv_T(T_cam_ee) 
    
    np.set_printoptions(suppress=True, precision=6)
    print(f"Used {len(keys)} frames")
    print("\nOpenCV output (T_cam_ee):")
    print(T_cam_ee)

    truth_path = Path("foundation_pose_data/T_ee_cam_truth.txt")
    if truth_path.exists():
        X_gt = np.loadtxt(truth_path).reshape(4, 4)
        print("\nTruth (T_cam_ee_gt):")
        print(X_gt)
        print(f"Translation error (cm): {np.linalg.norm(T_cam_ee[:3, 3] - X_gt[:3, 3]) * 100:.6f}")
        rot_err_rad = rotation_geodesic_distance(T_cam_ee[:3, :3], X_gt[:3, :3])
        print(f"Rotation Error (deg): {np.degrees(rot_err_rad):.6f}")


if __name__ == "__main__":
    main()
