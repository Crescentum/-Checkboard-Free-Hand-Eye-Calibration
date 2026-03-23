"""
Complete Experiment Pipeline for Robust Hand-Eye Calibration
Uses ORIGINAL Daniilidis dual quaternion method for Park-Martin
Uses ORIGINAL Tsai-Lenz implementation

Usage:
    python experiment_integrated.py
"""

import os
import sys
import json
import numpy as np
import torch
import argparse
import cv2
from datetime import datetime
from scipy.optimize import minimize_scalar

project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_folder)

from ransac_hand_eye import (
    ransac_hand_eye,
    RANSACConfig, 
    MEstimatorType,
    compute_all_errors,
    get_weight_function,
    compute_mad,
    solve_rotation_svd
)

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# =============================================================================
# Data Loading (PTH format)
# =============================================================================

def load_raw_camera_poses(filepath):
    data = torch.load(filepath, map_location='cpu')
    if isinstance(data, dict) and 'poses' in data:
        camera_poses = data['poses'].numpy() if isinstance(data['poses'], torch.Tensor) else data['poses']
    else:
        camera_poses = data.numpy() if isinstance(data, torch.Tensor) else data
    return camera_poses


def load_robot_poses(filepath):
    data = torch.load(filepath, map_location='cpu')
    robot_poses = data['eef_poses'].numpy() if isinstance(data['eef_poses'], torch.Tensor) else data['eef_poses']
    return robot_poses


def calculate_relative_transforms(poses):
    rels = []
    for i in range(len(poses) - 1):
        T1 = poses[i]
        T2 = poses[i + 1]
        T_rel = np.dot(np.linalg.inv(T1), T2)
        rels.append(T_rel)
    return rels


# =============================================================================
# Tsai-Lenz
# =============================================================================

def tsai_lenz_scaled(A_list, B_list, scale_range=(0.01, 10.0), num_scales=20000):

    n = len(A_list)
    
    M = np.zeros((3, 3))
    
    for i in range(n):
        Ra = A_list[i][:3, :3]
        Rb = B_list[i][:3, :3]
        
        alpha = cv2.Rodrigues(Ra)[0].flatten()
        beta = cv2.Rodrigues(Rb)[0].flatten()
        
        M += np.outer(beta, alpha)
        
    U, S, Vt = np.linalg.svd(M)
    Rx = np.dot(Vt.T, U.T)
    
    if np.linalg.det(Rx) < 0:
        Rx = -Rx
    
    vals_sc = np.linspace(scale_range[0], scale_range[1], num_scales)
    best_cost = float('inf')
    best_scale = 1.0
    best_tx = np.zeros(3)
    
    for scale in vals_sc:
        C = []
        d = []
        
        for i in range(n):
            Ra = A_list[i][:3, :3]
            ta = A_list[i][:3, 3].reshape(3, 1)
            tb = B_list[i][:3, 3].reshape(3, 1)
            
            I = np.eye(3)
            term1 = scale * np.dot(Rx, tb)
            term2 = I - Ra
            
            C.append(term2)
            d.append(ta - term1)
        
        C = np.vstack(C)
        d = np.vstack(d)
        
        tx, residuals, rank, s_vals = np.linalg.lstsq(C, d, rcond=None)

        cost = 0
        for i in range(n):
            err = A_list[i][:3, 3] - (scale * np.dot(Rx, B_list[i][:3, 3]) + 
                                      np.dot((np.eye(3) - A_list[i][:3, :3]), tx.flatten()))
            cost += np.linalg.norm(err) ** 2
        
        if cost < best_cost:
            best_cost = cost
            best_scale = scale
            best_tx = tx
    
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = best_tx.flatten()
    
    return best_scale, X


# =============================================================================
# Park-Martin
# =============================================================================

def homogeneous_inverse(T):
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    Rt = R.T
    t_inv = -Rt @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = Rt
    T_inv[:3, 3] = t_inv.flatten()
    return T_inv


def rot2quat(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def quat2rot(q):
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R


def skew(v):
    vx, vy, vz = v.flatten()
    return np.array([[  0, -vz,  vy],
                     [ vz,   0, -vx],
                     [-vy,  vx,   0]])


def qmult(s, t):

    s0, s1, s2, s3 = s.flatten()
    t0, t1, t2, t3 = t.flatten()
    q = np.array([
        s0*t0 - s1*t1 - s2*t2 - s3*t3,
        s0*t1 + s1*t0 + s2*t3 - s3*t2,
        s0*t2 - s1*t3 + s2*t0 + s3*t1,
        s0*t3 + s1*t2 - s2*t1 + s3*t0
    ])
    return q.reshape(4, 1)


def homogeneous2dualQuaternion(H):
    R = H[:3, :3]
    t = H[:3, 3].reshape(3, 1)
    q = rot2quat(R)
    qt = np.zeros((4, 1))
    qt[1:, 0] = t.flatten()
    q_prime = 0.5 * qmult(qt, q.reshape(4, 1))
    dq = np.vstack([q.reshape(4, 1), q_prime])
    return dq


def dualQuaternion2homogeneous(dq):

    q = dq[:4].flatten()
    q_prime = dq[4:].flatten()

    R = quat2rot(q)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    qt = 2 * qmult(q_prime.reshape(4, 1), q_conj.reshape(4, 1))
    t = qt[1:].flatten()

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def calibrate_hand_eye_daniilidis(R_gripper2base, t_gripper2base,
                                  R_target2cam, t_target2cam):
    def to_homogeneous(R, t):
        if R.shape == (3, 1) or R.shape == (1, 3) or R.size == 3:
            R = cv2.Rodrigues(R)[0]
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t.flatten()
        return H.astype(np.float64)

    Hg = [to_homogeneous(R, t) for R, t in zip(R_gripper2base, t_gripper2base)]
    Hc = [to_homogeneous(R, t) for R, t in zip(R_target2cam, t_target2cam)]

    n = len(Hg)
    K = n * (n - 1) // 2
    T = np.zeros((6 * K, 8))

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            Hgij = homogeneous_inverse(Hg[j]) @ Hg[i]
            Hcij = Hc[j] @ homogeneous_inverse(Hc[i])

            dualqa = homogeneous2dualQuaternion(Hgij)
            if dualqa[0, 0] < 0:
                dualqa *= -1
            dualqb = homogeneous2dualQuaternion(Hcij)
            if dualqb[0, 0] < 0:
                dualqb *= -1

            a = dualqa[1:4, 0]
            b = dualqb[1:4, 0]
            aprime = dualqa[5:8, 0]
            bprime = dualqb[5:8, 0]

            s00 = (a - b).reshape(1, 3)
            s01 = skew(a + b)
            s10 = (aprime - bprime).reshape(1, 3)
            s11 = skew(aprime + bprime)
            s12 = (a - b).reshape(1, 3)
            s13 = skew(a + b)

            T[idx*6 : idx*6+3, 0:4] = np.hstack([s00.T, s01])
            T[idx*6+3 : idx*6+6, 0:4] = np.hstack([s10.T, s11])
            T[idx*6+3 : idx*6+6, 4:8] = np.hstack([s12.T, s13])

            idx += 1

    # SVD on T
    _, _, Vt = np.linalg.svd(T)
    v = Vt.T

    u1 = v[0:4, 6]
    v1 = v[4:8, 6]
    u2 = v[0:4, 7]
    v2 = v[4:8, 7]

    ma = u1 @ v1
    mb = u1 @ v2 + u2 @ v1
    mc = u2 @ v2

    a, b, c = ma, mb, mc
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        discriminant = 0
    sqrt_disc = np.sqrt(discriminant)
    s1 = (-b + sqrt_disc) / (2 * a + 1e-12)
    s2 = (-b - sqrt_disc) / (2 * a + 1e-12)

    sol1 = s1*s1 * (u1 @ u1) + 2*s1 * (u1 @ u2) + (u2 @ u2)
    sol2 = s2*s2 * (u1 @ u1) + 2*s2 * (u1 @ u2) + (u2 @ u2)

    if sol1 > sol2:
        s = s1
        val = sol1
    else:
        s = s2
        val = sol2

    lambda2 = np.sqrt(1.0 / (val + 1e-12))
    lambda1 = s * lambda2

    dualq = lambda1 * v[:, 6] + lambda2 * v[:, 7]
    X = dualQuaternion2homogeneous(dualq)

    R_cam2gripper = X[:3, :3]
    t_cam2gripper = X[:3, 3]

    return R_cam2gripper, t_cam2gripper


def handeye_with_scale(robot_poses, camera_poses_noscale, scale):

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i in range(len(robot_poses)):
        T_gripper2base = robot_poses[i]
        R_gripper2base.append(T_gripper2base[:3, :3])
        t_gripper2base.append(T_gripper2base[:3, 3].reshape(3, 1))

        T_target2cam = camera_poses_noscale[i].copy()
        T_target2cam[:3, 3] *= scale
        
        T_cam2target = np.linalg.inv(T_target2cam)
        R_target2cam.append(T_cam2target[:3, :3])
        t_target2cam.append(T_cam2target[:3, 3].reshape(3, 1))

    R_cam2gripper, t_cam2gripper = calibrate_hand_eye_daniilidis(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
    )
    X_custom = np.eye(4)
    X_custom[:3, :3] = R_cam2gripper
    X_custom[:3, 3] = t_cam2gripper.flatten()

    return X_custom


if HAS_NUMBA:
    @jit(nopython=True, fastmath=True)
    def compute_translation_cost_jit(scale, Rx, A_rots, A_trans, B_trans):
        scale = float(scale)
        n = len(A_trans)
        
        C = np.zeros((3 * n, 3))
        d = np.zeros(3 * n)
        
        for i in range(n):
            Ra = A_rots[i]
            ta = A_trans[i]
            tb = B_trans[i]
            
            term1 = scale * (Rx @ tb)
            term2 = np.eye(3) - Ra
            
            C[3*i:3*(i+1), :] = term2
            d[3*i:3*(i+1)] = ta - term1
        
        tx = np.linalg.lstsq(C, d)[0]
        
        cost = 0.0
        for i in range(n):
            err = A_trans[i] - (scale * (Rx @ B_trans[i]) + (np.eye(3) - A_rots[i]) @ tx)
            cost += np.sum(err ** 2)
        
        return cost, tx
else:
    def compute_translation_cost_jit(scale, Rx, A_rots, A_trans, B_trans):

        scale = float(scale)
        n = len(A_trans)
        
        C = np.zeros((3 * n, 3))
        d = np.zeros(3 * n)
        
        for i in range(n):
            Ra = A_rots[i]
            ta = A_trans[i]
            tb = B_trans[i]
            
            term1 = scale * (Rx @ tb)
            term2 = np.eye(3) - Ra
            
            C[3*i:3*(i+1), :] = term2
            d[3*i:3*(i+1)] = ta - term1
        
        tx, *_ = np.linalg.lstsq(C, d, rcond=None)
        
        cost = 0.0
        for i in range(n):
            err = A_trans[i] - (scale * (Rx @ B_trans[i]) + (np.eye(3) - A_rots[i]) @ tx)
            cost += np.sum(err ** 2)
        
        return cost, tx


def solve_handeye_scale_search(robot_poses, camera_poses_noscale, scale_range=(0.01, 10.0), num_scales=20000):

    A_rel = calculate_relative_transforms(robot_poses)
    B_rel_noscale = calculate_relative_transforms(camera_poses_noscale)
    
    if len(A_rel) == 0:
        raise ValueError("Not enough poses to compute relative motion")

    X = handeye_with_scale(robot_poses, camera_poses_noscale, 1)
    Rx = X[:3, :3]

    n = len(A_rel)

    A_rots = np.array([A[:3, :3] for A in A_rel], dtype=np.float64)
    A_trans = np.array([A[:3, 3] for A in A_rel], dtype=np.float64)
    B_trans = np.array([B[:3, 3] for B in B_rel_noscale], dtype=np.float64)
    Rx = np.ascontiguousarray(Rx, dtype=np.float64)

    def cost_function(scale):
        cost, _ = compute_translation_cost_jit(scale, Rx, A_rots, A_trans, B_trans)
        return cost

    result = minimize_scalar(cost_function, bounds=(0.01, 100.0), method='bounded',
                            options={'xatol': 1e-8})
    
    scale_opt = result.x
    _, tx_opt = compute_translation_cost_jit(scale_opt, Rx, A_rots, A_trans, B_trans)
    
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx_opt
    
    return scale_opt, X


# =============================================================================
# M-Estimator
# =============================================================================

def m_estimator_with_scale_search(A_list, B_list, 
                                   m_type=MEstimatorType.HUBER,
                                   scale_range=(0.1, 5.0),
                                   scale_steps=2000,
                                   irls_iter=10):
    n = len(A_list)
    weight_func = get_weight_function(m_type)
    
    best_X = np.eye(4)
    best_scale = 1.0
    best_cost = float('inf')
    
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    
    for scale in scales:
        weights = np.ones(n)
        
        B_scaled = []
        for B in B_list:
            Bs = B.copy()
            Bs[:3, 3] *= scale
            B_scaled.append(Bs)
        
        X = None
        for irls_it in range(irls_iter):
            try:
                Rx = solve_rotation_svd(A_list, B_scaled, weights)
                
                C, d = [], []
                for i in range(n):
                    w_sqrt = np.sqrt(weights[i])
                    Ra = A_list[i][:3, :3]
                    ta = A_list[i][:3, 3]
                    tb = B_scaled[i][:3, 3]
                    C.append(w_sqrt * (np.eye(3) - Ra))
                    d.append(w_sqrt * (ta - Rx @ tb))
                
                C = np.vstack(C)
                d = np.concatenate(d)
                tx, *_ = np.linalg.lstsq(C, d, rcond=None)
                
                X = np.eye(4)
                X[:3, :3] = Rx
                X[:3, 3] = tx
                
                errors = compute_all_errors(A_list, B_list, X, scale)
                sigma = compute_mad(errors)
                if sigma < 1e-10:
                    sigma = 1e-10
                
                weights = np.array([weight_func(e / sigma, 1.345) for e in errors])
                weights = np.clip(weights, 0.01, 1.0)
                weights = weights / np.sum(weights) * n
                
            except Exception:
                break
        
        if X is not None:
            errors = compute_all_errors(A_list, B_list, X, scale)
            cost = np.sum(errors ** 2 * weights)
            
            if cost < best_cost:
                best_cost = cost
                best_X = X.copy()
                best_scale = scale
    
    return best_X, best_scale


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def get_gt_hand_eye(pth_path: str) -> np.ndarray:
    data = torch.load(pth_path, map_location="cpu")
    
    cam_poses = data['poses']
    eef_poses = data['eef_poses']
    
    if isinstance(cam_poses, np.ndarray):
        cam_poses = torch.from_numpy(cam_poses)
    if isinstance(eef_poses, np.ndarray):
        eef_poses = torch.from_numpy(eef_poses)
    
    cam_poses = cam_poses.to(dtype=torch.float64)
    eef_poses = eef_poses.to(dtype=torch.float64)
    
    n = min(cam_poses.shape[0], eef_poses.shape[0])
    eef_inv = torch.linalg.inv(eef_poses[:n])
    hand_eye_mats = eef_inv @ cam_poses[:n]
    
    return hand_eye_mats.mean(dim=0).numpy()


def eval_error(T_gt: np.ndarray, X_pred: np.ndarray) -> dict:
    Delta = np.linalg.inv(T_gt) @ X_pred
    R = Delta[:3, :3]
    t = Delta[:3, 3]
    return {
        "rot_deg": rotation_angle_deg(R),
        "trans_cm": float(np.linalg.norm(t)) * 100.0,
    }


def run_experiment_on_dataset(raw_path: str, eef_path: str, dataset_name: str,
                               use_gt_path: str = None) -> dict:
    """
    Run all calibration methods on a single dataset
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print('='*60)
    
    # Load data
    camera_poses = load_raw_camera_poses(raw_path)
    robot_poses = load_robot_poses(eef_path)
    
    n = min(len(robot_poses), len(camera_poses))
    robot_poses = robot_poses[:n]
    camera_poses = camera_poses[:n]
    
    print(f"Loaded {n} pose pairs")
    
    A_list = calculate_relative_transforms(robot_poses)
    B_list = calculate_relative_transforms(camera_poses)
    
    print(f"Generated {len(A_list)} motion pairs")
    
    T_gt = None
    if use_gt_path and os.path.exists(use_gt_path):
        try:
            T_gt = get_gt_hand_eye(use_gt_path)
            print("Ground truth loaded")
        except:
            pass
    
    results = {}
    
    print("\n[1/4] Running Tsai-Lenz (Original)...")
    try:
        scale_tl, X_tl = tsai_lenz_scaled(A_list, B_list, num_scales=20000)
        results['tsai_lenz'] = {'X': X_tl, 'scale': scale_tl}
        if T_gt is not None:
            err = eval_error(T_gt, X_tl)
            results['tsai_lenz']['error'] = err
            print(f"      Scale={scale_tl:.4f}, Rot={err['rot_deg']:.4f}°, Trans={err['trans_cm']:.4f}cm")
    except Exception as e:
        print(f"      Error: {e}")
    
    print("[2/4] Running Park-Martin Daniilidis (Original)...")
    try:
        scale_pm, X_pm = solve_handeye_scale_search(robot_poses, camera_poses)
        results['park_martin'] = {'X': X_pm, 'scale': scale_pm}
        if T_gt is not None:
            err = eval_error(T_gt, X_pm)
            results['park_martin']['error'] = err
            print(f"      Scale={scale_pm:.4f}, Rot={err['rot_deg']:.4f}°, Trans={err['trans_cm']:.4f}cm")
    except Exception as e:
        print(f"      Error: {e}")
    
    print("[3/4] Running RANSAC...")
    try:
        config = RANSACConfig(
            min_samples=3,             
            max_iterations=1000,
            inlier_threshold=0.1,
            scale_range=(0.5, 5.0),
            scale_steps=2000,
            refine_with_m_estimator=False,
            pre_estimate_scale=True,   
            verbose=False
        )
        result = ransac_hand_eye(A_list, B_list, config)
        results['ransac'] = {
            'X': result.X, 
            'scale': result.scale,
            'inlier_ratio': result.inlier_ratio
        }
        if T_gt is not None:
            err = eval_error(T_gt, result.X)
            results['ransac']['error'] = err
            print(f"      Scale={result.scale:.4f}, Inliers={result.inlier_ratio*100:.1f}%, Rot={err['rot_deg']:.4f}°, Trans={err['trans_cm']:.4f}cm")
    except Exception as e:
        print(f"      Error: {e}")
    
    print("[4/4] Running M-Estimator (Huber)...")
    try:
        X_me, scale_me = m_estimator_with_scale_search(
            A_list, B_list,
            m_type=MEstimatorType.HUBER,
            scale_range=(0.1, 5.0),
            scale_steps=2000
        )
        results['m_estimator'] = {'X': X_me, 'scale': scale_me}
        if T_gt is not None:
            err = eval_error(T_gt, X_me)
            results['m_estimator']['error'] = err
            print(f"      Scale={scale_me:.4f}, Rot={err['rot_deg']:.4f}°, Trans={err['trans_cm']:.4f}cm")
    except Exception as e:
        print(f"      Error: {e}")
    
    return results


def run_full_experiment(clean_data_dir: str, noisy_data_dir: str, results_dir: str):
    """
    Run complete experiment comparing methods on clean and noisy data
    """
    print("\n" + "=" * 70)
    print("   ROBUST HAND-EYE CALIBRATION EXPERIMENT")
    print("   Using Original Tsai-Lenz and Daniilidis Dual Quaternion")
    print("=" * 70)
    print(f"Clean data: {clean_data_dir}")
    print(f"Noisy data: {noisy_data_dir}")
    print(f"Results:    {results_dir}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Find all datasets
    all_files = os.listdir(clean_data_dir)
    raw_files = [f for f in all_files if f.endswith('_raw.pth')]
    
    datasets = []
    for raw_file in raw_files:
        eef_file = raw_file.replace('_raw.pth', '.pth')
        if eef_file in all_files:
            name = raw_file.replace('_raw.pth', '')
            datasets.append((name, raw_file, eef_file))
    
    print(f"\nFound {len(datasets)} datasets")
    
    all_results = {
        'clean': {},
        'noisy': {},
        'summary': {}
    }
    
    # ==================== CLEAN DATA ====================
    print("\n" + "=" * 70)
    print("   PHASE 1: CLEAN DATA (Baseline)")
    print("=" * 70)
    
    for name, raw_file, eef_file in datasets:
        raw_path = os.path.join(clean_data_dir, raw_file)
        eef_path = os.path.join(clean_data_dir, eef_file)
        gt_path = eef_path
        
        results = run_experiment_on_dataset(raw_path, eef_path, f"{name} (clean)", gt_path)
        all_results['clean'][name] = results
        
        # Save individual results
        for method, res in results.items():
            if 'X' in res:
                np.save(os.path.join(results_dir, f"{method}_X_{name}_clean.npy"), res['X'])
    
    # ==================== NOISY DATA ====================
    if noisy_data_dir and os.path.exists(noisy_data_dir):
        print("\n" + "=" * 70)
        print("   PHASE 2: NOISY DATA (Robustness Test)")
        print("=" * 70)
        
        for name, raw_file, eef_file in datasets:
            raw_path = os.path.join(noisy_data_dir, raw_file)
            eef_path = os.path.join(noisy_data_dir, eef_file)
            gt_path = os.path.join(clean_data_dir, eef_file)
            
            if not os.path.exists(raw_path):
                print(f"Skipping {name}: noisy data not found")
                continue
            
            results = run_experiment_on_dataset(raw_path, eef_path, f"{name} (noisy)", gt_path)
            all_results['noisy'][name] = results
            
            for method, res in results.items():
                if 'X' in res:
                    np.save(os.path.join(results_dir, f"{method}_X_{name}_noisy.npy"), res['X'])
    
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    
    methods = ['tsai_lenz', 'park_martin', 'ransac', 'm_estimator']
    
    for condition in ['clean', 'noisy']:
        if not all_results[condition]:
            continue
        
        print(f"\n{condition.upper()} DATA:")
        print("-" * 70)
        print(f"{'Method':<25} {'Avg Rot (°)':<15} {'Avg Trans (cm)':<15}")
        print("-" * 70)
        
        summary = {}
        for method in methods:
            rot_errors = []
            trans_errors = []
            
            for name, results in all_results[condition].items():
                if method in results and 'error' in results[method]:
                    rot_errors.append(results[method]['error']['rot_deg'])
                    trans_errors.append(results[method]['error']['trans_cm'])
            
            if rot_errors:
                avg_rot = np.mean(rot_errors)
                avg_trans = np.mean(trans_errors)
                std_rot = np.std(rot_errors)
                std_trans = np.std(trans_errors)
                
                summary[method] = {
                    'avg_rot': avg_rot,
                    'std_rot': std_rot,
                    'avg_trans': avg_trans,
                    'std_trans': std_trans,
                    'count': len(rot_errors)
                }
                
                print(f"{method:<25} {avg_rot:.4f} ± {std_rot:.4f}   {avg_trans:.4f} ± {std_trans:.4f}")
        
        all_results['summary'][condition] = summary
    
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(all_results)
    
    with open(os.path.join(results_dir, 'experiment_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}/experiment_results.json")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run robust hand-eye calibration experiment')
    
    parser.add_argument('--clean_dir', type=str,
                        default=os.path.join(project_folder, 'data', 'dust3r_saved_output'),
                        help='Directory with clean/original data')
    parser.add_argument('--noisy_dir', type=str, default=None,
                        help='Directory with noisy data (default: auto-detect)')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(project_folder, 'results', 'robust_experiment'),
                        help='Directory to save results')
    
    args = parser.parse_args()

    if args.noisy_dir is None:
        parent_dir = os.path.dirname(args.clean_dir)
        noisy_candidates = [d for d in os.listdir(parent_dir) if d.startswith('noisy_')]
        if noisy_candidates:
            args.noisy_dir = os.path.join(parent_dir, noisy_candidates[0])
            print(f"Auto-detected noisy data: {args.noisy_dir}")
    
    run_full_experiment(args.clean_dir, args.noisy_dir, args.results_dir)


if __name__ == "__main__":
    main()