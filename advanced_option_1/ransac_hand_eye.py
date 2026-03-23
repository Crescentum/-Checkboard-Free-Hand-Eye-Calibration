import numpy as np
import cv2
import sys
import os
import torch
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_folder)


# =============================================================================
# Data Loading 
# =============================================================================

def load_raw_camera_poses(filepath: str) -> np.ndarray:
    data = torch.load(filepath, map_location='cpu')
    if isinstance(data, dict) and 'poses' in data:
        camera_poses = data['poses'].numpy() if isinstance(data['poses'], torch.Tensor) else data['poses']
    else:
        camera_poses = data.numpy() if isinstance(data, torch.Tensor) else data
    return camera_poses


def load_robot_poses(filepath: str) -> np.ndarray:
    data = torch.load(filepath, map_location='cpu')
    robot_poses = data['eef_poses'].numpy() if isinstance(data['eef_poses'], torch.Tensor) else data['eef_poses']
    return robot_poses


def calculate_relative_transforms(poses: np.ndarray) -> List[np.ndarray]:
    rels = []
    for i in range(len(poses) - 1):
        T1 = poses[i]
        T2 = poses[i + 1]
        T_rel = np.linalg.inv(T1) @ T2
        rels.append(T_rel)
    return rels


# =============================================================================
# Rotation Utilities
# =============================================================================

def rot_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    return q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# =============================================================================
# M-Estimators
# =============================================================================

class MEstimatorType(Enum):
    L2 = "l2"           
    HUBER = "huber"     
    TUKEY = "tukey"     
    CAUCHY = "cauchy"   


def huber_weight(residual: float, k: float = 1.345) -> float:
    abs_r = np.abs(residual)
    if abs_r <= k:
        return 1.0
    return k / abs_r


def tukey_weight(residual: float, c: float = 4.685) -> float:
    abs_r = np.abs(residual)
    if abs_r > c:
        return 0.0
    ratio = residual / c
    return (1 - ratio**2) ** 2


def cauchy_weight(residual: float, c: float = 2.385) -> float:
    return 1.0 / (1.0 + (residual / c) ** 2)


def get_weight_function(m_type: MEstimatorType) -> Callable[[float, float], float]:
    if m_type == MEstimatorType.L2:
        return lambda r, k: 1.0
    elif m_type == MEstimatorType.HUBER:
        return huber_weight
    elif m_type == MEstimatorType.TUKEY:
        return tukey_weight
    elif m_type == MEstimatorType.CAUCHY:
        return cauchy_weight
    else:
        return lambda r, k: 1.0


def compute_mad(residuals: np.ndarray) -> float:
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    return mad * 1.4826


def solve_rotation_svd(A_list: List[np.ndarray], B_list: List[np.ndarray],
                       weights: Optional[np.ndarray] = None) -> np.ndarray:

    n = len(A_list)
    if weights is None:
        weights = np.ones(n)
    
    M = np.zeros((3, 3))
    for i in range(n):
        Ra = A_list[i][:3, :3]
        Rb = B_list[i][:3, :3]
        
        alpha = cv2.Rodrigues(Ra)[0].flatten()
        beta = cv2.Rodrigues(Rb)[0].flatten()
        
        M += weights[i] * np.outer(beta, alpha)
    
    U, S, Vt = np.linalg.svd(M)
    Rx = Vt.T @ U.T
    
    if np.linalg.det(Rx) < 0:
        Rx = -Rx
    
    return Rx


def solve_rotation_quaternion(A_list: List[np.ndarray], B_list: List[np.ndarray],
                               weights: Optional[np.ndarray] = None) -> np.ndarray:

    def quat_left_matrix(q):
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])
    
    def quat_right_matrix(q):
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w,  z, -y],
            [y, -z,  w,  x],
            [z,  y, -x,  w]
        ])
    
    n = len(A_list)
    if weights is None:
        weights = np.ones(n)
    
    valid_indices = []
    for i in range(n):
        Ra = A_list[i][:3, :3]
        Rb = B_list[i][:3, :3]
        angle_a = np.arccos(np.clip((np.trace(Ra) - 1) / 2, -1, 1))
        angle_b = np.arccos(np.clip((np.trace(Rb) - 1) / 2, -1, 1))
        if angle_a > 1e-3 and angle_b > 1e-3:
            valid_indices.append(i)
    
    if len(valid_indices) < 2:
        raise ValueError("Not enough non-degenerate motions for quaternion solver")
    
    M_rows = []
    for i in valid_indices:
        Ra = A_list[i][:3, :3]
        Rb = B_list[i][:3, :3]
        qa = rot_to_quat(Ra)
        qb = rot_to_quat(Rb)
        qa /= np.linalg.norm(qa)
        qb /= np.linalg.norm(qb)
        L_qa = quat_left_matrix(qa)
        R_qb = quat_right_matrix(qb)
        row = np.sqrt(weights[i]) * (L_qa - R_qb)
        M_rows.append(row)
    
    M = np.vstack(M_rows)
    U, S, Vt = np.linalg.svd(M)
    qr = Vt[-1]
    qr /= np.linalg.norm(qr)
    if qr[0] < 0:
        qr = -qr
    
    return quat_to_rot(qr)


def solve_translation_with_scale(A_list: List[np.ndarray], B_list: List[np.ndarray],
                                  Rx: np.ndarray, scale: float,
                                  weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Solve for translation given rotation and scale
    (I - R_A) t_X = t_A - s * R_X * t_B
    """
    n = len(A_list)
    if weights is None:
        weights = np.ones(n)
    
    C = []
    d = []
    for i in range(n):
        w_sqrt = np.sqrt(weights[i])
        Ra = A_list[i][:3, :3]
        ta = A_list[i][:3, 3]
        tb = B_list[i][:3, 3]
        
        C.append(w_sqrt * (np.eye(3) - Ra))
        d.append(w_sqrt * (ta - scale * Rx @ tb))
    
    C = np.vstack(C)
    d = np.concatenate(d)
    t_X, *_ = np.linalg.lstsq(C, d, rcond=None)
    
    return t_X


def hand_eye_solver_basic(A_list: List[np.ndarray], B_list: List[np.ndarray],
                           scale: float = 1.0, weights: Optional[np.ndarray] = None,
                           rotation_method: str = "svd") -> np.ndarray:

    B_scaled = []
    for B in B_list:
        Bs = B.copy()
        Bs[:3, 3] *= scale
        B_scaled.append(Bs)

    if rotation_method == "quaternion":
        Rx = solve_rotation_quaternion(A_list, B_scaled, weights)
    else:
        Rx = solve_rotation_svd(A_list, B_scaled, weights)

    tx = solve_translation_with_scale(A_list, B_list, Rx, scale, weights)
    
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx
    
    return X



def compute_motion_error(A: np.ndarray, B: np.ndarray, X: np.ndarray, 
                         scale: float = 1.0) -> Tuple[float, float, float]:
  
    B_scaled = B.copy()
    B_scaled[:3, 3] *= scale
    
    err_mat = A @ X - X @ B_scaled
    
    total_error = np.linalg.norm(err_mat, 'fro')
    
    R_err = err_mat[:3, :3]
    t_err = err_mat[:3, 3]
    
    rotation_error = np.linalg.norm(R_err, 'fro')
    translation_error = np.linalg.norm(t_err)
    
    return total_error, rotation_error, translation_error


def compute_all_errors(A_list: List[np.ndarray], B_list: List[np.ndarray],
                       X: np.ndarray, scale: float = 1.0) -> np.ndarray:
    errors = []
    for A, B in zip(A_list, B_list):
        total_err, _, _ = compute_motion_error(A, B, X, scale)
        errors.append(total_err)
    return np.array(errors)


def estimate_scale_robust(A_list: List[np.ndarray], B_list: List[np.ndarray],
                          scale_range: Tuple[float, float] = (0.1, 10.0),
                          scale_steps: int = 2000,
                          use_median: bool = False) -> float:
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    
    best_scale = 1.0
    best_error = float('inf')
    
    for scale in scales:
        try:
            X = hand_eye_solver_basic(A_list, B_list, scale, rotation_method="svd")
            
            # Compute per-motion errors
            errors = compute_all_errors(A_list, B_list, X, scale)
            
            if use_median:
                aggregate_error = np.median(errors)
            else:
                sorted_errors = np.sort(errors)
                n_keep = max(3, int(len(errors) * 0.8))
                aggregate_error = np.mean(sorted_errors[:n_keep])
            
            if aggregate_error < best_error:
                best_error = aggregate_error
                best_scale = scale
                
        except Exception:
            continue
    
    return best_scale


def estimate_scale_ransac_style(A_list: List[np.ndarray], B_list: List[np.ndarray],
                                 scale_range: Tuple[float, float] = (0.1, 10.0),
                                 scale_steps: int = 1000,
                                 inlier_threshold: float = 0.1,
                                 num_trials: int = 100) -> Tuple[float, float]:

    n = len(A_list)
    subset_size = max(n // 2, 4)  # Use at least half the data
    
    best_scale = 1.0
    best_inlier_count = 0
    
    for _ in range(num_trials):
        indices = np.random.choice(n, subset_size, replace=False)
        A_sub = [A_list[i] for i in indices]
        B_sub = [B_list[i] for i in indices]
        
        scale = estimate_scale_robust(A_sub, B_sub, scale_range, scale_steps // 10)

        X = hand_eye_solver_basic(A_list, B_list, scale)
        errors = compute_all_errors(A_list, B_list, X, scale)
        inlier_count = np.sum(errors < inlier_threshold)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_scale = scale
    
    return best_scale, best_inlier_count / n


@dataclass
class RANSACConfig:
    min_samples: int = 3        
    max_iterations: int = 1000    
    inlier_threshold: float = 0.1  
    min_inlier_ratio: float = 0.3  
    confidence: float = 0.99     
    adaptive_iterations: bool = True  
    scale_range: Tuple[float, float] = (0.1, 10.0) 
    scale_steps: int = 2000     
    rotation_method: str = "svd"  
    refine_with_m_estimator: bool = True  
    m_estimator_type: MEstimatorType = MEstimatorType.HUBER
    pre_estimate_scale: bool = True 
    verbose: bool = False


@dataclass
class RANSACResult:
    X: np.ndarray                
    scale: float                  
    inlier_mask: np.ndarray       
    num_inliers: int             
    inlier_ratio: float           
    iterations_used: int          
    final_error: float            
    rotation_errors: np.ndarray   
    translation_errors: np.ndarray  


def compute_ransac_iterations(inlier_ratio: float, min_samples: int, 
                               confidence: float = 0.99) -> int:
    if inlier_ratio <= 0:
        return int(1e6)
    if inlier_ratio >= 1:
        return 1
    
    w_n = inlier_ratio ** min_samples
    if w_n >= 1:
        return 1
    
    k = np.log(1 - confidence) / np.log(1 - w_n)
    return int(np.ceil(k))


def ransac_hand_eye(A_list: List[np.ndarray], B_list: List[np.ndarray],
                    config: RANSACConfig = None) -> RANSACResult:

    if config is None:
        config = RANSACConfig()
    
    n = len(A_list)
    
    if n < config.min_samples:
        raise ValueError(f"Need at least {config.min_samples} motion pairs, got {n}")
    
    coarse_scale = None
    if config.pre_estimate_scale:
        coarse_scale = estimate_scale_robust(
            A_list, B_list,
            scale_range=config.scale_range,
            scale_steps=500,  
            use_median=False
        )
        scale_min = coarse_scale * 0.5
        scale_max = coarse_scale * 1.5
        if config.verbose:
            print(f"  Coarse scale estimate: {coarse_scale:.4f}")
            print(f"  Acceptable range: [{scale_min:.2f}, {scale_max:.2f}]")
    else:
        scale_min, scale_max = config.scale_range
    
    best_inlier_count = 0
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_X = np.eye(4)
    best_scale = 1.0
    best_error = float('inf')
    
    max_iter = config.max_iterations
    iteration = 0
    
    if config.verbose:
        print(f"Starting RANSAC with {n} motion pairs...")
        print(f"  Min samples: {config.min_samples}")
        print(f"  Inlier threshold: {config.inlier_threshold}")
    
    while iteration < max_iter:
        sample_indices = np.random.choice(n, config.min_samples, replace=False)
        A_sample = [A_list[i] for i in sample_indices]
        B_sample = [B_list[i] for i in sample_indices]
        
        try:
            scales = np.linspace(config.scale_range[0], config.scale_range[1], 
                                 config.scale_steps // 10)
            best_sample_error = float('inf')
            sample_scale = 1.0
            sample_X = np.eye(4)
            
            for s in scales:
                try:
                    X_try = hand_eye_solver_basic(A_sample, B_sample, s,
                                                  rotation_method=config.rotation_method)
                    errors = compute_all_errors(A_sample, B_sample, X_try, s)
                    err = np.sum(errors ** 2)
                    if err < best_sample_error:
                        best_sample_error = err
                        sample_scale = s
                        sample_X = X_try
                except:
                    continue

            if config.pre_estimate_scale:
                if sample_scale < scale_min or sample_scale > scale_max:
                    iteration += 1
                    continue  
            
            scale = sample_scale
            X = sample_X
            
            errors = compute_all_errors(A_list, B_list, X, scale)
            
            inlier_mask = errors < config.inlier_threshold
            inlier_count = np.sum(inlier_mask)
            
            if inlier_count > best_inlier_count or \
               (inlier_count == best_inlier_count and inlier_count > 0 and 
                np.mean(errors[inlier_mask]) < best_error):
                best_inlier_count = inlier_count
                best_inlier_mask = inlier_mask
                best_X = X
                best_scale = scale
                best_error = np.mean(errors[inlier_mask]) if inlier_count > 0 else float('inf')
                
                if config.adaptive_iterations and inlier_count > 0:
                    inlier_ratio = inlier_count / n
                    new_max_iter = compute_ransac_iterations(
                        inlier_ratio, config.min_samples, config.confidence
                    )
                    max_iter = min(config.max_iterations, new_max_iter + iteration)
        
        except Exception:
            pass  
        
        iteration += 1
    
    if config.verbose:
        print(f"RANSAC completed: {iteration} iterations")
        print(f"  Best inliers: {best_inlier_count}/{n} ({100*best_inlier_count/n:.1f}%)")
    
    if best_inlier_count >= config.min_samples:
        A_inliers = [A_list[i] for i in range(n) if best_inlier_mask[i]]
        B_inliers = [B_list[i] for i in range(n) if best_inlier_mask[i]]
        
        scales = np.linspace(config.scale_range[0], config.scale_range[1], config.scale_steps)
        best_refine_error = float('inf')
        
        for s in scales:
            try:
                X_try = hand_eye_solver_basic(A_inliers, B_inliers, s,
                                              rotation_method=config.rotation_method)
                errors = compute_all_errors(A_inliers, B_inliers, X_try, s)
                err = np.sum(errors ** 2)
                if err < best_refine_error:
                    best_refine_error = err
                    best_scale = s
                    best_X = X_try
            except:
                continue
        
        if config.refine_with_m_estimator:
            best_X, best_scale = refine_with_m_estimator(
                A_list, B_list, best_X, best_scale, 
                config.m_estimator_type, config.inlier_threshold
            )
            
            errors = compute_all_errors(A_list, B_list, best_X, best_scale)
            best_inlier_mask = errors < config.inlier_threshold
            best_inlier_count = np.sum(best_inlier_mask)
    
    all_errors = compute_all_errors(A_list, B_list, best_X, best_scale)
    
    rot_errors = []
    trans_errors = []
    for i in range(n):
        _, r_err, t_err = compute_motion_error(A_list[i], B_list[i], best_X, best_scale)
        rot_errors.append(r_err)
        trans_errors.append(t_err)
    
    return RANSACResult(
        X=best_X,
        scale=best_scale,
        inlier_mask=best_inlier_mask,
        num_inliers=best_inlier_count,
        inlier_ratio=best_inlier_count / n,
        iterations_used=iteration,
        final_error=np.mean(all_errors[best_inlier_mask]) if best_inlier_count > 0 else float('inf'),
        rotation_errors=np.array(rot_errors),
        translation_errors=np.array(trans_errors)
    )


def refine_with_m_estimator(A_list: List[np.ndarray], B_list: List[np.ndarray],
                            X_init: np.ndarray, scale_init: float,
                            m_type: MEstimatorType = MEstimatorType.HUBER,
                            threshold: float = 0.1,
                            max_iter: int = 20,
                            tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    n = len(A_list)
    weight_func = get_weight_function(m_type)
    
    X = X_init.copy()
    scale = scale_init
    
    for iteration in range(max_iter):
        errors = compute_all_errors(A_list, B_list, X, scale)
        
        sigma = compute_mad(errors)
        if sigma < 1e-10:
            sigma = 1e-10

        normalized_errors = errors / sigma
        weights = np.array([weight_func(e, threshold / sigma) for e in normalized_errors])
        
        if np.sum(weights) < 1e-10:
            weights = np.ones(n)
        
        weights = weights / np.sum(weights) * n
        
        X_new = hand_eye_solver_basic(A_list, B_list, scale, weights)
        

        delta = np.linalg.norm(X_new - X, 'fro')
        X = X_new
        
        if delta < tol:
            break
    
    return X, scale


# =============================================================================
# Combined RANSAC + M-Estimator Pipeline
# =============================================================================

def robust_hand_eye_calibration(A_list: List[np.ndarray], B_list: List[np.ndarray],
                                 config: RANSACConfig = None) -> RANSACResult:
    if config is None:
        config = RANSACConfig()
        config.refine_with_m_estimator = True
        config.pre_estimate_scale = True  # Enable the fix by default
    
    result = ransac_hand_eye(A_list, B_list, config)
    
    if config.verbose:
        print("\n=== Robust Hand-Eye Calibration Results ===")
        print(f"Scale: {result.scale:.6f}")
        print(f"Inliers: {result.num_inliers}/{len(A_list)} ({result.inlier_ratio*100:.1f}%)")
        print(f"Mean error (inliers): {result.final_error:.6f}")
        print(f"Hand-eye matrix X:")
        print(result.X)
    
    return result

def process_dataset_ransac(raw_path: str, eef_path: str, dataset_name: str,
                            results_dir: str, config: RANSACConfig = None):
    """Process a single dataset with RANSAC hand-eye calibration"""
    print(f"\nProcessing dataset: {dataset_name}")
    
    camera_poses = load_raw_camera_poses(raw_path)
    robot_poses = load_robot_poses(eef_path)
    
    print(f"  Loaded {len(robot_poses)} robot poses")
    print(f"  Loaded {len(camera_poses)} camera poses")
    
    if len(robot_poses) != len(camera_poses):
        print("  Warning: Pose count mismatch!")
        min_len = min(len(robot_poses), len(camera_poses))
        robot_poses = robot_poses[:min_len]
        camera_poses = camera_poses[:min_len]
        print(f"  Using first {min_len} poses")
    
    A_list = calculate_relative_transforms(robot_poses)
    B_list = calculate_relative_transforms(camera_poses)
    
    print(f"  Generated {len(A_list)} motion pairs")
    
    if config is None:
        config = RANSACConfig()
    
    result = robust_hand_eye_calibration(A_list, B_list, config)

    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, f"ransac_scale_{dataset_name}.npy"), result.scale)
    np.save(os.path.join(results_dir, f"ransac_X_{dataset_name}.npy"), result.X)
    np.save(os.path.join(results_dir, f"ransac_inliers_{dataset_name}.npy"), result.inlier_mask)
    np.save(os.path.join(results_dir, f"ransac_errors_{dataset_name}.npy"), {
        "R_L": result.rotation_errors,
        "t_L": result.translation_errors,
        "inlier_ratio": result.inlier_ratio,
        "num_inliers": result.num_inliers
    })
    
    print(f"  Results saved to {results_dir}")
    
    return result


def main():
    print("=" * 60)
    print("RANSAC Hand-Eye Calibration with M-Estimator")
    print("=" * 60)
    
    output_dir = os.path.join(project_folder, "data", "dust3r_saved_output")
    results_dir = os.path.join(project_folder, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(output_dir):
        print(f"\nError: Output directory not found: {output_dir}")
        print("Please run the data processing pipeline first.")
        return
    
    config = RANSACConfig(
        min_samples=4,            
        max_iterations=1000,
        inlier_threshold=0.1,       
        confidence=0.99,
        scale_range=(0.1, 10.0),
        scale_steps=2000,
        rotation_method="svd",
        refine_with_m_estimator=True,
        m_estimator_type=MEstimatorType.HUBER,
        pre_estimate_scale=True,  
        verbose=True
    )
    
    all_files = os.listdir(output_dir)
    raw_files = [f for f in all_files if f.endswith('_raw.pth')]
    
    pairs = []
    for raw_file in raw_files:
        eef_file = raw_file.replace('_raw.pth', '.pth')
        if eef_file in all_files:
            pairs.append((raw_file, eef_file))
    
    if not pairs:
        print(f"\nError: No valid dataset pairs found in {output_dir}")
        return
    
    print(f"Found {len(pairs)} datasets to process.")
    
    all_results = {}
    
    for raw_file, eef_file in pairs:
        dataset_name = raw_file.replace('_raw.pth', '')
        raw_path = os.path.join(output_dir, raw_file)
        eef_path = os.path.join(output_dir, eef_file)
        
        try:
            result = process_dataset_ransac(raw_path, eef_path, dataset_name, 
                                             results_dir, config)
            all_results[dataset_name] = result
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Scale: {result.scale:.6f}")
        print(f"  Inliers: {result.num_inliers} ({result.inlier_ratio*100:.1f}%)")
        print(f"  Mean error: {result.final_error:.6f}")
    
    print("\n" + "=" * 60)
    print("Batch processing complete.")


if __name__ == "__main__":
    main()