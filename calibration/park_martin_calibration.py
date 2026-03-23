import numpy as np
import json
import os
from scipy.optimize import minimize_scalar
from numba import jit


def load_json_poses(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    poses = np.array(data['poses'])
    print(f"  Loaded {len(poses)} poses from {os.path.basename(filepath)}")
    return poses


def calculate_relative_transforms(poses):
    rels = []
    for i in range(len(poses) - 1):
        T1 = poses[i]
        T2 = poses[i + 1]
        T_rel = np.dot(np.linalg.inv(T1), T2)
        rels.append(T_rel)
    return rels


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
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S = 4 * qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S = 4 * qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S = 4 * qz
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

def normalize_rotation(R):
    det = np.linalg.det(R)
    if abs(det) < 1e-6:
        raise ValueError("Determinant is near zero.")
    R = np.cbrt(np.copysign(1, det) / abs(det)) * R
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

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

def calculate_relative_transforms(poses):
    rels = []
    for i in range(len(poses) - 1):
        T1 = poses[i]
        T2 = poses[i + 1]
        T_rel = np.dot(np.linalg.inv(T1), T2)
        rels.append(T_rel)
    return rels

@jit(nopython=True, fastmath=True)
def compute_reprojection_error_jit(A_rel_list, B_rel_list, X):
    total_err = 0.0
    for i in range(len(A_rel_list)):
        residual = A_rel_list[i] @ X - X @ B_rel_list[i]
        total_err += np.sum(residual ** 2)
    return total_err


def compute_reprojection_error_from_relatives(A_rel, B_rel, X):
    total_err = 0.0
    for A, B in zip(A_rel, B_rel):
        residual = A @ X - X @ B
        total_err += np.linalg.norm(residual, 'fro') ** 2
    return total_err



def solve_handeye_scale_search(robot_poses, camera_poses_noscale, scale_range=(0.01, 10.0), num_scales=20000):
    A_rel = calculate_relative_transforms(robot_poses)
    B_rel_noscale = calculate_relative_transforms(camera_poses_noscale)
    
    if len(A_rel) == 0:
        raise ValueError("Not enough poses to compute relative motion")

    print(f"  优化scale范围: {scale_range[0]} ~ {scale_range[1]})")
    
    A_rel_array = np.array(A_rel, dtype=np.float64)
    B_rel_noscale_array = np.array(B_rel_noscale, dtype=np.float64)
    
    def cost_function(scale):
        X = handeye_with_scale(robot_poses, camera_poses_noscale, scale)
        
        B_rel_scaled_array = B_rel_noscale_array.copy()
        B_rel_scaled_array[:, :3, 3] *= scale
        
        error = compute_reprojection_error_jit(A_rel_array, B_rel_scaled_array, X)
        return error
    
    result = minimize_scalar(cost_function, bounds=scale_range, method='bounded',
                            options={'xatol': 1e-6, 'maxiter': 500})
    
    best_scale = result.x
    best_error = result.fun
    
    print(f"最优尺度: {best_scale:.6f}, 误差: {best_error:.6f}")
    
    best_X = handeye_with_scale(robot_poses, camera_poses_noscale, best_scale)
    
    errors_rot = []
    errors_trans = []
    B_rel_scaled = []
    for B in B_rel_noscale:
        B_scaled = B.copy()
        B_scaled[:3, 3] *= best_scale
        B_rel_scaled.append(B_scaled)
    
    for i in range(len(A_rel)):
        residual = A_rel[i] @ best_X - best_X @ B_rel_scaled[i]
        errors_rot.append(np.linalg.norm(residual[:3, :3], 'fro'))
        errors_trans.append(np.linalg.norm(residual[:3, 3]))
    
    return best_scale, best_X, np.array(errors_rot), np.array(errors_trans)


def process_dataset(dataset_path, dataset_name):
    
    inputs_dir = os.path.join(dataset_path, 'inputs')
    robot_poses = load_json_poses(os.path.join(inputs_dir, 'robot_poses.json'))
    camera_poses_raw = load_json_poses(os.path.join(inputs_dir, 'camera_poses_raw.json'))
    
    if len(robot_poses) != len(camera_poses_raw):
        min_len = min(len(robot_poses), len(camera_poses_raw))
        robot_poses = robot_poses[:min_len]
        camera_poses_raw = camera_poses_raw[:min_len]

    scale, X, R_L, t_L = solve_handeye_scale_search(
        robot_poses, camera_poses_raw, scale_range=(0.01, 10.0), num_scales=200000
    )
    
    print(f"\n{'─'*60}")
    print("标定结果:")
    print('─'*60)
    print(f"尺度因子 s: {scale:.6f}")
    print(f"\n手眼变换矩阵 X:")
    print(X)
    print(f"\n误差统计:")
    print(f"  旋转误差 - 均值: {R_L.mean():.6f}, 最大: {R_L.max():.6f}")
    print(f"  平移误差 - 均值: {t_L.mean():.6f}, 最大: {t_L.max():.6f}")
    
    return scale, X, R_L, t_L


def save_results(dataset_path, dataset_name, scale, X):
    outputs_dir = os.path.join(dataset_path, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    result_data = {
        'dataset_name': dataset_name,
        'algorithm': 'Park-Martin with Scale Estimation',
        'scale_factor': float(scale),
        'hand_eye_transform': X.tolist(),
        'description': 'Hand-eye calibration result: X transforms from camera to robot end-effector'
    }
    
    output_file = os.path.join(outputs_dir, 'park_martin_result.json')
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)


def main():
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    organized_data_dir = os.path.join(project_folder, 'organized_data')
    
    if not os.path.exists(organized_data_dir):
        return
    
    datasets = [d for d in os.listdir(organized_data_dir) 
                if os.path.isdir(os.path.join(organized_data_dir, d))]
    
    datasets.sort()
    
    for dataset_name in datasets:
        dataset_path = os.path.join(organized_data_dir, dataset_name)
        
        try:
            scale, X, errors_rot, errors_trans = process_dataset(dataset_path, dataset_name)
            save_results(dataset_path, dataset_name, scale, X)
        except Exception as e:
            import traceback
            traceback.print_exc()
    

if __name__ == "__main__":
    main()
