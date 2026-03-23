import numpy as np
import json
import os
from scipy.optimize import minimize_scalar
from numba import jit


def rotation_matrix_to_vector(R):
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    if theta < 1e-10:
        return np.zeros(3)
    
    r = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))
    
    return theta * r


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


def tsai_lenz_scaled(A_list, B_list):
    n = len(A_list)
    print(f"  使用 {n} 对相对运动进行标定")
    
    M = np.zeros((3, 3))
    
    for i in range(n):
        Ra = A_list[i][:3, :3]
        Rb = B_list[i][:3, :3]
        
        alpha = rotation_matrix_to_vector(Ra)
        beta = rotation_matrix_to_vector(Rb)
        
        M += np.outer(beta, alpha)
    
    U, S, Vt = np.linalg.svd(M)
    Rx = np.dot(Vt.T, U.T)
    
    if np.linalg.det(Rx) < 0:
        Rx = -Rx
    
    print("  旋转矩阵求解完成")
    
    A_rots = np.array([A[:3, :3] for A in A_list], dtype=np.float64)
    A_trans = np.array([A[:3, 3] for A in A_list], dtype=np.float64)
    B_trans = np.array([B[:3, 3] for B in B_list], dtype=np.float64)
    Rx = np.ascontiguousarray(Rx, dtype=np.float64)
    
    print("  优化尺度因子...")
    
    def cost_function(scale):
        cost, _ = compute_translation_cost_jit(scale, Rx, A_rots, A_trans, B_trans)
        return cost
    
    result = minimize_scalar(cost_function, bounds=(0.01, 100.0), method='bounded',
                            options={'xatol': 1e-8})
    
    scale_opt = result.x
    _, tx_opt = compute_translation_cost_jit(scale_opt, Rx, A_rots, A_trans, B_trans)
    
    print(f"  最优尺度: {scale_opt:.6f}")
    
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx_opt
    
    errors_rot = []
    errors_trans = []
    
    for i in range(n):
        B_scaled = B_list[i].copy()
        B_scaled[:3, 3] = scale_opt * B_list[i][:3, 3]
        
        residual = A_list[i] @ X - X @ B_scaled
        
        errors_rot.append(np.linalg.norm(residual[:3, :3], 'fro'))
        errors_trans.append(np.linalg.norm(residual[:3, 3]))
    
    return scale_opt, X, np.array(errors_rot), np.array(errors_trans)


def process_dataset(dataset_path, dataset_name):
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print('='*60)
    
    inputs_dir = os.path.join(dataset_path, 'inputs')
    robot_poses = load_json_poses(os.path.join(inputs_dir, 'robot_poses.json'))
    camera_poses_raw = load_json_poses(os.path.join(inputs_dir, 'camera_poses_raw.json'))
    
    if len(robot_poses) != len(camera_poses_raw):
        print(f"  警告: 位姿数量不匹配! Robot: {len(robot_poses)}, Camera: {len(camera_poses_raw)}")
        min_len = min(len(robot_poses), len(camera_poses_raw))
        robot_poses = robot_poses[:min_len]
        camera_poses_raw = camera_poses_raw[:min_len]
        print(f"  使用前 {min_len} 个位姿")
    
    print("\n计算相对运动...")
    A_list = calculate_relative_transforms(robot_poses)
    B_list = calculate_relative_transforms(camera_poses_raw)
    
    print("\n运行 Tsai-Lenz 标定算法...")
    scale, X, errors_rot, errors_trans = tsai_lenz_scaled(A_list, B_list)
    
    print(f"\n{'─'*60}")
    print("标定结果:")
    print('─'*60)
    print(f"尺度因子 s: {scale:.6f}")
    print(f"\n手眼变换矩阵 X:")
    print(X)
    print(f"\n误差统计:")
    print(f"  旋转误差 - 均值: {errors_rot.mean():.6f}, 最大: {errors_rot.max():.6f}")
    print(f"  平移误差 - 均值: {errors_trans.mean():.6f}, 最大: {errors_trans.max():.6f}")
    
    return scale, X, errors_rot, errors_trans


def save_results(dataset_path, dataset_name, scale, X):
    outputs_dir = os.path.join(dataset_path, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    result_data = {
        'dataset_name': dataset_name,
        'algorithm': 'Tsai-Lenz with Scale Estimation',
        'scale_factor': float(scale),
        'hand_eye_transform': X.tolist(),
        'description': 'Hand-eye calibration result: X transforms from camera to robot end-effector'
    }
    
    output_file = os.path.join(outputs_dir, 'tsai_lenz_result.json')
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")


def main():
    print("="*70)
    print("Tsai-Lenz 手眼标定 - 从 organized_data 读取JSON数据")
    print("="*70)
    
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    organized_data_dir = os.path.join(project_folder, 'organized_data')
    
    if not os.path.exists(organized_data_dir):
        print(f"\n错误: 找不到数据目录: {organized_data_dir}")
        return
    
    datasets = [d for d in os.listdir(organized_data_dir) 
                if os.path.isdir(os.path.join(organized_data_dir, d))]
    
    datasets.sort()
    print(f"\n找到 {len(datasets)} 个数据集: {', '.join(datasets)}")
    
    for dataset_name in datasets:
        dataset_path = os.path.join(organized_data_dir, dataset_name)
        
        try:
            scale, X, errors_rot, errors_trans = process_dataset(dataset_path, dataset_name)
            save_results(dataset_path, dataset_name, scale, X)
        except Exception as e:
            print(f"\n处理 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("所有数据集处理完成!")
    print("="*70)


if __name__ == "__main__":
    main()
