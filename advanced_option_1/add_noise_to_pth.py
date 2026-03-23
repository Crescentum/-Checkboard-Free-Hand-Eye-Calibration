import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple

project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_folder)


# =============================================================================
# SE(3) Noise Utilities
# =============================================================================

def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0]
    ])


def so3_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w).reshape(3,)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    K = hat(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def add_pose_noise(T: np.ndarray, sigma_t: float, sigma_r_deg: float) -> np.ndarray:

    T_noisy = T.copy()
    
    sigma_r = np.deg2rad(sigma_r_deg)
    w = np.random.normal(0.0, sigma_r, size=(3,))
    dR = so3_exp(w)
    T_noisy[:3, :3] = T[:3, :3] @ dR
    
    dt = np.random.normal(0.0, sigma_t, size=(3,))
    T_noisy[:3, 3] = T[:3, 3] + dt
    
    return T_noisy


def add_noise_to_sequence(poses: np.ndarray, sigma_t: float, sigma_r_deg: float) -> np.ndarray:

    poses_noisy = np.array(poses, dtype=np.float64, copy=True)
    for i in range(len(poses)):
        poses_noisy[i] = add_pose_noise(poses[i], sigma_t, sigma_r_deg)
    return poses_noisy


def add_outliers_to_sequence(poses: np.ndarray, outlier_ratio: float, 
                              outlier_sigma_t: float, outlier_sigma_r_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(poses)
    num_outliers = int(n * outlier_ratio)
    
    # Randomly select outlier indices
    outlier_indices = np.random.choice(n, num_outliers, replace=False)
    outlier_mask = np.zeros(n, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    poses_out = poses.copy()
    for idx in outlier_indices:
        poses_out[idx] = add_pose_noise(poses[idx], outlier_sigma_t, outlier_sigma_r_deg)
    
    return poses_out, outlier_mask


# =============================================================================
# PTH File Processing
# =============================================================================

def load_pth_data(filepath: str) -> dict:
    data = torch.load(filepath, map_location='cpu')
    return data


def save_pth_data(filepath: str, data: dict):
    torch.save(data, filepath)


def process_pth_file(input_path: str, output_path: str,
                     sigma_t_robot: float, sigma_r_robot: float,
                     sigma_t_cam: float, sigma_r_cam: float,
                     outlier_ratio: float = 0.0,
                     outlier_sigma_t: float = 0.05,
                     outlier_sigma_r: float = 5.0):
    data = load_pth_data(input_path)
    
    if 'eef_poses' in data:
        eef_poses = data['eef_poses']
        if isinstance(eef_poses, torch.Tensor):
            eef_poses = eef_poses.numpy()
        
        eef_noisy = add_noise_to_sequence(eef_poses, sigma_t_robot, sigma_r_robot)

        if outlier_ratio > 0:
            eef_noisy, _ = add_outliers_to_sequence(
                eef_noisy, outlier_ratio, outlier_sigma_t, outlier_sigma_r
            )
        
        data['eef_poses'] = torch.from_numpy(eef_noisy)
        data['eef_poses_original'] = torch.from_numpy(eef_poses) 
    
    if 'poses' in data:
        cam_poses = data['poses']
        if isinstance(cam_poses, torch.Tensor):
            cam_poses = cam_poses.numpy()
        
        cam_noisy = add_noise_to_sequence(cam_poses, sigma_t_cam, sigma_r_cam)
        
        if outlier_ratio > 0:
            cam_noisy, outlier_mask = add_outliers_to_sequence(
                cam_noisy, outlier_ratio, outlier_sigma_t, outlier_sigma_r
            )
            data['outlier_mask'] = torch.from_numpy(outlier_mask)
        
        data['poses'] = torch.from_numpy(cam_noisy)
        data['poses_original'] = torch.from_numpy(cam_poses)  
    
    data['noise_params'] = {
        'sigma_t_robot': sigma_t_robot,
        'sigma_r_robot_deg': sigma_r_robot,
        'sigma_t_cam': sigma_t_cam,
        'sigma_r_cam_deg': sigma_r_cam,
        'outlier_ratio': outlier_ratio,
        'outlier_sigma_t': outlier_sigma_t,
        'outlier_sigma_r_deg': outlier_sigma_r,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_pth_data(output_path, data)
    print(f"  Saved: {output_path}")


def process_raw_pth_file(input_path: str, output_path: str,
                         sigma_t_cam: float, sigma_r_cam: float,
                         outlier_ratio: float = 0.0,
                         outlier_sigma_t: float = 0.05,
                         outlier_sigma_r: float = 5.0):

    data = load_pth_data(input_path)
    
    if 'poses' in data:
        cam_poses = data['poses']
        if isinstance(cam_poses, torch.Tensor):
            cam_poses = cam_poses.numpy()
        
        cam_noisy = add_noise_to_sequence(cam_poses, sigma_t_cam, sigma_r_cam)
        
        if outlier_ratio > 0:
            cam_noisy, outlier_mask = add_outliers_to_sequence(
                cam_noisy, outlier_ratio, outlier_sigma_t, outlier_sigma_r
            )
            data['outlier_mask'] = torch.from_numpy(outlier_mask)
        
        data['poses'] = torch.from_numpy(cam_noisy)
        data['poses_original'] = torch.from_numpy(cam_poses)
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        cam_poses = data.numpy() if isinstance(data, torch.Tensor) else data
        cam_noisy = add_noise_to_sequence(cam_poses, sigma_t_cam, sigma_r_cam)
        
        if outlier_ratio > 0:
            cam_noisy, _ = add_outliers_to_sequence(
                cam_noisy, outlier_ratio, outlier_sigma_t, outlier_sigma_r
            )
        
        data = {'poses': torch.from_numpy(cam_noisy), 
                'poses_original': torch.from_numpy(cam_poses)}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_pth_data(output_path, data)
    print(f"  Saved: {output_path}")


# =============================================================================
# Batch Processing
# =============================================================================

def batch_add_noise(data_dir: str, output_dir: str,
                    sigma_t_robot: float, sigma_r_robot: float,
                    sigma_t_cam: float, sigma_r_cam: float,
                    outlier_ratio: float = 0.0):
    """
    Process all .pth files in a directory
    """
    print("=" * 60)
    print("Adding Noise to PTH Files")
    print("=" * 60)
    print(f"Input dir:  {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Robot noise: t={sigma_t_robot}m, r={sigma_r_robot}°")
    print(f"Camera noise: t={sigma_t_cam}m, r={sigma_r_cam}°")
    print(f"Outlier ratio: {outlier_ratio*100:.1f}%")
    print("=" * 60)
    
    all_files = os.listdir(data_dir)
    
    main_files = [f for f in all_files if f.endswith('.pth') and not f.endswith('_raw.pth')]
    
    for filename in main_files:
        print(f"\nProcessing: {filename}")
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            process_pth_file(
                input_path, output_path,
                sigma_t_robot, sigma_r_robot,
                sigma_t_cam, sigma_r_cam,
                outlier_ratio
            )
        except Exception as e:
            print(f"  Error: {e}")
    
    raw_files = [f for f in all_files if f.endswith('_raw.pth')]
    
    for filename in raw_files:
        print(f"\nProcessing: {filename}")
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            process_raw_pth_file(
                input_path, output_path,
                sigma_t_cam, sigma_r_cam,
                outlier_ratio
            )
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Noise injection complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Add noise to PTH pose files')
    
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(project_folder, 'data', 'dust3r_saved_output'),
                        help='Input directory with .pth files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_dir/noisy_rX_tY)')
    
    parser.add_argument('--sigma_t_robot', type=float, default=0.001,
                        help='Robot translation noise std (meters), default=0.001 (1mm)')
    parser.add_argument('--sigma_r_robot', type=float, default=0.2,
                        help='Robot rotation noise std (degrees), default=0.2')
    parser.add_argument('--sigma_t_cam', type=float, default=0.005,
                        help='Camera translation noise std (meters), default=0.005 (5mm)')
    parser.add_argument('--sigma_r_cam', type=float, default=0.5,
                        help='Camera rotation noise std (degrees), default=0.5')
    
    parser.add_argument('--outlier_ratio', type=float, default=0.1,
                        help='Ratio of outliers to add (0.0-1.0), default=0.1 (10%)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        noise_tag = f"noisy_r{args.sigma_r_cam}_t{args.sigma_t_cam}"
        if args.outlier_ratio > 0:
            noise_tag += f"_out{int(args.outlier_ratio*100)}"
        args.output_dir = os.path.join(os.path.dirname(args.data_dir), noise_tag)
    
    batch_add_noise(
        args.data_dir, args.output_dir,
        args.sigma_t_robot, args.sigma_r_robot,
        args.sigma_t_cam, args.sigma_r_cam,
        args.outlier_ratio
    )


if __name__ == "__main__":
    main()