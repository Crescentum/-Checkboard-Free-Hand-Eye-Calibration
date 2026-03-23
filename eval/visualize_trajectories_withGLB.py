import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import trimesh
import pyrender


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_camera_poses_in_base(robot_poses, camera_poses_raw, hand_eye_transform, scale_factor):
    """
    用第一帧计算标定板系到基座系的变换：
       T_base_calib = T_base_end[0] @ X @ inv(T_calib_camera[0])
    """
    X = np.array(hand_eye_transform)
    
    T_calib_camera_0_raw = np.array(camera_poses_raw[0])
    T_calib_camera_0 = T_calib_camera_0_raw.copy()
    T_calib_camera_0[:3, 3] *= scale_factor
    
    T_base_end_0 = np.array(robot_poses[0])
    T_base_calib = T_base_end_0 @ X @ np.linalg.inv(T_calib_camera_0)
    
    camera_poses = []
    for camera_pose_raw in camera_poses_raw:
        T_calib_camera = np.array(camera_pose_raw).copy()
        T_calib_camera[:3, 3] *= scale_factor
        
        T_base_camera = T_base_calib @ T_calib_camera
        camera_poses.append(T_base_camera)
    
    return camera_poses


def extract_positions(poses):
    positions = []
    for pose in poses:
        pose_array = np.array(pose)
        position = pose_array[:3, 3]
        positions.append(position)
    return np.array(positions)


def plot_trajectories(robot_positions, camera_positions, dataset_name, algorithm, save_path=None):

    fig = plt.figure(figsize=(14, 10))
    
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], robot_positions[:, 2],
            'b-o', linewidth=2, markersize=6, label='End-effector trajectory', alpha=0.7)
    
    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
            'r-s', linewidth=2, markersize=6, label='Camera trajectory', alpha=0.7)
    
    ax.scatter(robot_positions[0, 0], robot_positions[0, 1], robot_positions[0, 2],
              c='blue', s=150, marker='*', label='End-effector start', edgecolors='black', linewidths=1.5)
    ax.scatter(camera_positions[0, 0], camera_positions[0, 1], camera_positions[0, 2],
              c='red', s=150, marker='*', label='Camera start', edgecolors='black', linewidths=1.5)
    
    ax.scatter(robot_positions[-1, 0], robot_positions[-1, 1], robot_positions[-1, 2],
              c='blue', s=150, marker='X', label='End-effector end', edgecolors='black', linewidths=1.5)
    ax.scatter(camera_positions[-1, 0], camera_positions[-1, 1], camera_positions[-1, 2],
              c='red', s=150, marker='X', label='Camera end', edgecolors='black', linewidths=1.5)
    
    ax.scatter(0, 0, 0, c='green', s=200, marker='^', 
              label='Robot base (origin)', edgecolors='black', linewidths=2)
    
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    
    title = f'Trajectories in Robot Base Frame\nDataset: {dataset_name} | Algorithm: {algorithm}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    all_positions = np.vstack([robot_positions, camera_positions])
    max_range = np.array([
        all_positions[:, 0].max() - all_positions[:, 0].min(),
        all_positions[:, 1].max() - all_positions[:, 1].min(),
        all_positions[:, 2].max() - all_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图片已保存到: {Path(save_path).name}")
    
    return fig


def plot_distance_between_trajectories(robot_positions, camera_positions, dataset_name, algorithm, save_path=None):
    distances = []
    for i in range(len(robot_positions)):
        dist = np.linalg.norm(camera_positions[i] - robot_positions[i])
        distances.append(dist)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(len(distances)), distances, 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=np.mean(distances), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(distances):.4f} m')
    
    ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distance between End-effector and Camera\nDataset: {dataset_name} | Algorithm: {algorithm}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  距离图已保存到: {Path(save_path).name}")
    
    return fig


def visualize_with_trimesh(robot_positions, camera_positions, camera_poses_raw, 
                          scale_factor, hand_eye_transform, robot_poses,
                          dataset_name, algorithm, glb_file=None, save_path=None):
    """
    保持GLB点云标定板系不变。
    将机械臂和相机轨迹反向变换到相机raw坐标系中
    """
    scene = trimesh.Scene()
    
    X = np.array(hand_eye_transform)
    T_calib_camera_0_raw = np.array(camera_poses_raw[0])
    T_calib_camera_0 = T_calib_camera_0_raw.copy()
    T_calib_camera_0[:3, 3] *= scale_factor
    T_base_end_0 = np.array(robot_poses[0])
    T_base_calib = T_base_end_0 @ X @ np.linalg.inv(T_calib_camera_0)
    T_calib_base = np.linalg.inv(T_base_calib)
    
    robot_positions_calib = []
    camera_positions_calib = []
    
    for i in range(len(robot_positions)):
        robot_pos_homo = np.append(robot_positions[i], 1)
        robot_pos_calib = (T_calib_base @ robot_pos_homo)[:3]
        robot_positions_calib.append(robot_pos_calib)
        
        camera_pos_homo = np.append(camera_positions[i], 1)
        camera_pos_calib = (T_calib_base @ camera_pos_homo)[:3]
        camera_positions_calib.append(camera_pos_calib)
    
    robot_positions_calib = np.array(robot_positions_calib)
    camera_positions_calib = np.array(camera_positions_calib)
    
    robot_positions_calib = np.array(robot_positions_calib)
    camera_positions_calib = np.array(camera_positions_calib)
    
    robot_path = trimesh.load_path(robot_positions_calib)
    robot_path.colors = np.array([[0, 0, 255, 255]] * len(robot_path.entities))
    scene.add_geometry(robot_path)
    
    for pos in robot_positions_calib[::2]:
        sphere = trimesh.creation.icosphere(radius=0.008)
        sphere.visual.vertex_colors = [0, 0, 255, 255]
        sphere.apply_translation(pos)
        scene.add_geometry(sphere)
    
    camera_path = trimesh.load_path(camera_positions_calib)
    camera_path.colors = np.array([[255, 0, 0, 255]] * len(camera_path.entities))
    scene.add_geometry(camera_path)
    
    for pos in camera_positions_calib[::2]:
        sphere = trimesh.creation.icosphere(radius=0.008)
        sphere.visual.vertex_colors = [255, 0, 0, 255]
        sphere.apply_translation(pos)
        scene.add_geometry(sphere)
    
    start_sphere_robot = trimesh.creation.icosphere(radius=0.02)
    start_sphere_robot.visual.vertex_colors = [0, 0, 200, 255]
    start_sphere_robot.apply_translation(robot_positions_calib[0])
    scene.add_geometry(start_sphere_robot)
    
    start_sphere_camera = trimesh.creation.icosphere(radius=0.02)
    start_sphere_camera.visual.vertex_colors = [200, 0, 0, 255]
    start_sphere_camera.apply_translation(camera_positions_calib[0])
    scene.add_geometry(start_sphere_camera)
    
    end_box_robot = trimesh.creation.box(extents=[0.03, 0.03, 0.03])
    end_box_robot.visual.vertex_colors = [0, 0, 255, 255]
    end_box_robot.apply_translation(robot_positions_calib[-1])
    scene.add_geometry(end_box_robot)
    
    end_box_camera = trimesh.creation.box(extents=[0.03, 0.03, 0.03])
    end_box_camera.visual.vertex_colors = [255, 0, 0, 255]
    end_box_camera.apply_translation(camera_positions_calib[-1])
    scene.add_geometry(end_box_camera)
    
    axis_length = 0.15
    axis_radius = 0.003
    x_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
    x_axis.visual.vertex_colors = [255, 0, 0, 255]
    x_axis.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    x_axis.apply_translation([axis_length/2, 0, 0])
    scene.add_geometry(x_axis)
    y_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
    y_axis.visual.vertex_colors = [0, 255, 0, 255]
    y_axis.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    y_axis.apply_translation([0, axis_length/2, 0])
    scene.add_geometry(y_axis)
    z_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
    z_axis.visual.vertex_colors = [0, 0, 255, 255]
    z_axis.apply_translation([0, 0, axis_length/2])
    scene.add_geometry(z_axis)
    
    if glb_file and Path(glb_file).exists():
        try:
            glb_scene = trimesh.load(str(glb_file))
            
            if isinstance(glb_scene, trimesh.Scene):
                for name, geom in glb_scene.geometry.items():
                    scene.add_geometry(geom, node_name=f'glb_{name}')
            else:
                scene.add_geometry(glb_scene, node_name='glb_mesh')
            
            print(f"  已加载GLB场景: {Path(glb_file).name}")
        except Exception as e:
            print(f"  加载GLB文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"  正在显示场景...（关闭窗口以继续）")
    print(f"  坐标系统：标定板系（相机raw坐标系）")
    scene.show()


def process_dataset(dataset_folder, algorithm='park_martin', show_plots=False, use_open3d=True, glb_file=None):
    """
    处理单个数据集
    """
    dataset_name = Path(dataset_folder).name
    
    robot_poses_file = Path(dataset_folder) / 'inputs' / 'robot_poses.json'
    camera_poses_raw_file = Path(dataset_folder) / 'inputs' / 'camera_poses_raw.json'
    result_file = Path(dataset_folder) / 'outputs' / f'{algorithm}_result.json'
    
    robot_data = load_json(str(robot_poses_file))
    camera_data = load_json(str(camera_poses_raw_file))
    result_data = load_json(str(result_file))
    
    robot_poses = robot_data['poses']
    camera_poses_raw = camera_data['poses']
    hand_eye_transform = result_data['hand_eye_transform']
    scale_factor = result_data['scale_factor']
    
    camera_poses = compute_camera_poses_in_base(robot_poses, camera_poses_raw, 
                                               hand_eye_transform, scale_factor)
    
    robot_positions = extract_positions(robot_poses)
    camera_positions = extract_positions(camera_poses)
    
    vis_folder = Path(dataset_folder) / 'visualizations'
    vis_folder.mkdir(exist_ok=True)
    
    trajectory_path = vis_folder / f'{algorithm}_trajectories.png'
    fig1 = plot_trajectories(robot_positions, camera_positions, 
                             dataset_name, algorithm, str(trajectory_path))
    
    distance_path = vis_folder / f'{algorithm}_distance.png'
    fig2 = plot_distance_between_trajectories(robot_positions, camera_positions,
                                              dataset_name, algorithm, str(distance_path))
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    if use_open3d:
        trimesh_screenshot_path = vis_folder / f'{algorithm}_trimesh_with_glb.png'
        visualize_with_trimesh(robot_positions, camera_positions, camera_poses_raw,
                            scale_factor, hand_eye_transform, robot_poses,
                            dataset_name, algorithm, glb_file, str(trimesh_screenshot_path))
    
    distances = [np.linalg.norm(camera_positions[i] - robot_positions[i]) 
                for i in range(len(robot_positions))]
    
    stats = {
        'dataset_name': dataset_name,
        'algorithm': algorithm,
        'num_frames': len(robot_positions),
        'robot_trajectory': {
            'min': robot_positions.min(axis=0).tolist(),
            'max': robot_positions.max(axis=0).tolist(),
            'mean': robot_positions.mean(axis=0).tolist(),
            'range': (robot_positions.max(axis=0) - robot_positions.min(axis=0)).tolist()
        },
        'camera_trajectory': {
            'min': camera_positions.min(axis=0).tolist(),
            'max': camera_positions.max(axis=0).tolist(),
            'mean': camera_positions.mean(axis=0).tolist(),
            'range': (camera_positions.max(axis=0) - camera_positions.min(axis=0)).tolist()
        },
        'end_effector_to_camera_distance': {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances))
        }
    }
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化')
    parser.add_argument('--glb', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=['park_martin', 'tsai_lenz', 'all'])
    parser.add_argument('--no-open3d', action='store_true')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    organized_data_dir = project_dir / 'organized_data'
    
    print("=" * 70)
    if args.glb:
        print(f"GLB点云文件: {args.glb}")
    print("=" * 70)
    
    dataset_folders = sorted([d for d in organized_data_dir.iterdir() 
                             if d.is_dir() and not d.name.startswith('.')])
    
    if args.dataset:
        dataset_folders = [d for d in dataset_folders if d.name == args.dataset]
        if not dataset_folders:
            print(f"错误: 未找到数据集 '{args.dataset}'")
            return
    
    all_stats = []
    
    algorithms = ['park_martin', 'tsai_lenz'] if args.algorithm == 'all' else [args.algorithm]
    
    for algorithm in algorithms:
        print(f"\n{'='*70}")
        print(f"算法: {algorithm.upper()}")
        print(f"{'='*70}")
        
        for dataset_folder in dataset_folders:
            dataset_name = dataset_folder.name
            print(f"\n处理数据集: {dataset_name}")
            
            try:
                stats = process_dataset(str(dataset_folder), algorithm, 
                                      show_plots=False, 
                                      use_open3d=not args.no_open3d,
                                      glb_file=args.glb)
                all_stats.append(stats)
                
                print(f"  帧数: {stats['num_frames']}")
                print(f"  末端到相机距离 - 平均: {stats['end_effector_to_camera_distance']['mean']:.4f} m, "
                      f"标准差: {stats['end_effector_to_camera_distance']['std']:.6f} m")
                
            except Exception as e:
                print(f"  错误: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    stats_file = organized_data_dir / 'trajectory_visualization_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'description': '可视化统计',
            'generated_date': '2025-12-17',
            'statistics': all_stats
        }, f, indent=2)


if __name__ == '__main__':
    main()
