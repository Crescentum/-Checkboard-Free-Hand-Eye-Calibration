import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from matplotlib.animation import FuncAnimation
import os

def draw_frame(ax, T, length=0.1, label=None):
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    ax.quiver(*origin, *x_axis, length=length, color='r', linewidth=1.5)
    ax.quiver(*origin, *y_axis, length=length, color='g', linewidth=1.5)
    ax.quiver(*origin, *z_axis, length=length, color='b', linewidth=1.5)

    if label:
        ax.text(*(origin + z_axis * length), label, fontweight='bold')

def visualize_frames_animation(dataset_folder, algorithm='park_martin', interval=200, save=False):
    dataset_path = Path(dataset_folder)
    robot_data = json.load(open(dataset_path / 'inputs/robot_poses.json'))
    camera_data = json.load(open(dataset_path / 'inputs/camera_poses_raw.json'))
    result_data = json.load(open(dataset_path / 'outputs' / f'{algorithm}_result.json'))

    T_end_cam = np.array(result_data['hand_eye_transform'])
    scale = result_data.get('scale_factor', 1.0)
    
    num_frames = len(robot_data['poses'])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        
        T_base_end = np.array(robot_data['poses'][frame_idx])
        
        T_obj_cam_raw = np.array(camera_data['poses'][frame_idx])
        T_obj_cam_raw[:3, 3] *= scale
        T_cam_obj = np.linalg.inv(T_obj_cam_raw)

        T_base_cam = T_base_end @ T_end_cam
        T_base_obj = T_base_cam @ T_cam_obj

        draw_frame(ax, np.eye(4), length=0.2, label='Base')
        draw_frame(ax, T_base_end, length=0.15, label='End-Effector')
        draw_frame(ax, T_base_cam, length=0.1, label='Camera')
        draw_frame(ax, T_base_obj, length=0.15, label='Object')

        pts = np.array([T_base_end[:3, 3], T_base_cam[:3, 3], T_base_obj[:3, 3]])
        ax.plot(pts[:,0], pts[:,1], pts[:,2], 'k--', alpha=0.3)

        ax.set_xlim([-0.5, 1.0])
        ax.set_ylim([-0.5, 1.0])
        ax.set_zlim([-0.1, 1.0])
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Frame: {frame_idx}/{num_frames} | Algorithm: {algorithm}')

    ani = FuncAnimation(fig, update, frames=range(num_frames), 
                        interval=interval, repeat=True)

    plt.show()
    
    if save:
        ani.save(os.path.join(dataset_folder, "visualizations/coord_system_viz.mp4"), writer='ffmpeg')

if __name__ == '__main__':
    visualize_frames_animation('./organized_data/7obj_4cluster_15', 'park_martin', interval=1000, save=True)
    