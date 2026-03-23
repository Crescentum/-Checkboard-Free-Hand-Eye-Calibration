"""
评估标定结果
从 organized_data/ 读取ground_truth并与outputs/中的结果对比
"""

import os
import json
import numpy as np


def load_json_matrix(filepath, is_ground_truth=False):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if is_ground_truth:
        # Ground truth使用T_computed
        if 'T_computed' in data and 'matrix' in data['T_computed']:
            matrix = np.array(data['T_computed']['matrix'])
        else:
            raise KeyError(f"无法在ground truth中找到变换矩阵: {filepath}")
    else:
        if 'hand_eye_transform' in data:
            matrix = np.array(data['hand_eye_transform'])
        elif 'matrix' in data:
            matrix = np.array(data['matrix'])
        else:
            raise KeyError(f"无法找到变换矩阵: {filepath}")
    
    return matrix


def rotation_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def evaluate_transform(T_gt, X_pred):
    Delta = np.linalg.inv(T_gt) @ X_pred
    R = Delta[:3, :3]
    t = Delta[:3, 3]
    
    return {
        "rot_deg": rotation_angle_deg(R),
        "trans_cm": float(np.linalg.norm(t)) * 100.0,
        "Delta": Delta
    }


def evaluate_dataset(dataset_path, dataset_name):
    print(f"\n{'='*60}")
    print(f"评估数据集: {dataset_name}")
    print('='*60)
    
    gt_path = os.path.join(dataset_path, 'ground_truth', 'hand_eye_transform.json')
    if not os.path.exists(gt_path):
        print(f"  警告: 找不到ground truth: {gt_path}")
        return None
    
    T_gt = load_json_matrix(gt_path, is_ground_truth=True)
    print(f"  加载 ground truth (T_computed)")
    
    results = {}
    algorithms = [
        ('tsai_lenz', 'tsai_lenz_result.json'),
        ('park_martin', 'park_martin_result.json')
    ]
    
    outputs_dir = os.path.join(dataset_path, 'outputs')
    
    for alg_name, filename in algorithms:
        result_path = os.path.join(outputs_dir, filename)
        
        if not os.path.exists(result_path):
            print(f"  {alg_name}: 结果文件不存在")
            continue
        
        try:
            X_pred = load_json_matrix(result_path)
            eval_result = evaluate_transform(T_gt, X_pred)
            
            results[alg_name] = {
                'rot_deg': eval_result['rot_deg'],
                'trans_cm': eval_result['trans_cm']
            }
            
            print(f"  {alg_name:12s}: 旋转误差 = {eval_result['rot_deg']:7.4f}°, "
                  f"平移误差 = {eval_result['trans_cm']:7.4f} cm")
            
        except Exception as e:
            print(f"  {alg_name}: 评估失败 - {e}")
    
    return results


def main():
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    organized_data_dir = os.path.join(project_folder, 'organized_data')
    
    if not os.path.exists(organized_data_dir):
        print(f"\n错误: 找不到数据目录: {organized_data_dir}")
        return
    
    datasets = [d for d in os.listdir(organized_data_dir) 
                if os.path.isdir(os.path.join(organized_data_dir, d))]
    
    datasets.sort()
    print(f"\n找到 {len(datasets)} 个数据集")
    
    all_results = {}
    for dataset_name in datasets:
        dataset_path = os.path.join(organized_data_dir, dataset_name)
        
        try:
            results = evaluate_dataset(dataset_path, dataset_name)
            if results:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"\n评估 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    alg_stats = {}
    
    for dataset_name, dataset_results in all_results.items():
        for alg_name, metrics in dataset_results.items():
            if alg_name not in alg_stats:
                alg_stats[alg_name] = {'rot': [], 'trans': []}
            alg_stats[alg_name]['rot'].append(metrics['rot_deg'])
            alg_stats[alg_name]['trans'].append(metrics['trans_cm'])
    
    for alg_name, stats in sorted(alg_stats.items()):
        if stats['rot']:
            avg_rot = np.mean(stats['rot'])
            avg_trans = np.mean(stats['trans'])
            std_rot = np.std(stats['rot'])
            std_trans = np.std(stats['trans'])
            count = len(stats['rot'])
            
            print(f"\n{alg_name.upper()} ({count} 个数据集):")
            print(f"  旋转误差:  {avg_rot:7.4f}° ± {std_rot:6.4f}°")
            print(f"  平移误差:  {avg_trans:7.4f} cm ± {std_trans:6.4f} cm")
    
if __name__ == "__main__":
    main()
