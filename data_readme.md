## 📊 数据构成简述

### 🗂️ 六个数据集来源

所有数据都来自同一个实验采集，只是场景不同：

| 数据集名称 | 场景 | 位姿数 |
|----------|------|-------|
| 7obj_4cluster_15 | 7个物体，4cluster采样 | 15个 |
| 7obj_divangs_15 | 7个物体，不同角度采样 | 15个 |
| 8obj_4cluster_15 | 8个物体，4cluster采样 | 15个 |
| 8obj_divangs_15 | 8个物体，不同角度采样 | 15个 |
| shelf_4cluster_15 | 货架场景，4cluster采样 | 15个 |
| shelf_divangs_15 | 货架场景，不同角度采样 | 15个 |

### 📥 数据来源拆解

每个数据集有**两个源文件**：

#### 1. **主文件** (`xxx.pth`)
包含：
- ✅ **robot_poses** (`eef_poses`) - 机器人末端执行器位姿（精确已知）
- ✅ **camera_poses_scaled** (`poses`) - **DUST3R输出 + 尺度优化后**的相机位姿
- 📍 点云、颜色等附加数据

#### 2. **Raw文件** (`xxx_raw.pth`)
包含：
- 🔵 **camera_poses_raw** (`poses`) - **DUST3R直接输出**（未缩放，有尺度模糊）

### 🎯 我们的标定使用了什么

#### 标定算法的**输入** (inputs/)：
```
1. robot_poses.json          ← 来自机器人（精确）
2. camera_poses_raw.json     ← 来自DUST3R（未缩放）
```

#### 标定算法的**输出** (outputs/)：
```
- 手眼变换 X (4×4矩阵)
- 尺度因子 s
```

#### 用于**验证**的参考值 (ground_truth/)：
```
1. camera_poses_scaled.json  ← 来自DUST3R + 尺度优化（更准的参考）
2. hand_eye_transform.json   ← 从上述数据计算的真值X
```

### 🔄 数据关系图

```
实验采集
    │
    ├─> 机器人记录 ──────────> robot_poses (inputs/)
    │                          精确已知，作为标定基准
    │
    └─> 相机拍摄图像
            │
            └─> DUST3R处理
                    │
                    ├─> 直接输出 ──> camera_poses_raw (inputs/)
                    │                存在尺度模糊，需要标定求解
                    │
                    └─> + 尺度优化 ─> camera_poses_scaled (ground_truth/)
                                      "更准的参考"，用于验证标定结果
```

### 💡 关键点

1. **DUST3R的两个版本**：
   - `camera_poses_raw` - 原始输出，缺尺度（标定的**输入**）
   - `camera_poses_scaled` - 优化后，有尺度（标定的**参考真值**）

2. **你们的任务**：
   - 输入：机器人位姿 + DUST3R未缩放输出
   - 求解：手眼变换X + 尺度s
   - 验证：与DUST3R优化后的结果对比

3. **为什么用DUST3R优化版做真值**：
   - 因为它结合了多帧信息做了尺度优化
   - 比单纯的未缩放版本更接近真实尺度
   - 但仍有2-3cm的噪声（视觉估计的固有误差）