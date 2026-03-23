## 🤖 2. Checkerboard-Free Hand-Eye Calibration

[cite_start]Hand-eye calibration is a fundamental problem in robotics and 3D vision that aims to determine the precise geometric relationship between a robot's end-effector ("hand") and a camera ("eye") attached to it[cite: 2]. [cite_start]This transformation, typically represented as a rigid 6-DoF pose, is essential for tasks such as visual servoing, 3D reconstruction, object grasping, and human-robot interaction[cite: 3].

[cite_start]In practice, the calibration problem can be expressed as solving the equation $AX=XB$[cite: 4], where:
* [cite_start]$A$ represents the robot's motion between two configurations (in the robot base frame)[cite: 5].
* [cite_start]$B$ represents the corresponding camera motion (in the camera frame)[cite: 6].
* [cite_start]$X$ is the unknown homogeneous transformation between the hand and eye[cite: 7].

[cite_start]Despite its apparent simplicity, real-world calibration is complicated by sensor noise, imperfect motion estimation, time synchronization errors, and non-linear optimization challenges[cite: 8]. [cite_start]Accurate hand-eye calibration is therefore a critical prerequisite for high-precision robotic vision applications[cite: 9].

[cite_start]Traditionally, this calibration is performed using a known calibration target, such as a checkerboard[cite: 15]. [cite_start]While effective, this approach can be error-prone in real-world scenarios, e.g., when the checkerboard is not fully observed or even not available[cite: 16]. [cite_start]**Checkerboard-free methods** overcome this limitation by leveraging natural features present in the environment to estimate the camera's motion, enabling more flexible and autonomous calibration[cite: 17]. [cite_start]Nowadays, data-driven foundation models like DUSTt3R and VGGT can jointly estimate point maps and camera transformation, which make checkerboard-free methods feasible[cite: 18].

---

### Basic Requirements

* [cite_start]Understand classical methods such as Tsai-Lenz algorithm (1989) and Park-Martin dual quaternion method (1994)[cite: 20].
* [cite_start]Implement a hand-eye calibration algorithm that estimates the transformation $X$ from a set of paired motions $(A_i, B_i)$ while $B_i$ is ambiguous at scale (estimated from foundation models or COLMAP)[cite: 21, 22].
    * [cite_start]You can start with data provided by [https://github.com/tomtang502/arm_3d_reconstruction](https://github.com/tomtang502/arm_3d_reconstruction)[cite: 22].
    * [cite_start]Combine estimating camera motion and the hand-eye calibration algorithm and evaluate on provided synthetic data[cite: 22].
* [cite_start]**Metrics:** Rotation error (in degrees) and translation error (in centimeters)[cite: 23].
* [cite_start]Visualize the result by showing the transformed camera coordinate system relative to the robot arm in 3D (e.g., Open3D, pytransform3d, or matplotlib 3D)[cite: 24].

---

### Advanced Options

* [cite_start]**Noise-Robust Estimation:** Introduce realistic noise in robot and camera motions and design a robust optimization method (e.g., RANSAC, M-estimator) to handle outliers[cite: 26].
* [cite_start]**Deep Learning-Based Approach:** Implement or fine-tune a neural network that directly predicts the hand-eye transformation from paired motion trajectories or RGB-D frames[cite: 27].

---

### Reference

1.  [cite_start](IEEE T-RA 1989) Tsai & Lenz: A new technique for fully autonomous and efficient 3D robotics hand-eye calibration [cite: 29]
2.  [cite_start](IEEE T-RA 1994) Park & Martin: Robot sensor calibration: Solving $AX=XB$ on the Euclidean group [cite: 30]
3.  [cite_start](IROS 2017) Simultaneous hand-eye calibration and reconstruction [cite: 31]
4.  [cite_start](RA-L 2024) Unifying Representation and Calibration With 3D Foundation Models [cite: 32]
5.  [cite_start][https://sites.google.com/andrew.cmu.edu/deep-hand-eye-calibration](https://sites.google.com/andrew.cmu.edu/deep-hand-eye-calibration) [cite: 33]

---

Would you like to know more about the classical hand-eye calibration methods like the Tsai-Lenz algorithm?