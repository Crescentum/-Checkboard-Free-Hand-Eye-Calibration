# Noise-Robust Hand-Eye Calibration

This project implements **robust hand-eye calibration methods** that can handle realistic noise and outliers in robot and camera pose measurements. It extends traditional calibration algorithms (Tsai-Lenz, Park-Martin) with robust optimization techniques including **RANSAC** and **M-Estimators**.

## Overview

Hand-eye calibration solves the transformation $X$ between a robot end-effector and a camera mounted on it:

$$A_i X = s \cdot X B_i$$

where:
- $A_i$: Robot motion (relative transform)
- $B_i$: Camera motion (relative transform)  
- $X$: Hand-eye transformation (unknown)
- $s$: Scale factor (for monocular cameras)

Traditional methods assume noise-free measurements, but real-world data contains:
- **Gaussian noise** in pose estimation
- **Outliers** from tracking failures or misalignment

My work addresses these challenges with robust optimization methods.

## Methods Implemented

### Robust Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **RANSAC** | Random sample consensus with scale sanity check | High outlier ratio (>30%) |
| **M-Estimator (Huber)** | Iteratively Reweighted Least Squares (IRLS) | Moderate noise, small datasets |

## Experimental Results

### Summary (6 datasets, 15 pose pairs each)

| Condition | Method | Rotation Error (°) | Translation Error (cm) |
|-----------|--------|-------------------|----------------------|
| **Clean** | Tsai-Lenz | 0.00 ± 0.00 | **0.73 ± 0.39** |
| **Clean** | Park-Martin | 0.00 ± 0.00 | 0.81 ± 0.27 |
| **Clean** | RANSAC | 0.00 ± 0.00 | 3.10 ± 2.82 |
| **Clean** | M-Estimator | 0.00 ± 0.00 | 3.57 ± 2.68 |
| **Noisy** | Tsai-Lenz | 0.23 ± 0.50 | 11.43 ± 8.90 |
| **Noisy** | Park-Martin | 0.00 ± 0.00 | 11.28 ± 8.90 |
| **Noisy** | RANSAC | 0.46 ± 1.03 | 7.48 ± 5.71 |
| **Noisy** | M-Estimator | 0.00 ± 0.00 | **6.31 ± 3.46** |

### Key Findings

1. **Clean Data**: Traditional methods (Tsai-Lenz, Park-Martin) achieve sub-centimeter accuracy (~0.7-0.8 cm)

2. **Noisy Data**: 
   - **M-Estimator achieves 44% improvement** over traditional methods (6.31 cm vs 11.28 cm)
   - **RANSAC achieves 34% improvement** (7.48 cm vs 11.28 cm)

3. **Recommendation**:
   - Use **Tsai-Lenz** for clean, well-controlled data
   - Use **M-Estimator (Huber)** for noisy real-world data with small sample sizes
   - Use **RANSAC** when outlier ratio is known to be high (>30%)

## Project Structure

```
robust_hand_eye/
├── ransac_hand_eye.py      # Core robust calibration algorithms
│   ├── RANSAC implementation
│   ├── M-Estimator (Huber, Tukey, Cauchy)
│   └── IRLS optimization
├── experiment.py           # Full experiment pipeline
│   ├── Tsai-Lenz implementation
│   ├── Park-Martin (Daniilidis) implementation
│   └── Comparison framework
├── add_noise_to_pth.py     # Noise injection utilities
│   ├── SE(3) Gaussian noise
│   └── Outlier generation
├── evaluate_all.py         # Evaluation against ground truth
└── experiment_results.json # Experimental results
```

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy scipy torch opencv-python

# Optional (for acceleration)
pip install numba
```

### Environment

- Python 3.8+
- NumPy 1.20+
- PyTorch 1.9+
- OpenCV 4.5+
- SciPy 1.7+

## Usage

### 1. Prepare Data

Data should be in `.pth` format with the following structure:

```python
{
    'poses': torch.Tensor,      # Camera poses (N, 4, 4)
    'eef_poses': torch.Tensor,  # Robot end-effector poses (N, 4, 4)
}
```

### 2. Add Noise

```bash
python add_noise_to_pth.py \
    --data_dir data/dust3r_saved_output \
    --sigma_t_cam 0.005 \
    --sigma_r_cam 0.5 \
    --outlier_ratio 0.1
```
The output directory will be automatically named in a format like noisy_r0.5_t0.005_out10, making it easy to distinguish different noise settings.<br>

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sigma_t_robot` | 0.001 m | Robot translation noise std |
| `--sigma_r_robot` | 0.2° | Robot rotation noise std |
| `--sigma_t_cam` | 0.005 m | Camera translation noise std |
| `--sigma_r_cam` | 0.5° | Camera rotation noise std |
| `--outlier_ratio` | 0.1 | Fraction of outliers (0.0-1.0) |

### 3. Run Experiment

```bash
python experiment.py \
    --clean_dir data/dust3r_saved_output \
    --noisy_dir data/noisy_r0.5_t0.005_out10 \
    --results_dir results/robust_experiment
```


## API Reference

### RANSAC Hand-Eye Calibration

```python
from ransac_hand_eye import ransac_hand_eye, RANSACConfig

# Configure RANSAC
config = RANSACConfig(
    min_samples=3,              # Minimum samples per iteration
    max_iterations=1000,        # Maximum RANSAC iterations
    inlier_threshold=0.1,       # Error threshold for inliers
    scale_range=(0.5, 5.0),     # Scale search range
    scale_steps=2000,           # Scale grid resolution
    pre_estimate_scale=True,    # Enable scale sanity check
    verbose=False
)

# Run calibration
result = ransac_hand_eye(A_list, B_list, config)

print(f"Scale: {result.scale}")
print(f"Inlier ratio: {result.inlier_ratio * 100:.1f}%")
print(f"Hand-eye matrix:\n{result.X}")
```

### M-Estimator Calibration

```python
from ransac_hand_eye import m_estimator_with_scale_search, MEstimatorType

# Run M-Estimator with Huber loss
X, scale = m_estimator_with_scale_search(
    A_list, B_list,
    m_type=MEstimatorType.HUBER,
    scale_range=(0.1, 5.0),
    scale_steps=2000,
    irls_iter=10
)
```

## Algorithm Details

### RANSAC with Scale Sanity Check

Traditional RANSAC estimates scale from only 3 random samples, which can lead to catastrophic failures on small datasets. Our implementation:

1. **Pre-estimate scale** using all data with trimmed mean (robust to outliers)
2. **Sanity check**: Reject samples with scale outside ±50% of pre-estimate
3. **Standard RANSAC** for outlier rejection with fixed scale bounds

```
Input: Motion pairs (A_i, B_i), config
1. coarse_scale = estimate_scale_robust(all data)
2. scale_bounds = [0.5 * coarse_scale, 1.5 * coarse_scale]
3. for iteration in 1..max_iterations:
     a. sample 3 random motion pairs
     b. estimate scale_sample from sample
     c. if scale_sample outside bounds: continue  # SANITY CHECK
     d. estimate X from sample
     e. count inliers
     f. update best model if better
4. Refine with all inliers
Output: X, scale, inlier_mask
```

### M-Estimator (IRLS)

Iteratively Reweighted Least Squares with robust weight functions:

| **Huber** | $$w(r) = \min(1, k / |r|)$$  

Linear for large errors.

| **Tukey** | 
$$ w(r) = (1-(r/c)^2)^2$$

if $|r| \leq c$, else 0.
Hard rejection.

| **Cauchy** | 
$$
w(r) = 1/(1+(r/c)^2)
$$

Smooth, heavy-tailed.

Scale estimation uses **Median Absolute Deviation (MAD)**:

$$\sigma = 1.4826 \cdot \text{median}(|r_i - \text{median}(r)|)$$

## Noise Model

The noise injection follows an SE(3) perturbation model:

```python
T_noisy = T @ δT

where:
    δR = exp(so(3), ω)    # ω ~ N(0, σ_r²)
    δt ~ N(0, σ_t²)
```

This ensures proper SE(3) geometry (rotation remains in SO(3)).

