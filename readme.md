# dataset位置
`./organized_data`
# 代码位置
`./calibration`
# 安装依赖
(按需)
```
pip install numpy
pip install scipy
pip install numba
```
# 运行方法
(clone后进入主目录,直接运行以下步骤即可)
```
cd calibration
python tsai_lenz_calibration.py
python park_martin_calibration.py
cd ../eval
python evaluate_calibration.py
python visualize_trajectories.py
```