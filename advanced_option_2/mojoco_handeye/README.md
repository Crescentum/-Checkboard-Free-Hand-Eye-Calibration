# 环境

## FoundationPose
在FoundationPose文件夹中，放置所有[FoundationPose](https://github.com/NVlabs/FoundationPose)代码，环境配置也请参考[FoundationPose](https://github.com/NVlabs/FoundationPose)。

最终的文件结构是：
```text
FoundationPose/
├── foundationPose_handeye.py
├── 所有FoundationPose源码
```

## Mujoco
```bash
cd mojoco_handeye
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```
修改mujoco_menagerie/franka_emika_panda/panda.xml：
```txt
<body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
<inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
<joint name="finger_joint2" class="finger"/>
<geom mesh="finger_0" material="off_white" class="visual"/>
<geom mesh="finger_1" material="black" class="visual"/>
<geom mesh="finger_0" class="collision"/>
<geom class="fingertip_pad_collision_1"/>
<geom class="fingertip_pad_collision_2"/>
<geom class="fingertip_pad_collision_3"/>
<geom class="fingertip_pad_collision_4"/>
<geom class="fingertip_pad_collision_5"/>
</body>

# 新增
<camera name="end_effector_cam" pos="0 0 0.15" euler="-3.14 0 -1.57" fovy="60"/>
<site name="attachment_site" pos="0 0 0" rgba="1 0 0 1" size="0.01"/>
```

替换mujoco_menagerie/franka_emika_panda/scene.xml为：
```text
<mujoco model="panda scene">
  <include file="panda.xml"/>

  <statistic center="0.3 0 0.4" extent="1"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <!-- <asset>
    <texture name="can_texture" type="2d" builtin="flat" rgb1="0.8 0.1 0.1" rgb2="1 1 1" 
             width="200" height="200" mark="cross" markrgb="1 1 1"/>
    <material name="can_mat" texture="can_texture" texrepeat="2 1" specular="0.3" shininess="0.5"/>
  </asset> -->

  <asset>
    <texture name="obj_texture" type="2d" file="/home/cjy/mojoco_handeye/mesh/texture_map.png"/>
    <material name="obj_material" texture="obj_texture"/>
    <mesh name="obj_mesh" file="/home/cjy/mojoco_handeye/mesh/textured_simple.obj"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- <body name="target_object" pos="2 1 0.08">
      <joint name="target_free" type="free"/>
      <geom name="target_geom" type="cylinder" size="0.06 0.08" material="can_mat" density="500" condim="4"/>
    </body> -->
  </worldbody>

  <worldbody>
    <body name="target_body" pos="1 1 0.08" euler="0 0 180">
      <freejoint name="target_free"/>
      <geom type="mesh" mesh="obj_mesh" material="obj_material"/>
    </body>
  </worldbody>

</mujoco>
```

conda环境安装：
```bash
pip list --format=freeze > requirements.txt
```

# 运行

## 采集手眼标定的数据
```bash
cd mojoco_handeye
python mojoco_foundationPose.py
```
按p键自动采集，按q键退出。

## 6D姿态估计和手眼标定
将mesh文件夹复制到foundation_pose_data文件夹下。
```bash
sh ./handeye_main.sh
```

