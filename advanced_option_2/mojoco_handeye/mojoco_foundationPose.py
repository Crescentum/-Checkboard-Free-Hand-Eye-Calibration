import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
import os
import shutil

def euler_to_quat(roll, pitch, yaw):
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]

def get_4x4_matrix(pos, mat):
    T = np.eye(4)
    T[:3, :3] = mat.reshape(3, 3)
    T[:3, 3] = pos
    return T

T_mj_cv = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

T_world_cam_mj = np.array([
    [1,  0,  0, 0],
    [0,  0, -1, 0],
    [0,  1,  0, 0],
    [0,  0,  0, 1]
])

T_world_cv = T_world_cam_mj @ T_mj_cv

OBJECT_POSITION = [1, 1, 0.1]
OBJECT_EULER = [0, 0, 270]
SAVE_DIR = "foundation_pose_data"
SOURCE_MESH_DIR = "mesh"
FPS = 20
SAVE_INTERVAL = 1.0 / FPS

AUTO_RANDOM_MOTION = True
GOAL_HOLD_SEC = 1.0           
JOINT_DELTA_RANGE = 0.35      
MAX_CTRL_STEP = 0.06         
VIEW_MARGIN_PX = 40          
MIN_Z_M = 0.5              
MAX_Z_M = 1.5              

def reset_save_dir(dir_path, K_matrix, src_mesh):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    
    sub_dirs = ["rgb", "depth", "masks"]
    for sd in sub_dirs:
        os.makedirs(os.path.join(dir_path, sd), exist_ok=True)
    os.makedirs(os.path.join(dir_path, "ee_pose"), exist_ok=True)
    os.makedirs(os.path.join(dir_path, "obj_in_cam"), exist_ok=True)

    if os.path.exists(src_mesh):
        target_mesh_path = os.path.join(dir_path, "mesh")
        shutil.copytree(src_mesh, target_mesh_path)
    else:
        print(f"no mesh source {src_mesh}")
    
    np.savetxt(os.path.join(dir_path, "cam_K.txt"), K_matrix)

model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/scene.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

def get_intrinsics(model, renderer, cam_name="end_effector_cam"):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = model.cam_fovy[cam_id]
    h, w = renderer.height, renderer.width
    f = 0.5 * h / np.tan(fovy * np.pi / 360)
    return np.array([[f, 0, (w-1)/2], [0, f, (h-1)/2], [0, 0, 1]], dtype=np.float32)

K = get_intrinsics(model, renderer)

reset_save_dir(SAVE_DIR, K, SOURCE_MESH_DIR)

def _get_robot_hinge_joint_ids(model, qpos_limit=7):
    joint_ids = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        adr = int(model.jnt_qposadr[j])
        if 0 <= adr < qpos_limit:
            joint_ids.append(j)
    joint_ids.sort(key=lambda jid: int(model.jnt_qposadr[jid]))
    return joint_ids

def _get_joint_limits(model, joint_ids, default=(-2.9, 2.9)):
    lo = []
    hi = []
    for jid in joint_ids:
        r = model.jnt_range[jid]
        if np.isfinite(r[0]) and np.isfinite(r[1]) and r[0] < r[1]:
            lo.append(float(r[0]))
            hi.append(float(r[1]))
        else:
            lo.append(float(default[0]))
            hi.append(float(default[1]))
    return np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32)

_ARM_JOINT_IDS = _get_robot_hinge_joint_ids(model, qpos_limit=7)
_ARM_LO, _ARM_HI = _get_joint_limits(model, _ARM_JOINT_IDS)

def _world_to_cam_mj(data, cam_id, p_w):
    t_w = data.cam_xpos[cam_id].copy()
    R_w_cam = data.cam_xmat[cam_id].reshape(3, 3).copy()  
    R_cam_w = R_w_cam.T
    return R_cam_w @ (p_w - t_w)

def _cam_mj_to_cv(p_cam_mj):
    return (T_mj_cv[:3, :3] @ p_cam_mj.reshape(3, 1)).reshape(3)

def project_target_to_pixel(model, data, renderer, K, cam_name, target_body_name="target_body"):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)

    p_w = data.xpos[body_id].copy()
    p_cam_mj = _world_to_cam_mj(data, cam_id, p_w)
    p_cam_cv = _cam_mj_to_cv(p_cam_mj)
    x, y, z = float(p_cam_cv[0]), float(p_cam_cv[1]), float(p_cam_cv[2])
    if not np.isfinite(z) or z <= 0:
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return float(u), float(v), z

def is_target_in_view(model, data, renderer, K, cam_name, margin_px=VIEW_MARGIN_PX):
    proj = project_target_to_pixel(model, data, renderer, K, cam_name)
    if proj is None:
        return False
    u, v, z = proj
    if z < MIN_Z_M or z > MAX_Z_M:
        return False
    w, h = renderer.width, renderer.height
    return (margin_px <= u <= (w - 1 - margin_px)) and (margin_px <= v <= (h - 1 - margin_px))

def initialize_scene(model, data, target_pos):
    target_qpos = [0, 0, 0.811, -1.66, 0, 2.49, -2.26]
    # target_qpos = [0, 0.24, 0.811, -1.42, 0, 1.25, 0.695]
    # target_qpos = [0, 0, 0, 0, 0, 0, 0]

    data.qpos[0:7] = target_qpos
    if model.nu >= 7:
        data.ctrl[:7] = target_qpos
    obj_joint_adr = int(model.joint('target_free').qposadr)
    data.qpos[obj_joint_adr : obj_joint_adr+3] = target_pos
    data.qpos[obj_joint_adr+3 : obj_joint_adr+7] = euler_to_quat(OBJECT_EULER[0], OBJECT_EULER[1], OBJECT_EULER[2])
    mujoco.mj_forward(model, data)

initialize_scene(model, data, OBJECT_POSITION)

DEFAULT_ARM_QPOS = np.array(data.qpos[0:7].copy(), dtype=np.float32)

def clamp_arm_qpos(q, lo=_ARM_LO, hi=_ARM_HI):
    return np.minimum(np.maximum(q, lo), hi)

def sample_visible_arm_goal(model, data, renderer, K, cam_name, *,
                            base_q=None,
                            delta_range=JOINT_DELTA_RANGE,
                            max_tries=200,
                            margin_px=VIEW_MARGIN_PX):
    if base_q is None:
        base_q = data.qpos[0:7].copy()
    base_q = np.array(base_q, dtype=np.float32)

    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    act_backup = data.act.copy() if hasattr(data, 'act') and data.act is not None else None
    ctrl_backup = data.ctrl.copy() if model.nu > 0 else None

    try:
        for _ in range(max_tries):
            delta = np.random.uniform(-delta_range, delta_range, size=7).astype(np.float32)
            q_candidate = clamp_arm_qpos(base_q + delta)

            data.qpos[0:7] = q_candidate
            data.qvel[0:7] = 0
            if model.nu >= 7:
                data.ctrl[:7] = q_candidate
            mujoco.mj_forward(model, data)

            if is_target_in_view(model, data, renderer, K, cam_name, margin_px=margin_px):
                return q_candidate.copy()
    finally:
        data.qpos[:] = qpos_backup
        data.qvel[:] = qvel_backup
        if act_backup is not None:
            data.act[:] = act_backup
        if ctrl_backup is not None:
            data.ctrl[:] = ctrl_backup
        mujoco.mj_forward(model, data)

    return None

ee_site_name = "attachment_site"
cam_name = "end_effector_cam"

def get_static_transforms_for_opencv():
    T_mj_cv = np.array([
        [-1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, 1, 0],
        [0,  0,  0, 1]
    ])
    
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    
    T_w_ee = get_4x4_matrix(data.site_xpos[ee_id], data.site_xmat[ee_id])
    T_w_cam_mj = get_4x4_matrix(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    
    T_ee_cam_cv = (np.linalg.inv(T_w_ee) @ T_w_cam_mj) @ T_mj_cv

    return T_ee_cam_cv

def get_static_transforms():
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    
    T_w_ee = get_4x4_matrix(data.site_xpos[ee_id], data.site_xmat[ee_id])
    T_w_cam_mj = get_4x4_matrix(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    
    T_ee_cam = np.linalg.inv(T_w_ee) @ (T_w_cam_mj)

    return T_ee_cam

T_ee_cam_gt_cv = get_static_transforms_for_opencv()
np.savetxt(os.path.join(SAVE_DIR, "T_ee_cam_truth.txt"), T_ee_cam_gt_cv)

obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_body")
obj_pos = data.xpos[obj_id]
obj_mat = data.xmat[obj_id].reshape(3, 3)
T_w_obj = get_4x4_matrix(obj_pos, obj_mat)

is_recording = False
frame_count = 0
last_save_time = 0

arm_goal_q = data.qpos[0:7].copy()
arm_last_goal_time = time.time()
arm_last_visible_q = arm_goal_q.copy()
warned_no_actuators = False

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 60.0:
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        renderer.update_scene(data, camera="end_effector_cam")
        rgb_image = np.flipud(renderer.render())
        
        renderer.enable_depth_rendering()
        depth_image = np.flipud(renderer.render()) 

        renderer.disable_depth_rendering()

        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera View (Press P to Record)', rgb_bgr)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            is_recording = not is_recording
            if is_recording:
                print(f"recording to: {SAVE_DIR}")
                frame_count = 0

                if AUTO_RANDOM_MOTION:
                    if model.nu < 7:
                        if not warned_no_actuators:
                            warned_no_actuators = True
                    else:
                        cur_q = data.qpos[0:7].copy()
                        if is_target_in_view(model, data, renderer, K, cam_name):
                            arm_last_visible_q = cur_q.copy()
                        arm_goal_q = sample_visible_arm_goal(model, data, renderer, K, cam_name, base_q=cur_q)
                        if arm_goal_q is None:
                            arm_goal_q = cur_q.copy()
                        data.ctrl[:7] = cur_q
                        arm_last_goal_time = time.time()
            else:
                print("stop recording")

        if is_recording and AUTO_RANDOM_MOTION and model.nu >= 7:
            visible = is_target_in_view(model, data, renderer, K, cam_name)
            cur_q = data.qpos[0:7].copy()
            if visible:
                arm_last_visible_q = cur_q.copy()

            now = time.time()

            if not visible:
                arm_goal_q = arm_last_visible_q.copy()
                arm_last_goal_time = now
            else:
                if (now - arm_last_goal_time) >= GOAL_HOLD_SEC:
                    new_goal = sample_visible_arm_goal(model, data, renderer, K, cam_name, base_q=cur_q)
                    if new_goal is not None:
                        arm_goal_q = new_goal
                        arm_last_goal_time = now

            delta = arm_goal_q - data.ctrl[:7]
            step = np.clip(delta, -MAX_CTRL_STEP, MAX_CTRL_STEP)
            data.ctrl[:7] = data.ctrl[:7] + step

        if is_recording:
            current_time = time.time()
            if current_time - last_save_time >= SAVE_INTERVAL:
                cv2.imwrite(f"{SAVE_DIR}/rgb/{frame_count:06d}.png", rgb_bgr)
                
                depth_mm = (depth_image * 1000).astype(np.uint16)
                cv2.imwrite(f"{SAVE_DIR}/depth/{frame_count:06d}.png", depth_mm)

                ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
                T_w_ee = get_4x4_matrix(data.site_xpos[ee_id], data.site_xmat[ee_id])

                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                T_w_cam_mj = get_4x4_matrix(data.cam_xpos[cam_id], data.cam_xmat[cam_id])

                T_obj_cam_cv = (np.linalg.inv(T_w_cam_mj) @ T_w_obj)
                
                np.savetxt(f"{SAVE_DIR}/ee_pose/{frame_count:06d}.txt", T_w_ee)

                np.savetxt(f"{SAVE_DIR}/obj_in_cam/{frame_count:06d}.txt", T_obj_cam_cv)

                if frame_count == 0:
                    renderer.enable_segmentation_rendering()
                    seg = np.flipud(renderer.render())
                    renderer.disable_segmentation_rendering()
                    mask = (seg[:, :, 0] > 0).astype(np.uint8) * 255
                    cv2.imwrite(f"{SAVE_DIR}/masks/000000.png", mask)

                last_save_time = current_time
                frame_count += 1

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

cv2.destroyAllWindows()