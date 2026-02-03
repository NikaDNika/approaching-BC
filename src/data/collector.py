import os
os.environ["MUJOCO_GL"] = "egl" # 强制使用 EGL 后端

import time
import numpy as np
import torch.multiprocessing as mp
import mujoco
from src.env.scene_generator import SceneGenerator
from src.env.mujoco_env import MujocoSimEnv
from src.planning.planner import CuRoboWrapper 
from scipy.spatial.transform import Rotation as R
from src.env.assets_manager import AssetsManager
from curobo.geom.types import Mesh 

# --- 路径处理 ---
# 获取当前脚本所在目录 (src/data)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (DRP_FULL)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# 获取 XML 绝对路径
BASE_XML_PATH = os.path.join(PROJECT_ROOT, "configs", "mobile_panda.xml")

# --- 配置 ---
DATA_SAVE_DIR = "/media/j16/8deb00a4-cceb-e842-a899-55532424da473/dataset_v2"
TOTAL_EPISODES = 10000 
NUM_WORKERS = 4
SCENARIOS = ['living_room', 'kitchen', 'storage', 'corner']
CHUNK_SIZE = 10
N_POINTS = 4096 
GRASP_OFFSET = np.array([0, 0, 0.08]) 

IMG_WIDTH = 320 
IMG_HEIGHT = 240

# --- 姿态库 ---
GRASP_ORIENTATIONS = [
    [0, 1, 0, 0],           # Top-Down
    [1, 0, 0, 0],           # Top-Down Flip
    [0.707, 0.707, 0, 0],   # Forward
    [0, 0, 0.707, 0.707],   # Side
    [0.5, 0.5, 0.5, 0.5]    # Diagonal
]

def sample_base_position(target_pos_world, scenario_type):
    t_x, t_y = target_pos_world[0], target_pos_world[1]
    dist = np.random.uniform(0.55, 0.95)
    vec_len = np.linalg.norm([t_x, t_y])
    if vec_len < 0.01: vec_len = 1.0
    dir_x = t_x / vec_len
    dir_y = t_y / vec_len
    rot_noise = np.random.uniform(-0.5, 0.5) 
    cos_rot = np.cos(rot_noise)
    sin_rot = np.sin(rot_noise)
    back_x = -(dir_x * cos_rot - dir_y * sin_rot)
    back_y = -(dir_x * sin_rot + dir_y * cos_rot)
    base_x = t_x + back_x * dist
    base_y = t_y + back_y * dist
    base_to_target_angle = np.arctan2(t_y - base_y, t_x - base_x)
    base_theta = base_to_target_angle + np.random.uniform(-0.25, 0.25)
    if scenario_type == 'storage':
        base_x = min(base_x, 0.6)
    return np.array([base_x, base_y, base_theta])

def extract_mujoco_meshes(model, data, assets_db, link0_pos, link0_mat):
    curobo_meshes = []
    r_link0 = R.from_matrix(link0_mat)
    r_link0_inv = r_link0.inv()
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and (body_name.startswith("fix_") or body_name.startswith("fur_")):
            geom_id = -1
            for g_id in range(model.body_geomadr[i], model.body_geomadr[i] + model.body_geomnum[i]):
                if model.geom_type[g_id] == mujoco.mjtGeom.mjGEOM_MESH:
                    geom_id = g_id
                    break 
            if geom_id == -1: continue
            mesh_id = model.geom_dataid[geom_id]
            if mesh_id == -1: continue
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            face_adr = model.mesh_faceadr[mesh_id]
            face_num = model.mesh_facenum[mesh_id]
            raw_verts = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
            raw_faces = model.mesh_face[face_adr : face_adr + face_num].copy()
            scale_factor = 1.0
            for asset_key, asset_data in assets_db.assets.items():
                if asset_key in body_name or body_name.replace("fix_", "") in asset_key:
                    scale_factor = asset_data.get('scale_factor', 1.0)
                    break
            if scale_factor != 1.0: raw_verts *= scale_factor
            g_pos_local = model.geom_pos[geom_id]
            g_quat_local = model.geom_quat[geom_id]
            r_geom = R.from_quat([g_quat_local[1], g_quat_local[2], g_quat_local[3], g_quat_local[0]])
            g_mat_local = r_geom.as_matrix()
            verts_in_body_frame = g_pos_local + (g_mat_local @ raw_verts.T).T
            body_pos = data.xpos[i]
            body_mat = data.xmat[i].reshape(3, 3)
            verts_world = body_pos + (body_mat @ verts_in_body_frame.T).T
            verts_robot_frame = r_link0_inv.apply(verts_world - link0_pos)
            mesh_obj = Mesh(name=f"{body_name}_mesh", vertices=verts_robot_frame, faces=raw_faces, pose=[0, 0, 0, 1, 0, 0, 0])
            curobo_meshes.append(mesh_obj)
    return curobo_meshes

def worker_process(worker_id, scenario_type, target_count):
    # 1. 初始化随机种子
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
    worker_dir = os.path.join(DATA_SAVE_DIR, scenario_type)
    os.makedirs(worker_dir, exist_ok=True)
    xml_path = f"temp_scene_w{worker_id}.xml"
    
    import glob
    existing_files = glob.glob(os.path.join(worker_dir, f"traj_{scenario_type}_{worker_id}_*.npz"))
    start_count = 0
    if existing_files:
        try:
            indices = [int(os.path.basename(f).split('_')[-2]) for f in existing_files]
            if indices: start_count = max(indices) + 1
        except: pass
    success_count = start_count
    
    # 显式传入绝对路径
    if not os.path.exists(BASE_XML_PATH):
        print(f"[Worker {worker_id}] ERROR: Config file not found at {BASE_XML_PATH}")
        return

    generator = SceneGenerator(base_xml_path=BASE_XML_PATH)
    assets_db = AssetsManager()
    
    stats = {"total": 0, "planner_fail": 0}
    fail_reasons = {}

    print(f"[Worker {worker_id}] Started {scenario_type}. Goal: {target_count}")

    while success_count < target_count:
        stats["total"] += 1
        current_seed = np.random.randint(0, 1000000)
        
        try:
            target_pos_world = generator.generate_target(scenario_type, xml_path, seed=current_seed)
            if target_pos_world is None: continue
        except Exception: continue

        try:
            env = MujocoSimEnv(xml_path, width=IMG_WIDTH, height=IMG_HEIGHT)
        except Exception as e:
            continue

        robot_base_qpos = sample_base_position(target_pos_world, scenario_type)
        q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        start_q_full = np.concatenate([robot_base_qpos, q_home])
        env.data.qpos[:10] = start_q_full
        env.data.ctrl[3:10] = q_home
        env.data.ctrl[:3] = robot_base_qpos 
        
        mujoco.mj_forward(env.model, env.data)

        link0_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        link0_pos = env.data.xpos[link0_id]
        link0_mat = env.data.xmat[link0_id].reshape(3, 3)

        curobo_meshes = extract_mujoco_meshes(env.model, env.data, assets_db, link0_pos, link0_mat)

        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        try:
            planner = CuRoboWrapper(BASE_XML_PATH, initial_meshes=curobo_meshes)
        except Exception: continue

        traj = None
        final_status_msg = "NONE"
        final_quat = None # [新增] 记录成功的姿态
        
        for quat_list in GRASP_ORIENTATIONS:
            # 计算 Approach Vector
            r = R.from_quat([quat_list[1], quat_list[2], quat_list[3], quat_list[0]])
            approach_vec = r.apply([0, 0, 1])
            
            # Target = Object - 0.15 * Approach (使用 15cm 确保安全)
            real_target_world = target_pos_world - (0.12 * approach_vec)
            target_local = link0_mat.T @ (real_target_world - link0_pos)
            
            if np.linalg.norm(target_local) > 1.3: continue

            t, s = planner.plan_local(q_home, target_local, target_quat=quat_list)
            if t is not None:
                traj = t
                final_status_msg = s
                final_quat = quat_list # 记录
                break 
            else:
                final_status_msg = s
        
        del planner 
        
        if traj is None:
            stats["planner_fail"] += 1
            fail_reasons[final_status_msg] = fail_reasons.get(final_status_msg, 0) + 1
            if stats["planner_fail"] % 50 == 0:
                print(f"[Worker {worker_id}] Fails ({scenario_type}): {fail_reasons}")
            continue

        if len(traj) < CHUNK_SIZE: continue

        # --- 验证阶段 (Validation) ---
        env.data.qpos[:3] = robot_base_qpos
        env.data.qpos[3:10] = traj[-1]
        env.data.ctrl[3:10] = traj[-1]
        mujoco.mj_forward(env.model, env.data)
        
        # 1. 距离验证
        ee_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_viz")
        final_pos = env.data.xpos[ee_id]
        
        # 重新计算目标点 (因为循环里 real_target_world 变了，要用 final_quat 算出来的那个)
        r = R.from_quat([final_quat[1], final_quat[2], final_quat[3], final_quat[0]])
        expected_vec = r.apply([0, 0, 1])
        expected_target = target_pos_world - (0.12 * expected_vec)
        
        if np.linalg.norm(final_pos - expected_target) > 0.03:
            continue

        # 2. [核心新增] 方向一致性校验 (Anti-Clipping Check)
        # 逻辑：手腕到目标的向量 (Actual) 必须与 期望的进近方向 (Expected) 一致
        
        # 实际向量：从当前手腕指向物体中心
        actual_vec = target_pos_world - final_pos
        dist_to_obj = np.linalg.norm(actual_vec)
        
        if dist_to_obj > 0.001:
            actual_dir = actual_vec / dist_to_obj
            
            # 点积: dot(A, B) = cos(theta)
            # 如果方向一致，接近 1。如果反向(穿模跑到后面去了)，接近 -1。
            dot_prod = np.dot(actual_dir, expected_vec)
            
            # 阈值：0.5 对应 60度以内。
            # 如果小于 0.5，说明手腕虽然到了位置，但方向不对（比如侧着切入或反向切入），可能穿模。
            if dot_prod < 0.5:
                # print(f"[Worker {worker_id}] Filtered Clipping: dot={dot_prod:.2f}")
                continue
                
            # 双重保险：如果距离物体太近 (< 10cm)，即便方向对也可能穿模了
            if dist_to_obj < 0.10:
                continue

        # --- 数据采集 ---
        ep_data = {
            'pc_ar': [], 'q_arm': [], 'action': [],
            'cam_left_rgb': [], 'cam_left_depth': [],
            'cam_right_rgb': [], 'cam_right_depth': [],
            'cam_wrist_rgb': [], 'cam_wrist_depth': []
        }
        
        traj_len = len(traj)
        
        for t in range(traj_len):
            curr_q_arm = traj[t]
            env.data.qpos[:3] = robot_base_qpos
            env.data.qpos[3:10] = curr_q_arm
            env.data.ctrl[3:10] = curr_q_arm
            mujoco.mj_forward(env.model, env.data)
            
            start_idx = t + 1
            end_idx = start_idx + CHUNK_SIZE
            if end_idx <= traj_len:
                future_traj = traj[start_idx : end_idx]
            else:
                valid_traj = traj[start_idx:]
                pad_len = CHUNK_SIZE - len(valid_traj)
                last_pose = traj[-1]
                pad_traj = np.tile(last_pose, (pad_len, 1))
                if len(valid_traj) > 0:
                    future_traj = np.concatenate([valid_traj, pad_traj], axis=0)
                else:
                    future_traj = pad_traj
            action_chunk = future_traj - curr_q_arm
            
            images = env.get_images()
            curr_link0_pos = env.data.xpos[link0_id]
            curr_link0_mat = env.data.xmat[link0_id].reshape(3, 3)
            raw_pc = env.get_fused_pointcloud(curr_link0_pos, curr_link0_mat)
            
            if len(raw_pc) > N_POINTS:
                indices = np.random.choice(len(raw_pc), N_POINTS, replace=False)
                pc_sample = raw_pc[indices]
            else:
                pad = np.zeros((N_POINTS - len(raw_pc), 3))
                if len(raw_pc) > 0:
                    pc_sample = np.concatenate([raw_pc, pad], axis=0)
                else:
                    pc_sample = pad 
            
            ep_data['pc_ar'].append(pc_sample.astype(np.float16))
            ep_data['q_arm'].append(curr_q_arm)
            ep_data['action'].append(action_chunk)
            
            ep_data['cam_left_rgb'].append(images['cam_left_rgb'])
            ep_data['cam_left_depth'].append(images['cam_left_depth'])
            ep_data['cam_right_rgb'].append(images['cam_right_rgb'])
            ep_data['cam_right_depth'].append(images['cam_right_depth'])
            ep_data['cam_wrist_rgb'].append(images['cam_wrist_rgb'])
            ep_data['cam_wrist_depth'].append(images['cam_wrist_depth'])

        # 黑屏过滤
        is_bad_data = False
        for k in ['cam_left_rgb', 'cam_right_rgb', 'cam_wrist_rgb']:
            if len(ep_data[k]) > 0:
                mean_val = np.mean(ep_data[k][0])
                if mean_val < 10.0:
                    is_bad_data = True
                    break
        if is_bad_data: continue

        if len(ep_data['q_arm']) > 0:
            timestamp = time.time_ns()
            fname = f"traj_{scenario_type}_{worker_id}_{success_count}_{timestamp}.npz"
            save_dict = {
                'pc_ar': np.array(ep_data['pc_ar']),
                'q_arm': np.array(ep_data['q_arm']),
                'action': np.array(ep_data['action']),
                'target_pos': target_pos_world,
                'robot_base': robot_base_qpos, 
                'scene_xml': xml_content
            }
            for k in images.keys():
                arr = np.array(ep_data[k])
                if 'rgb' in k:
                    save_dict[k] = arr.astype(np.uint8)
                else:
                    save_dict[k] = arr.astype(np.float16)
            np.savez_compressed(os.path.join(worker_dir, fname), **save_dict)
            success_count += 1
            if success_count % 10 == 0:
                print(f"[Worker {worker_id}] {scenario_type}: {success_count}/{target_count} (Total Attempts: {stats['total']})")

    print(f"[Worker {worker_id}] Done.")

def main():
    try: mp.set_start_method('spawn', force=True)
    except: pass
    if not os.path.exists(DATA_SAVE_DIR): os.makedirs(DATA_SAVE_DIR)
    processes = []
    per_worker = TOTAL_EPISODES // NUM_WORKERS 
    
    for i, scenario in enumerate(SCENARIOS):
        p = mp.Process(target=worker_process, args=(i, scenario, per_worker))
        p.start()
        processes.append(p)
        
    for p in processes: p.join()

if __name__ == "__main__":
    main()