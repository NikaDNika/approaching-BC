import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import sys
import cv2

# --- 路径处理 ---
# 获取当前脚本所在目录 (src/evaluation)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (DRP_FULL)
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 定义配置目录路径
CONFIG_DIR = os.path.join(project_root, "configs")
BASE_XML_PATH = os.path.join(CONFIG_DIR, "mobile_panda.xml")

# --- 导入模块 ---
from src.models.impact import IMPACTPolicy
from src.env.mujoco_env import MujocoSimEnv
from src.env.scene_generator import SceneGenerator
from src.planning.planner import CuRoboWrapper
from src.data.collector import extract_mujoco_meshes, AssetsManager, sample_base_position

# --- 辅助函数 ---
def get_robot_pc(model, data, num_points=512):
    """ 实时计算机器人点云 (Pr) """
    robot_points = []
    for i in range(model.ngeom):
        pos = data.geom_xpos[i]
        size = model.geom_size[i]
        if np.all(size == 0):
            rbound = model.geom_rbound[i]
            size = np.array([rbound]*3) * 0.5
        
        pts = pos + np.random.uniform(-size[:3], size[:3], (15, 3))
        robot_points.append(pts)
        
    if not robot_points: return np.zeros((num_points, 3))
    robot_points = np.concatenate(robot_points, axis=0)
    
    if len(robot_points) >= num_points:
        idx = np.random.choice(len(robot_points), num_points, replace=False)
        return robot_points[idx]
    else:
        idx = np.random.choice(len(robot_points), num_points, replace=True)
        return robot_points[idx]

def normalize_depth(depth_map):
    """ 将深度图转为伪彩色以便可视化 """
    valid_mask = (depth_map > 0.01) & (depth_map < 3.0)
    if np.sum(valid_mask) == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    min_d, max_d = np.min(depth_map[valid_mask]), np.max(depth_map[valid_mask])
    norm = (depth_map - min_d) / (max_d - min_d + 1e-6)
    norm = np.clip(norm, 0, 1)
    norm_uint8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

# --- 主逻辑 ---
def evaluate(checkpoint_path, scenario='living_room', max_steps=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {checkpoint_path}...")
    
    # 1. 加载模型
    model = IMPACTPolicy(chunk_size=10, action_dim=7, state_dim=7, target_dim=7).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    except TypeError:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    if not os.path.exists(BASE_XML_PATH):
        raise FileNotFoundError(f"Config file not found: {BASE_XML_PATH}")
    
    # 循环尝试直到生成一个可行的场景
    max_retries = 10
    env = None
    
    for attempt in range(max_retries):
        print(f"\n--- Scenario Generation Attempt {attempt+1}/{max_retries} ---")
        
        gen = SceneGenerator(base_xml_path=BASE_XML_PATH)
        temp_scene_xml = "temp_eval_scene.xml"
        target_pos_world = gen.generate_target(scenario, temp_scene_xml, seed=int(time.time())+attempt)
        
        if target_pos_world is None: continue
        
        # 动态注入 Target Marker (Site) 到 XML 中
        with open(temp_scene_xml, 'r', encoding='utf-8') as f:
            xml_content = f.read()
            
        target_marker_xml = f"""
        <body name="target_marker" pos="{target_pos_world[0]} {target_pos_world[1]} {target_pos_world[2]}" mocap="true">
            <geom type="sphere" size="0.03" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
            <site name="target_site" size="0.05" rgba="0 1 0 0.3"/>
        </body>
        """
        xml_content = xml_content.replace('</worldbody>', f'{target_marker_xml}\n  </worldbody>')
        
        with open(temp_scene_xml, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        # 初始化环境
        # [关键] 设置 EGL 后端以支持多窗口/多线程渲染
        os.environ["MUJOCO_GL"] = "egl"
        try:
            # 分辨率设为 320x240 以便可视化
            env = MujocoSimEnv(temp_scene_xml, width=320, height=240)
        except Exception as e: 
            print(f"Env init failed: {e}")
            continue
        
        # 设置状态
        q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        robot_base = sample_base_position(target_pos_world, scenario)
        
        env.data.qpos[:3] = robot_base
        env.data.qpos[3:10] = q_home
        mujoco.mj_forward(env.model, env.data)
        
        # 计算 Goal Joint
        assets_db = AssetsManager()
        link0_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        link0_pos = env.data.xpos[link0_id]
        link0_mat = env.data.xmat[link0_id].reshape(3, 3)
        meshes = extract_mujoco_meshes(env.model, env.data, assets_db, link0_pos, link0_mat)
        
        planner = CuRoboWrapper(BASE_XML_PATH, initial_meshes=meshes)
        
        # 这里的 target_local 仅用于求解 IK 拿到 Goal Joint，具体姿态不重要
        target_local = link0_mat.T @ (target_pos_world - link0_pos)
        
        traj, _ = planner.plan_local(q_home, target_local)
        del planner
        
        if traj is not None:
            q_goal = traj[-1]
            print("Scenario ready!")
            break
        else:
            print("Target unreachable.")
            
    if env is None or traj is None:
        print("Failed to init scenario.")
        return

    print("Starting Rollout...")
    print("Press 'ESC' in viewer to exit.")
    
    # 4. 启动 Viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        curr_q = q_home
        
        # 预热渲染器 (防止第一帧黑屏)
        env.renderer.update_scene(env.data)
        
        for step in range(max_steps):
            if not viewer.is_running(): break
            
            step_start = time.time()
            
            # --- 1. 获取输入数据 (视觉) ---
            l0_p = env.data.xpos[link0_id]
            l0_m = env.data.xmat[link0_id].reshape(3, 3)
            raw_pc = env.get_fused_pointcloud(l0_p, l0_m)
            
            if len(raw_pc) >= 2048:
                idx = np.random.choice(len(raw_pc), 2048, replace=False)
                pc = raw_pc[idx]
            else:
                pad = np.zeros((2048-len(raw_pc), 3))
                if len(raw_pc)>0: pc = np.concatenate([raw_pc, pad])
                else: pc = pad
            
            robot_pc = get_robot_pc(env.model, env.data)
            
            # Tensor
            pc_t = torch.from_numpy(pc).float().unsqueeze(0).to(device)
            pr_t = torch.from_numpy(robot_pc).float().unsqueeze(0).to(device)
            q_t = torch.from_numpy(curr_q).float().unsqueeze(0).to(device)
            g_t = torch.from_numpy(q_goal).float().unsqueeze(0).to(device)
            
            # --- [新增] 实时显示三相机深度图 ---
            # 获取图像
            images = env.get_images()
            
            rgb_l = images['cam_left_rgb']
            dep_l = images['cam_left_depth']
            rgb_r = images['cam_right_rgb']
            dep_r = images['cam_right_depth']
            rgb_w = images['cam_wrist_rgb']
            dep_w = images['cam_wrist_depth']
            
            # 拼接显示
            # 注意：cv2.imshow 需要 BGR 格式
            row1 = np.hstack([cv2.cvtColor(rgb_l, cv2.COLOR_RGB2BGR), normalize_depth(dep_l)])
            row2 = np.hstack([cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR), normalize_depth(dep_r)])
            row3 = np.hstack([cv2.cvtColor(rgb_w, cv2.COLOR_RGB2BGR), normalize_depth(dep_w)])
            grid = np.vstack([row1, row2, row3])
            
            # 缩放以适应屏幕
            if grid.shape[0] > 800:
                grid = cv2.resize(grid, (0,0), fx=0.6, fy=0.6)
                
            cv2.imshow("Live Agent View (RGB-D)", grid)
            cv2.waitKey(1)
            
            # --- 2. 网络推理 ---
            with torch.no_grad():
                action_chunk = model(pc_t, pr_t, q_t, g_t)
                
            # --- 3. 执行动作 ---
            next_action = action_chunk[0, 0].cpu().numpy()
            target_q = curr_q + next_action
            
            # --- 4. 物理步进 (Physics Step) ---
            # 使用控制信号驱动，而非直接修改 qpos，以产生真实的物理碰撞
            env.data.ctrl[:3] = robot_base
            env.data.ctrl[3:10] = target_q
            
            # 运行 10 个物理微步 (约 20ms)
            for _ in range(10):
                mujoco.mj_step(env.model, env.data)
                
            viewer.sync()
            curr_q = env.data.qpos[3:10].copy()
            
            # --- 5. 检查距离 ---
            tcp_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
            tcp_pos = env.data.site_xpos[tcp_id]
            dist = np.linalg.norm(tcp_pos - target_pos_world)
            
            print(f"Step {step}: Dist = {dist:.4f} m", end="\r")
            
            if dist < 0.05:
                print(f"\nSuccess! Reached goal in {step} steps.")
                time.sleep(1.0)
                break
                
            # 简单的帧率控制
            time.sleep(0.02)
            
    cv2.destroyAllWindows()
    if os.path.exists("temp_eval_scene.xml"): os.remove("temp_eval_scene.xml")

if __name__ == "__main__":
    # 自动查找最新模型
    ckpt_dir = os.path.join(project_root, "checkpoints")
    exps = sorted(os.listdir(ckpt_dir)) if os.path.exists(ckpt_dir) else []
    
    if exps:
        # 找最新的一个包含 best_model.pth 的实验
        for exp in reversed(exps):
            best_model = os.path.join(ckpt_dir, exp, "best_model.pth")
            if os.path.exists(best_model):
                print(f"Evaluating: {best_model}")
                evaluate(best_model, scenario='living_room')
                break
        else:
            print("No best_model.pth found in any experiment.")
    else:
        print("No checkpoints found.")