import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# 路径 Hack
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置 EGL
os.environ["MUJOCO_GL"] = "egl"

from src.env.scene_generator import SceneGenerator
from src.env.mujoco_env import MujocoSimEnv
from src.planning.planner import CuRoboWrapper
from src.data.collector import extract_mujoco_meshes, sample_base_position, GRASP_ORIENTATIONS, AssetsManager

# 配置
CONFIG_DIR = os.path.join(project_root, "configs")
BASE_XML_PATH = os.path.join(CONFIG_DIR, "mobile_panda.xml")
SCENARIO = 'living_room' # 你想调试的场景

def visualize_failure():
    print(f"Starting Failure Debugger for {SCENARIO}...")
    
    # 初始化
    generator = SceneGenerator(base_xml_path=BASE_XML_PATH)
    assets_db = AssetsManager()
    temp_xml = "temp_debug_fail.xml"
    
    # 循环寻找失败案例
    while True:
        seed = np.random.randint(0, 100000)
        target_pos_world = generator.generate_target(SCENARIO, temp_xml, seed=seed)
        if target_pos_world is None: continue
        
        # 注入 Target Marker (方便看)
        with open(temp_xml, 'r', encoding='utf-8') as f: content = f.read()
        marker = f'<body name="debug_target" pos="{target_pos_world[0]} {target_pos_world[1]} {target_pos_world[2]}" mocap="true"><geom type="sphere" size="0.05" rgba="1 0 0 0.5" contype="0" conaffinity="0"/></body>'
        content = content.replace('</worldbody>', f'{marker}\n</worldbody>')
        with open(temp_xml, 'w', encoding='utf-8') as f: f.write(content)

        # 初始化环境
        try:
            env = MujocoSimEnv(temp_xml, width=320, height=240)
        except Exception: continue
        
        # 设置 Base
        robot_base = sample_base_position(target_pos_world, SCENARIO)
        q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        
        env.data.qpos[:3] = robot_base
        env.data.qpos[3:10] = q_home
        mujoco.mj_forward(env.model, env.data)
        
        # 准备 Planner
        l0_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        l0_pos = env.data.xpos[l0_id]
        l0_mat = env.data.xmat[l0_id].reshape(3,3)
        meshes = extract_mujoco_meshes(env.model, env.data, assets_db, l0_pos, l0_mat)
        
        try:
            planner = CuRoboWrapper(BASE_XML_PATH, initial_meshes=meshes)
        except Exception as e:
            print(f"Planner Init Error: {e}")
            continue
            
        # 尝试规划
        fail_reason = None
        success = False
        
        for quat in GRASP_ORIENTATIONS:
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            app_vec = r.apply([0, 0, 1])
            real_target = target_pos_world - (0.15 * app_vec)
            target_local = l0_mat.T @ (real_target - l0_pos)
            
            if np.linalg.norm(target_local) > 1.3:
                fail_reason = "Distance > 1.3m"
                continue
                
            traj, status = planner.plan_local(q_home, target_local, target_quat=quat)
            
            if traj is not None:
                success = True
                break
            else:
                fail_reason = status # 记录最后一个失败原因
        
        del planner
        
        if success:
            print(".", end="", flush=True) # 成功就跳过
            continue
            
        # --- 发现失败案例！启动可视化 ---
        print(f"\n\n[FAILURE FOUND] Reason: {fail_reason}")
        print(f"Base: {robot_base}")
        print(f"Target: {target_pos_world}")
        print("Opening Viewer... (Press ESC to close and find next failure)")
        
        # 重新加载（去掉 EGL，用 GLFW 显示）
        # 注意：这里可能需要一个新的进程或者重置 context，但在简单脚本里直接 launch 应该可以
        del os.environ["MUJOCO_GL"] # 尝试清除 EGL 强制
        
        # 为了显示，我们把刚才计算的 Real Target (Hand 应该在的位置) 也画出来
        # 取最后一个尝试的 quat 对应的 Real Target
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        vec = r.apply([0,0,1])
        hand_target = target_pos_world - (0.15 * vec)
        
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            # 画 Hand Target (蓝色小球)
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.03, 0, 0], pos=hand_target,
                mat=np.eye(3).flatten(), rgba=[0, 0, 1, 0.8]
            )
            
            while viewer.is_running():
                # 保持机器人静止在 Home，让你看它离目标有多远
                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
                time.sleep(0.02)
        
        # 恢复 EGL
        os.environ["MUJOCO_GL"] = "egl"
        print("Searching next...")

if __name__ == "__main__":
    visualize_failure()