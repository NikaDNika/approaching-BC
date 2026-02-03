import numpy as np
import mujoco
import mujoco.viewer
import cv2
import os
import time
import copy

os.environ['MUJOCO_GL'] = 'glfw' # 强制使用窗口模式

# --- 配置 ---
TARGET_FILE = "/media/j16/8deb00a4-cceb-e842-a899-55532424da473/dataset_v2/living_room/traj_living_room_0_6_1770012884993942817.npz"
OUTPUT_VIDEO = "output/video/output_user_view.mp4"
WIDTH, HEIGHT = 1280, 720  # 最终视频的分辨率（可以比预览窗口更清晰）
FPS = 30

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"Error: File not found {TARGET_FILE}"); return

    # 1. 加载数据
    print(f"Loading: {TARGET_FILE}")
    data_pkg = np.load(TARGET_FILE, allow_pickle=True)
    
    # 解析轨迹
    if 'q_full' in data_pkg:
        traj_data = data_pkg['q_full']
        is_mobile = True
        print("Type: Mobile Robot")
    elif 'q_arm' in data_pkg:
        traj_data = data_pkg['q_arm']
        robot_base_fixed = data_pkg.get('robot_base', np.zeros(3))
        is_mobile = False
        print("Type: Fixed Arm")
    else: return

    # 2. 加载模型
    xml_content = str(data_pkg['scene_xml'])
    temp_xml = "viz_temp_interactive.xml"
    with open(temp_xml, "w", encoding='utf-8') as f: f.write(xml_content)
        
    model = mujoco.MjModel.from_xml_path(temp_xml)
    data = mujoco.MjData(model)

    # 设置为第0帧状态，方便调整视角
    if is_mobile:
        data.qpos[:10] = traj_data[0]
    else:
        data.qpos[:3] = robot_base_fixed
        data.qpos[3:10] = traj_data[0]
    mujoco.mj_forward(model, data)

    # =========================================================
    # 阶段 1: 交互式选择视角
    # =========================================================
    print("\n" + "="*60)
    print("【操作指南】")
    print("1. 窗口将弹出，显示轨迹的第一帧。")
    print("2. 使用鼠标调整视角 (左键旋转, 右键/滚轮缩放, Shift+右键平移)。")
    print("3. 调整满意后，>>> 直接关闭窗口 <<< 即可开始录制。")
    print("="*60 + "\n")

    # 用于保存用户视角的变量
    user_cam_params = None

    # 启动被动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 循环直到用户关闭窗口
        while viewer.is_running():
            # 每一帧都同步一下物理状态（虽然这里是静止的）
            viewer.sync()
            
            # 【关键】实时复制当前的相机参数
            # 我们必须深拷贝，因为一旦 viewer 关闭，viewer.cam 可能无法访问
            user_cam_params = {
                "azimuth": viewer.cam.azimuth,
                "elevation": viewer.cam.elevation,
                "distance": viewer.cam.distance,
                "lookat": viewer.cam.lookat.copy(),
                "type": viewer.cam.type
            }
            time.sleep(0.05)

    print("View captured! Starting render...")

    # =========================================================
    # 阶段 2: 使用捕获的视角进行渲染
    # =========================================================
    
    # 重新初始化一个 Renderer (使用刚才的上下文或新建)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    
    # 创建一个新的 MjvCamera 对象并应用用户的参数
    render_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, render_cam) # 初始化默认值
    
    if user_cam_params:
        render_cam.azimuth = user_cam_params["azimuth"]
        render_cam.elevation = user_cam_params["elevation"]
        render_cam.distance = user_cam_params["distance"]
        render_cam.lookat = user_cam_params["lookat"]
        # 注意：不要覆盖 type，通常保持 free camera 即可
        print(f"Camera Params: Azim={render_cam.azimuth:.1f}, Elev={render_cam.elevation:.1f}, Dist={render_cam.distance:.2f}")
    
    # 初始化视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))
    
    traj_len = len(traj_data)
    
    for t in range(traj_len):
        if is_mobile:
            data.qpos[:10] = traj_data[t]
        else:
            data.qpos[:3] = robot_base_fixed
            data.qpos[3:10] = traj_data[t]
        
        mujoco.mj_forward(model, data)
        
        # 使用我们手动设置的 render_cam
        renderer.update_scene(data, camera=render_cam)
        
        pixels = renderer.render()
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr)
        
        if t % 50 == 0:
            print(f"  Rendering: {t}/{traj_len}", end='\r')

    print(f"\nDone! Video saved to {OUTPUT_VIDEO}")
    video_writer.release()
    try: os.remove(temp_xml)
    except: pass

if __name__ == "__main__":
    main()