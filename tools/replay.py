import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import glob
import argparse
import cv2

# 设置环境变量，防止 OpenCV 冲突
os.environ["MUJOCO_GL"] = "egl"

def normalize_depth(depth_map):
    """ 将深度图转为伪彩色以便可视化 """
    # 简单的归一化: 0.1m - 2.0m 映射到 0-255
    # 如果 depth 是 float16 (米)
    valid_mask = (depth_map > 0.01) & (depth_map < 3.0)
    if np.sum(valid_mask) == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
        
    min_d, max_d = np.min(depth_map[valid_mask]), np.max(depth_map[valid_mask])
    norm = (depth_map - min_d) / (max_d - min_d + 1e-6)
    norm = np.clip(norm, 0, 1)
    norm_uint8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def replay_file(npz_file):
    print(f"Replaying: {npz_file}")
    
    # 1. 加载数据
    data = np.load(npz_file)
    q_arm_traj = data['q_arm']
    robot_base = data['robot_base']
    target_pos = data['target_pos']
    scene_xml = str(data['scene_xml'])
    
    # 检查是否有图像数据
    has_visual = 'cam_left_rgb' in data
    
    # 2. 创建临时 XML
    temp_xml = "temp_replay.xml"
    with open(temp_xml, 'w', encoding='utf-8') as f:
        f.write(scene_xml)
        
    # 3. 加载 MuJoCo
    model = mujoco.MjModel.from_xml_path(temp_xml)
    d = mujoco.MjData(model)
    
    # 4. 启动 Viewer
    with mujoco.viewer.launch_passive(model, d) as viewer:
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0], pos=target_pos,
            mat=np.eye(3).flatten(), rgba=[0, 1, 0, 0.8]
        )
        
        print(f"Traj length: {len(q_arm_traj)}")
        
        # 循环播放
        for t, q_arm in enumerate(q_arm_traj):
            step_start = time.time()
            
            # --- 更新 MuJoCo ---
            d.qpos[:3] = robot_base
            d.qpos[3:10] = q_arm
            mujoco.mj_forward(model, d)
            viewer.sync()
            
            # --- 更新 OpenCV 窗口 (如果有数据) ---
            if has_visual:
                # 获取当前帧图像
                # 注意: 数据长度可能和 traj 不完全一致 (由于 padding)
                idx = min(t, len(data['cam_left_rgb']) - 1)
                
                rgb_l = data['cam_left_rgb'][idx]
                dep_l = data['cam_left_depth'][idx]
                rgb_r = data['cam_right_rgb'][idx]
                dep_r = data['cam_right_depth'][idx]
                rgb_w = data['cam_wrist_rgb'][idx]
                dep_w = data['cam_wrist_depth'][idx]
                
                # 拼接
                row1 = np.hstack([to_bgr(rgb_l), normalize_depth(dep_l)])
                row2 = np.hstack([to_bgr(rgb_r), normalize_depth(dep_r)])
                row3 = np.hstack([to_bgr(rgb_w), normalize_depth(dep_w)])
                grid = np.vstack([row1, row2, row3])
                
                # 缩放
                if grid.shape[0] > 800:
                    grid = cv2.resize(grid, (0,0), fx=0.6, fy=0.6)
                
                cv2.imshow("Multi-View Depth Replay", grid)
                cv2.waitKey(1) # 必须调用一次 waitKey 才能刷新窗口

            # 控制帧率
            time_until_next = 0.05 - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
            
            if not viewer.is_running(): break
            
    cv2.destroyAllWindows()
    if os.path.exists(temp_xml): os.remove(temp_xml)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='kitchen')
    parser.add_argument('--data_dir', type=str, default='/media/j16/8deb00a4-cceb-e842-a899-55532424da473/dataset_v2')
    parser.add_argument('--num_replays', type=int, default=10)
    args = parser.parse_args()
    
    search_path = os.path.join(args.data_dir, args.scene, "*.npz")
    files = glob.glob(search_path)
    if not files:
        print("No files found.")
        return
    
    files.sort(key=os.path.getmtime, reverse=True)
    
    for i, f in enumerate(files[:args.num_replays]):
        replay_file(f)
        if i < args.num_replays - 1:
            if input("Next? (y/n): ").lower() == 'n': break

if __name__ == "__main__":
    main()