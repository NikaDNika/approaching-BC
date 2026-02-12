import os
import torch
import torch.nn.functional as F
import numpy as np
import mujoco
import random

# 彻底禁用图形界面和渲染，确保不会卡死
os.environ["MUJOCO_GL"] = "osmesa" 

from src.env.scene_generator import SceneGenerator
from src.training.train_bc_with_ar_transformer_multistep import TransformerPolicyWithAR
from src.models.teacher_engine import GFTeacher

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "checkpoints/bc_with_ar_transformer_multistep_20260209_174203/best_model.pth"
BASE_XML_PATH = "configs/mobile_panda.xml"
TEMP_XML = "temp_finetune_iteration.xml"

def get_observation_cpu(model, data, teacher_robot_ids, l0_id, obj_p, goal_p):
    """
    完全在 CPU 上通过几何体采样生成点云，不使用任何渲染器。
    """
    l0_pos = data.xpos[l0_id]
    l0_mat = data.xmat[l0_id].reshape(3, 3)
    
    pts_list = []

    # 遍历所有几何体进行采样
    for i in range(model.ngeom):
        # 1. 判断是否是机器人 (利用教师预计算的集合)
        body_id = model.geom_bodyid[i]
        is_robot = body_id in teacher_robot_ids
        
        # 2. 过滤地面
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and any(k in name.lower() for k in ["ground", "floor", "grasp", "viz"]):
            continue
            
        # 3. 采样参数
        g_pos = data.geom_xpos[i]
        g_size = model.geom_size[i] # (3,)
        
        # 机器人采少点(30)，环境采多点(50)
        n_samples = 30 if is_robot else 50
        
        # 处理 Mesh (size=0) 的情况，给默认厚度
        size_arr = g_size[:3].copy()
        size_arr[size_arr < 0.01] = 0.05 
        
        local_pts = np.random.uniform(-size_arr, size_arr, (n_samples, 3))
        world_pts = g_pos + local_pts
        pts_list.append(world_pts)

    # 合并点云
    if pts_list:
        pc_world = np.concatenate(pts_list, axis=0)
        # 距离过滤：只保留离机器人基座 1.5m 内的点
        dist_to_base = np.linalg.norm(pc_world - l0_pos, axis=1)
        pc_world = pc_world[dist_to_base < 1.5]
    else:
        pc_world = np.zeros((1, 3))

    # 下采样到 1024
    if len(pc_world) > 1024:
        idx = np.random.choice(len(pc_world), 1024, replace=False)
        pc_final_w = pc_world[idx]
    else:
        pc_final_w = np.zeros((1024, 3))
        if len(pc_world) > 0:
            replace = len(pc_world) < 1024
            idx = np.random.choice(len(pc_world), 1024, replace=replace)
            pc_final_w = pc_world[idx]

    # 转到 link0 坐标系 (学生输入)
    pc_l0 = (l0_mat.T @ (pc_final_w - l0_pos).T).T

    # 构造 AR 向量
    def to_l0(p): return l0_mat.T @ (p - l0_pos)
    ar = np.concatenate([to_l0(obj_p), to_l0(goal_p)])
    
    return pc_l0, data.qpos[3:10].copy(), ar

def main():
    print(">>> System Boot: CPU-based Finetuning", flush=True)
    
    # 1. 加载学生
    student = TransformerPolicyWithAR().to(DEVICE)
    student.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False))
    student.train()
    print(">>> Student Model Loaded.", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    generator = SceneGenerator(base_xml_path=BASE_XML_PATH)

    for epoch in range(2000):
        print(f"Epoch {epoch}: Resetting...", end="", flush=True)
        # 强制采样困难场景
        scen = np.random.choice(["kitchen", "storage", "corner"])
        obj_p, goal_p = generator.generate_target(scen, TEMP_XML)
        
        # 直接加载 XML
        model = mujoco.MjModel.from_xml_path(TEMP_XML)
        data = mujoco.MjData(model)
        
        # 初始化教师
        teacher = GFTeacher(model, data, safe_dist=0.25)
        
        # 获取 link0 ID
        l0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        if l0_id == -1: l0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_link0")

        # 初始姿态
        data.qpos[3:10] = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        mujoco.mj_forward(model, data)
        print(" OK.", flush=True)

        epoch_loss, active_steps = 0, 0
        min_d_log = 999.0
        
        for step in range(300):
            # 1. CPU 采样观测 (修正了函数名)
            pc_l0, q_a, ar = get_observation_cpu(model, data, teacher.robot_body_ids, l0_id, obj_p, goal_p)
            
            # 2. 学生推理
            pc_t = torch.FloatTensor(pc_l0).unsqueeze(0).to(DEVICE)
            qc_t = torch.FloatTensor(q_a).unsqueeze(0).to(DEVICE)
            ar_t = torch.FloatTensor(ar).unsqueeze(0).to(DEVICE)
            
            student_delta_q = student(pc_t, qc_t, ar_t)
            s_np = student_delta_q[0].detach().cpu().numpy()
            
            # 3. 教师纠偏
            t_np, min_dist = teacher.get_corrected_delta(s_np, q_a, obj_p)
            min_d_log = min(min_d_log, min_dist)
            
            # 4. 学习
            if not np.allclose(s_np, t_np, atol=1e-4):
                loss = F.mse_loss(student_delta_q[0], torch.FloatTensor(t_np).to(DEVICE))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                active_steps += 1
            
            # 5. 物理步进
            data.qpos[3:10] += t_np
            mujoco.mj_forward(model, data)
            
            # 到达判定
            tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
            if np.linalg.norm(data.site_xpos[tcp] - obj_p) < 0.05:
                print(" [Goal Reached]", end="", flush=True)
                break

        avg_l = epoch_loss / active_steps if active_steps > 0 else 0
        print(f" -> Active: {active_steps}, Loss: {avg_l:.6f}, MinD: {min_d_log:.3f}", flush=True)
        
        if (epoch + 1) % 50 == 0:
            torch.save(student.state_dict(), f"checkpoints/fine/finetune/tune_ep{epoch+1}.pth")

if __name__ == "__main__":
    main()