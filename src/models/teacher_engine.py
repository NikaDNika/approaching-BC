import numpy as np
import mujoco

class GFTeacher:
    def __init__(self, model, data, safe_dist=0.25):
        self.model = model
        self.data = data
        self.safe_dist = safe_dist
        
        # --- 1. 预计算属于机器人的 Body ID 集合 (性能优化 + 准确性) ---
        self.robot_body_ids = set()
        
        # 尝试找到根节点
        root_name = "link0"
        l0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        if l0_id == -1: 
            root_name = "panda_link0"
            l0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        
        if l0_id != -1:
            # 遍历所有 body，检查其祖先是否包含 link0
            for i in range(model.nbody):
                curr = i
                path_depth = 0
                while curr != -1 and path_depth < 20: # 防止死循环
                    if curr == l0_id:
                        self.robot_body_ids.add(i)
                        break
                    curr = model.body_parentid[curr]
                    path_depth += 1
        else:
            print("Warning: Could not find robot root link (link0/panda_link0). Teacher might misidentify robot as obstacle.")

    def get_corrected_delta(self, student_delta_q, current_q_arm, object_pos):
        """
        输入:
            student_delta_q: (7,) float32
            current_q_arm: (7,) float64
            object_pos: (3,) float64 (目标物体位置，用于豁免)
        """
        # 备份状态
        old_qpos = self.data.qpos.copy()
        
        # 预演动作
        self.data.qpos[3:10] = current_q_arm + student_delta_q
        mujoco.mj_forward(self.model, self.data)
        
        # --- 2. 扫描障碍物 (上帝视角) ---
        obstacle_pts = []
        for i in range(self.model.ngeom):
            # 排除机器人自身的几何体
            if self.model.geom_bodyid[i] in self.robot_body_ids:
                continue
            
            # 排除地面和辅助可视化物体
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name:
                n_low = name.lower()
                if any(k in n_low for k in ["ground", "floor", "grasp", "viz", "site"]):
                    continue
            
            # 获取位置
            g_pos = self.data.geom_xpos[i]
            
            # **目标豁免**：如果这个障碍物离目标物体很近 (比如是桌子中心)，暂时忽略，否则抓不到
            if np.linalg.norm(g_pos - object_pos) < 0.10:
                continue
                
            obstacle_pts.append(g_pos.copy())
        
        # 如果场景极度空旷
        if not obstacle_pts:
            self.data.qpos[:] = old_qpos
            mujoco.mj_forward(self.model, self.data)
            return student_delta_q, 999.0

        obs_pts = np.array(obstacle_pts)
        
        # --- 3. 多点避障计算 ---
        # 检测点：TCP(手), Link6(腕), Link4(肘)
        check_points = [("site", "tcp_site"), ("body", "panda_link6"), ("body", "panda_link4")]
        
        correction_delta = student_delta_q.copy()
        global_min_dist = 999.0

        for p_type, p_name in check_points:
            p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE if p_type=="site" else mujoco.mjtObj.mjOBJ_BODY, p_name)
            if p_id == -1: continue
            
            pos = self.data.site_xpos[p_id] if p_type=="site" else self.data.xpos[p_id]
            
            # 计算到所有障碍物的距离
            dists = np.linalg.norm(obs_pts - pos, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            global_min_dist = min(global_min_dist, min_dist)

            # 触发避障
            if min_dist < self.safe_dist:
                nearest_pt = obs_pts[min_idx]
                repel_dir = (pos - nearest_pt) / (min_dist + 1e-6)
                
                jacp = np.zeros((3, self.model.nv))
                if p_type == "site":
                    mujoco.mj_jacSite(self.model, self.data, jacp, None, p_id)
                else:
                    mujoco.mj_jac(self.model, self.data, jacp, None, pos, p_id)
                
                jac_arm = jacp[:, 3:10] # 取 7 轴雅可比
                
                # 推力增益
                push_gain = 2.0 * (self.safe_dist - min_dist)
                correction_delta += jac_arm.T @ repel_dir * push_gain

        # 还原
        self.data.qpos[:] = old_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return correction_delta, global_min_dist