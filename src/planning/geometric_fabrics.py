import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class SDFComputer:
    """
    提供精确的几何体 SDF 计算功能。
    核心功能：计算一个点到一个任意旋转的 Box 的有向距离 (Signed Distance) 和梯度 (Normal)。
    """
    @staticmethod
    def get_box_sdf_and_gradient(point, box_pos, box_quat, box_dims):
        """
        Input:
            point: [3] 查询点坐标 (World Frame)
            box_pos: [3] Box 中心坐标
            box_quat: [4] (w, x, y, z) Box 四元数
            box_dims: [3] Box 全尺寸 (length, width, height)
        Returns:
            dist: 标量距离 (负数表示在内部)
            gradient: [3] 距离场梯度 (指向远离 Box 的方向)
        """
        # 1. 将点转换到 Box 的局部坐标系 (Local Frame)
        # MuJoCo Quat is [w, x, y, z] -> Scipy needs [x, y, z, w]
        r = R.from_quat([box_quat[1], box_quat[2], box_quat[3], box_quat[0]])
        # R_world_to_local = R_local_to_world.inv()
        rot_matrix = r.as_matrix()
        
        # p_local = R^T * (p_world - p_center)
        p_local = rot_matrix.T @ (point - box_pos)
        
        # 2. 计算 Axis-Aligned Box SDF
        # box_dims 是全尺寸，半尺寸 (half_extents) 是 dims/2
        half_extents = box_dims / 2.0
        
        # q 是点到第一象限边界的向量
        # d = |q|_2 if outside, etc.
        # algorithm ref: Inigo Quilez (SDF functions)
        q = np.abs(p_local) - half_extents
        
        # 外部距离：只取正分量的模长
        dist_outside = np.linalg.norm(np.maximum(q, 0.0))
        
        # 内部距离：取最大的负分量 (通常是负的，靠近边界的最大值)
        dist_inside = min(max(q[0], max(q[1], q[2])), 0.0)
        
        dist = dist_outside + dist_inside
        
        # 3. 计算局部梯度 (Normal)
        # 这是一个近似或解析梯度
        if dist > 0:
            # 在外部：梯度是 q 的正部分的方向
            grad_local = np.zeros(3)
            # 只有大于0的分量贡献梯度
            mask = q > 0
            if np.any(mask):
                grad_local[mask] = p_local[mask] - np.sign(p_local[mask]) * half_extents[mask]
                # Normalize
                norm = np.linalg.norm(grad_local)
                if norm > 1e-6:
                    grad_local /= norm
                else:
                    grad_local = np.array([1.0, 0, 0]) # Fallback
        else:
            # 在内部：梯度指向最近的表面
            # 找到 q 中最大的分量（最接近0的分量，即最浅的穿透）
            max_axis = np.argmax(q)
            grad_local = np.zeros(3)
            # 梯度方向就是该轴的方向，符号由 p_local 决定
            grad_local[max_axis] = np.sign(p_local[max_axis])
            
        # 4. 将梯度变换回世界坐标系
        # grad_world = R * grad_local
        grad_world = rot_matrix @ grad_local
        
        return dist, grad_world

class GeometricFabricsController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # === 1. 定义机器人碰撞代理点 (Collision Proxies) ===
        # 我们在 Link 的局部坐标系下定义一系列点，覆盖手臂的关键部位
        # 格式: {body_name: [[x, y, z], [x, y, z], ...]}
        self.robot_proxies = {
            "link3": [[0, 0, -0.1], [0, 0, 0], [0, 0, 0.1]],
            "link4": [[0, 0, 0], [0, 0, 0.1]], # Link4 很短
            "link5": [[0, 0, -0.1], [0, 0, 0], [0, 0.05, 0]], 
            "link6": [[0, 0, 0], [0, 0, 0.05]],
            "link7": [[0, 0, 0.05], [0, 0, 0.1]],
            "hand":  [[0, 0, 0.05], [0, -0.05, 0.05], [0, 0.05, 0.05]], # 手掌宽
            "left_finger": [[0, 0.01, 0.02]],
            "right_finger":[[0, -0.01, 0.02]]
        }
        
        # 缓存 body ids
        self.proxy_data = []
        for name, offsets in self.robot_proxies.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                self.proxy_data.append({
                    "id": bid,
                    "offsets": np.array(offsets) # Local offsets
                })

        # === 2. 调参 ===
        self.activation_dist = 0.15  # 距离小于 15cm 开始产生斥力
        self.stiffness = 5.0         # 吸引力系数 (去目标)
        self.repulsion_scale = 8.0   # 斥力系数 (避障)
        self.damping = 2.0           # 阻尼

    def compute_refined_action(self, q_curr, q_dot_curr, q_target, obstacles_list, dt=0.01):
        """
        Input:
            q_curr: [7] 当前关节角
            q_dot_curr: [7] 当前关节速度 (如果不知道可以传0)
            q_target: [7] IMPACT 预测的目标关节角
            obstacles_list: 障碍物列表
        Returns:
            q_next: [7] 修正后的下一时刻关节角
        """
        
        # 1. 吸引力 (Attractor): 简单的 PD 控制拉向 q_target
        # F_att = Kp * (q_des - q) - Kd * q_dot
        # 这里的 F 其实是关节空间的加速度或力矩
        acc_attract = self.stiffness * (q_target - q_curr) - self.damping * q_dot_curr
        
        # 2. 斥力 (Repulsion based on SDF)
        acc_repulse = np.zeros(7)
        
        # 确保数据已更新 (Kinematics)
        # 注意：外部调用 loop 通常已经 forward 过了，如果没有，这里需要 set qpos 并 forward
        # mujoco.mj_forward(self.model, self.data) (假设外部已做)

        # 遍历每一个身体代理点
        for p_data in self.proxy_data:
            bid = p_data['id']
            offsets = p_data['offsets']
            
            # 获取 Body 的世界坐标和旋转
            body_pos = self.data.xpos[bid]
            body_mat = self.data.xmat[bid].reshape(3, 3)
            
            # 计算该 Link 的雅可比矩阵 (6 x nv)
            # 我们只需要平动部分的 Jacobian (3 x 7)
            # 注意：MuJoCo 的 jacBody 计算的是 Body 原点的 Jacobian
            # 对于 Offset 点，J_point = J_trans - cross(pos_offset, J_rot)
            # 为了计算效率，近似认为同一个 Link 上的 J 是相似的，或者我们需要为每个点算 J
            # **为了精确**：我们使用 mj_jacPoint 为每个 Offset 点计算 J
            
            for offset in offsets:
                # 计算代理点的世界坐标
                point_world = body_pos + body_mat @ offset
                
                # 计算该点的 Jacobian (3 x nv)
                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jacp, None, point_world, bid)
                J = jacp[:, :7] # 取前7个关节
                
                # --- SDF 查询 ---
                min_dist = 999.0
                best_grad = np.zeros(3)
                
                for obs in obstacles_list:
                    dist, grad = SDFComputer.get_box_sdf_and_gradient(
                        point_world, 
                        np.array(obs['pos']), 
                        np.array(obs['quat']), 
                        np.array(obs['dims'])
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_grad = grad # 指向远离障碍物的方向
                
                # --- 生成斥力 ---
                if min_dist < self.activation_dist:
                    # RMP 风格的势能函数: phi = 0.5 * alpha * (dist - activation)^2
                    # Force = - grad(phi)
                    # 为了更强的效果，我们使用倒数形式: F ~ 1/dist
                    
                    # 距离越小，力越大。当进入内部(dist<0)时，力极大
                    w = np.clip(min_dist, -0.01, self.activation_dist)
                    
                    # 斥力强度函数 (Riemannian Metric 调制)
                    # magnitude = alpha / (dist + epsilon)^2
                    mag = self.repulsion_scale * (1.0 / (w + 0.02) - 1.0 / (self.activation_dist + 0.02))
                    
                    # Cartesian Force
                    f_cart = best_grad * mag
                    
                    # 映射回关节空间: tau = J^T * F
                    acc_repulse += J.T @ f_cart

        # 3. 求解动力学 (Simplified)
        # q_ddot = acc_attract + acc_repulse
        # 这里为了稳定，我们限制最大斥力
        acc_repulse = np.clip(acc_repulse, -15.0, 15.0)
        
        q_ddot_total = acc_attract + acc_repulse
        
        # 半隐式欧拉积分
        q_dot_next = q_dot_curr + q_ddot_total * dt
        q_next = q_curr + q_dot_next * dt
        
        return q_next