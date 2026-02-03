import torch
import numpy as np

class DCP_RMP:
    def __init__(self, repulsion_gain=0.5, influence_radius=0.5):
        """
        Dynamic Closest Point - Riemannian Motion Policy
        用于推理阶段的动态避障。
        """
        self.gain = repulsion_gain
        self.radius = influence_radius
        self.prev_pc = None

    def step(self, current_pc, current_q, original_goal_q):
        """
        输入:
            current_pc: 当前帧点云 (N, 3) numpy
            current_q: 当前关节角 (7,)
            original_goal_q: 原始目标关节角 (7,)
        输出:
            modified_goal_q: 修改后的目标关节角
        """
        # 1. 检测动态点 (简单帧差法)
        # 实际论文可能用了更复杂的流估计，这里用简化的距离阈值
        if self.prev_pc is None:
            self.prev_pc = current_pc
            return original_goal_q
            
        # 假设点云是有序的(来自相机)，直接算距离差。
        # 如果是无序点云，需要先做最近邻匹配。这里假设数据已处理好或直接由模拟器提供动态物体信息。
        # 为简化实现，我们假设 current_pc 中已经通过分割算法提取出了 "动态物体点云" dynamic_pts
        # 在仿真中，你可以直接获取动态障碍物的 geom position。
        
        # 模拟：假设所有离机器人很近的点都是威胁
        # 计算点云到机器人的距离（简化为到原点/基座的距离）
        dists = np.linalg.norm(current_pc, axis=1)
        
        # 找到最近的障碍点
        closest_idx = np.argmin(dists)
        closest_pt = current_pc[closest_idx]
        min_dist = dists[closest_idx]
        
        # 2. 如果最近点在影响半径内，产生斥力
        if min_dist < self.radius:
            # 斥力方向：从障碍点指向机器人(原点)
            # Robot is at 0, Obstacle is at closest_pt
            # Direction = 0 - closest_pt = -closest_pt
            repulsion_dir = -closest_pt / (np.linalg.norm(closest_pt) + 1e-6)
            
            # 斥力大小：距离越近越大
            magnitude = self.gain * (1.0 / (min_dist + 0.01) - 1.0 / self.radius)
            
            # 将笛卡尔空间的斥力映射到关节空间 (需要 Jacobian，这里简化为直接修改 Goal)
            # 这是一个简化的 RMP 实现：我们不直接修改力，而是把 Goal 往斥力方向挪
            
            # 这里的逻辑是：Goal_new = Goal_old + Repulsion_Joint_Space
            # 在没有 Jacobian 的情况下，我们简单地给 Goal 加噪或者根据末端位置反推
            # 论文中的 RMP 是加速度级的，这里我们做 Goal Proposal 级的简化：
            
            # 简单策略：如果检测到障碍，暂时保持当前姿态不动，或者向反方向微调
            # 这里返回一个“虚拟目标”，让 IMPACT 规划出避让动作
            
            # 既然 IMPACT 输入是 q_goal，我们可以把 q_goal 修改为：
            # "当前关节角 + 远离障碍的微小增量"
            
            modified_goal = original_goal_q + magnitude * 0.1 # 仅作示意
            return modified_goal
            
        self.prev_pc = current_pc
        return original_goal_q