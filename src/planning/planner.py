import torch
import numpy as np
import os
# 引入 Mesh 类型支持
from curobo.geom.types import WorldConfig, Cuboid, Mesh
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.types.math import Pose

class CuRoboWrapper:
    def __init__(self, robot_file="mujoco_menagerie/franka_emika_panda/panda.xml", initial_meshes=[]):
        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)
        # 假设机器人基座在 cuRobo 世界坐标系的原点
        # 这里的 offset 仅用于将世界坐标转为局部坐标的预处理
        self.robot_base_offset = np.array([0.25, 0.0, 0.3])

        # 基础静态障碍（地板等可以用简单的 Cuboid，节省显存）
        self.floor_cuboid = Cuboid(name="static_floor", pose=[0, 0, -0.35, 1, 0, 0, 0], dims=[10.0, 10.0, 0.1])
        self.base_cuboid = Cuboid(name="base_proxy", pose=[-0.25, 0, -0.15, 1, 0, 0, 0], dims=[0.45, 0.35, 0.2])
        
        static_cuboids = [self.floor_cuboid, self.base_cuboid]

        # 配置 WorldConfig，同时传入 Cuboid 和 Mesh
        # CuRobo 会自动处理这些 Mesh 用于碰撞检测 (SDF/Voxel)
        try:
            self.world_config = WorldConfig(cuboid=static_cuboids, mesh=initial_meshes)
        except TypeError:
            # 兼容不同 cuRobo 版本参数名 (部分版本使用复数形式 meshes)
            self.world_config = WorldConfig(cuboids=static_cuboids, meshes=initial_meshes)
        
        # 极小的激活距离，允许高精度贴合
        self.world_config.collision_activation_distance = 0.005

        # Load Config
        config = MotionGenConfig.load_from_robot_config(
            "franka.yml",
            world_model=self.world_config,
            tensor_args=self.tensor_args,
            interpolation_dt=0.02,
        )
        
        self.motion_gen = MotionGen(config)
        self.curobo_dof = self.motion_gen.kinematics.dof

    def _adapt_q(self, q):
        # 适配 7轴 (臂) 和 9轴 (臂+夹爪) 的差异
        if len(q) == 7 and self.curobo_dof == 9:
            return np.concatenate([q, [0.04, 0.04]])
        elif len(q) == 9 and self.curobo_dof == 7:
            return q[:7]
        return q

    def plan_local(self, start_q, target_pos_local, target_quat=[0, 1, 0, 0]):
        """
        规划局部路径。
        
        返回: 
            (trajectory, status_msg)
            - 成功: trajectory 为 (T, 7) 的 numpy 数组, status_msg 为 "SUCCESS"
            - 失败: trajectory 为 None, status_msg 为失败原因字符串
        """
        start_q_curobo = self._adapt_q(start_q)
        start_t = torch.tensor(start_q_curobo, device=self.tensor_args.device, dtype=self.tensor_args.dtype).unsqueeze(0)
        start_state = JointState.from_position(start_t)
        pos_t = torch.tensor(target_pos_local, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        quat_t = torch.tensor(target_quat, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        goal_pose = Pose(position=pos_t.unsqueeze(0), quaternion=quat_t.unsqueeze(0))
        
        try:
            result = self.motion_gen.plan_single(start_state=start_state, goal_pose=goal_pose)
            
            if result.success.item():
                traj = result.interpolated_plan.position.squeeze(0).cpu().numpy()
                return traj[:, :7], "SUCCESS"
            else:
                # [修复] 安全获取失败状态码
                s = result.status
                # 兼容不同版本: 有些是 Tensor，有些是 MotionGenStatus Enum
                if hasattr(s, 'item'):
                    status_code = s.item()
                else:
                    status_code = str(s)
                
                # 常见状态码参考: 2=IK_FAIL(不可达), 3=COLLISION(碰撞)
                return None, f"FAIL_{status_code}"
                
        except Exception as e:
            # 捕获其他异常（如显存不足、内部错误等）
            return None, f"EXCEPTION_{str(e)}"
        
        return None, "UNKNOWN_FAIL"

    def plan_to_pose(self, start_q, target_pos_world, target_quat=[0, 1, 0, 0]):
        """
        世界坐标系规划辅助函数
        """
        local_target_pos = np.array(target_pos_world) - self.robot_base_offset
        return self.plan_local(start_q, local_target_pos, target_quat)