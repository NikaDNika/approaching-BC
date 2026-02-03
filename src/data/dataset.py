import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import mujoco
import re

class ApproachDataset(Dataset):
    def __init__(self, data_dir, chunk_size=10, num_points=2048, num_robot_points=512):
        self.chunk_size = chunk_size
        self.num_points = num_points
        self.num_robot_points = num_robot_points
        
        # 1. 搜索文件
        self.files = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        
        if len(self.files) == 0:
            raise RuntimeError(f"No data found in {data_dir}")
        else:
            print(f"Found {len(self.files)} trajectories.")

        # 2. 智能定位 XML 路径
        # 获取当前脚本所在目录 (src/data)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录 (DRP_FULL)
        self.project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # 尝试多个可能的位置
        possible_paths = [
            os.path.join(self.project_root, "configs", "mobile_panda.xml"), 
            os.path.join(self.project_root, "mobile_panda.xml"),            
            "mobile_panda.xml"                                         
        ]
        
        self.xml_path = None
        for p in possible_paths:
            if os.path.exists(p):
                self.xml_path = p
                print(f"Dataset using XML: {self.xml_path}")
                break
        
        if self.xml_path is None:
            raise FileNotFoundError(f"Could not find mobile_panda.xml in checked paths: {possible_paths}")

        # 延迟加载
        self.model = None
        self.data = None

    def __len__(self):
        return len(self.files)

    def _init_mujoco(self):
        """
        在每个 Worker 进程内部初始化 MuJoCo。
        动态替换 meshdir 为绝对路径，防止多进程下路径迷失。
        """
        if self.model is None:
            try:
                # 读取原始 XML
                with open(self.xml_path, 'r') as f:
                    xml_content = f.read()
                
                # 构造 assets 的绝对路径
                # 优先级 1: mujoco_menagerie/...
                assets_abs_path = os.path.join(self.project_root, "mujoco_menagerie", "franka_emika_panda", "assets")
                
                # 优先级 2: assets/
                if not os.path.exists(assets_abs_path):
                    assets_abs_path = os.path.join(self.project_root, "assets")
                
                # 检查路径是否存在
                if not os.path.exists(assets_abs_path):
                    print(f"WARNING: Assets directory not found at {assets_abs_path}")
                    # 不报错，也许 XML 里写的绝对路径是对的
                else:
                    # 替换 XML 中的 meshdir
                    if 'meshdir="' in xml_content:
                        xml_content = re.sub(r'meshdir="[^"]+"', f'meshdir="{assets_abs_path}"', xml_content)
                
                # 从字符串加载模型
                self.model = mujoco.MjModel.from_xml_string(xml_content)
                self.data = mujoco.MjData(self.model)
                
            except Exception as e:
                print(f"!!! FATAL ERROR in _init_mujoco !!!")
                print(f"XML Path: {self.xml_path}")
                print(f"Project Root: {self.project_root}")
                print(f"Error: {e}")
                raise RuntimeError(f"Failed to load MuJoCo model. Error: {e}")

    def _get_robot_point_cloud(self, q_pos):
        """
        根据关节角计算机器人自身点云 Pr
        """
        self._init_mujoco()
        
        # 设置状态
        self.data.qpos[:3] = np.zeros(3) 
        self.data.qpos[3:10] = q_pos
        
        # 前向运动学
        mujoco.mj_kinematics(self.model, self.data)
        
        robot_points = []
        for i in range(self.model.ngeom):
            pos = self.data.geom_xpos[i]
            size = self.model.geom_size[i]
            
            # 处理 Mesh 类型 (size通常为0)
            if np.all(size == 0):
                rbound = self.model.geom_rbound[i]
                if rbound > 0:
                    size = np.array([rbound, rbound, rbound]) * 0.5
                else:
                    size = np.array([0.05, 0.05, 0.05]) # 默认极小值

            # 在包围盒内随机撒点
            pts = pos + np.random.uniform(-size[:3], size[:3], (15, 3))
            robot_points.append(pts)
            
        if not robot_points:
            return np.zeros((self.num_robot_points, 3), dtype=np.float32)

        robot_points = np.concatenate(robot_points, axis=0)
        
        curr_len = len(robot_points)
        if curr_len >= self.num_robot_points:
            idx = np.random.choice(curr_len, self.num_robot_points, replace=False)
            return robot_points[idx]
        else:
            idx = np.random.choice(curr_len, self.num_robot_points, replace=True)
            return robot_points[idx]

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            with np.load(file_path) as data:
                traj_len = len(data['q_arm'])
                
                if traj_len <= self.chunk_size + 1: 
                    # 随机换一个文件，避免递归过深
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx)
                
                t = np.random.randint(0, traj_len)
                
                # 1. Scene Point Cloud (Ps)
                pc = data['pc_ar'][t]
                if len(pc) >= self.num_points:
                    idx_pc = np.random.choice(len(pc), self.num_points, replace=False)
                    pc = pc[idx_pc]
                else:
                    idx_pc = np.random.choice(len(pc), self.num_points, replace=True)
                    pc = pc[idx_pc]
                
                # 2. Current Joint (qc)
                curr_q = data['q_arm'][t]
                
                # 3. Robot Point Cloud (Pr)
                robot_pc = self._get_robot_point_cloud(curr_q)
                
                # 4. Goal Joint Config (q_mg)
                goal_q = data['q_arm'][-1]
                
                # 5. Action Chunk (GT)
                action_chunk = data['action'][t]
                
                curr_chunk_len = action_chunk.shape[0]
                is_pad = np.zeros(self.chunk_size, dtype=np.float32)
                
                if curr_chunk_len < self.chunk_size:
                    pad_len = self.chunk_size - curr_chunk_len
                    last_act = action_chunk[-1] if curr_chunk_len > 0 else np.zeros(7)
                    padding = np.tile(last_act, (pad_len, 1))
                    action_chunk = np.vstack([action_chunk, padding])
                    is_pad[curr_chunk_len:] = 1.0

                return (
                    torch.from_numpy(pc).float(),        
                    torch.from_numpy(robot_pc).float(),  
                    torch.from_numpy(curr_q).float(),    
                    torch.from_numpy(goal_q).float(),    
                    torch.from_numpy(action_chunk).float(), 
                    torch.from_numpy(is_pad).float()     
                )
                
        except Exception as e:
            # 打印错误，方便调试
            print(f"[Dataset Error] File: {file_path}")
            print(f"Error: {e}")
            # 随机重试
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)