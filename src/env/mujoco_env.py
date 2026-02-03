import mujoco
import numpy as np
import cv2

class MujocoSimEnv:
    def __init__(self, xml_path, width=320, height=240):
        """
        初始化仿真环境。
        width, height: 渲染图像的分辨率。
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 渲染器分辨率
        self.width = width
        self.height = height
        
        # 初始化渲染器
        self.renderer = mujoco.Renderer(self.model, height, width)
        
        # 定义要使用的相机列表 (必须与 XML 中的 name 一致)
        self.cam_names = ["cam_left", "cam_right", "cam_wrist"]
        
        # 预计算相机内参
        self.cam_intrinsics = {}
        for name in self.cam_names:
            try:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if cam_id == -1:
                    print(f"Warning: Camera {name} not found in XML!")
                    continue
                    
                fovy = self.model.cam_fovy[cam_id]
                f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2)
                cx = width / 2
                cy = height / 2
                
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
                self.cam_intrinsics[name] = K
            except Exception as e:
                print(f"Error initializing camera {name}: {e}")

        # [新增] 渲染器自检 (Self-Check)
        # 必须先 update_scene 才能 render。使用第一个相机或默认视角进行测试。
        try:
            mujoco.mj_forward(self.model, self.data)
            # 尝试用第一个有效相机渲染
            test_cam = self.cam_names[0] if self.cam_names else None
            if test_cam:
                self.renderer.update_scene(self.data, camera=test_cam)
            else:
                self.renderer.update_scene(self.data)
                
            check_img = self.renderer.render()
            
            if np.max(check_img) == 0:
                # 如果最大像素值是0，说明渲染器挂了 (OpenGL Context Error)
                raise RuntimeError("Renderer sanity check failed: Image is completely black (all zeros). OpenGL context might be broken.")
                
        except Exception as e:
            # 向上抛出异常，让 worker 捕获并重试
            raise RuntimeError(f"Renderer initialization failed: {e}")

    def get_images(self):
        """
        获取所有相机的 RGB 和 Depth 图像。
        严格分离 RGB 和 Depth 的渲染过程。
        """
        obs = {}
        
        for name in self.cam_names:
            try:
                # --- 1. 获取 RGB ---
                self.renderer.disable_depth_rendering() # 关键：确保关闭深度模式
                self.renderer.update_scene(self.data, camera=name)
                rgb = self.renderer.render() # (H, W, 3) uint8
                
                # --- 2. 获取 Depth ---
                self.renderer.enable_depth_rendering() # 开启深度模式
                self.renderer.update_scene(self.data, camera=name)
                depth = self.renderer.render() # (H, W) float32
                
                # 恢复默认状态
                self.renderer.disable_depth_rendering()
                
                obs[f'{name}_rgb'] = rgb
                obs[f'{name}_depth'] = depth
                
            except Exception as e:
                print(f"Render error {name}: {e}")
                obs[f'{name}_rgb'] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                obs[f'{name}_depth'] = np.zeros((self.height, self.width), dtype=np.float32)
                
        return obs

    def depth_to_pointcloud(self, depth, K, cam_pose_world):
        """
        [内部工具] 将单张深度图反投影为世界坐标系点云
        """
        rows, cols = depth.shape
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        u = u.flatten()
        v = v.flatten()
        d = depth.flatten()
        
        # 过滤无效深度
        valid = (d > 0.01) & (d < 3.0)
        u = u[valid]
        v = v[valid]
        d = d[valid]
        
        if len(d) == 0: return np.zeros((0, 3))
        
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        x_cam = (u - cx) * d / fx
        y_cam = (v - cy) * d / fy
        z_cam = d  
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        ones = np.ones((points_cam.shape[0], 1))
        points_cam_homo = np.hstack([points_cam, ones])
        points_world = (cam_pose_world @ points_cam_homo.T).T 
        return points_world[:, :3]

    def get_fused_pointcloud(self, link0_pos, link0_mat):
        """
        获取三相机融合点云，并转换到 Robot Link0 (基座) 坐标系。
        """
        all_points = []
        
        # World -> Link0 变换矩阵
        R_inv = link0_mat.T
        t_inv = -R_inv @ link0_pos
        T_world_to_link0 = np.eye(4)
        T_world_to_link0[:3, :3] = R_inv
        T_world_to_link0[:3, 3] = t_inv
        
        self.renderer.enable_depth_rendering()
        
        for name in self.cam_names:
            if name not in self.cam_intrinsics: continue
            
            # 渲染深度
            self.renderer.update_scene(self.data, camera=name)
            depth = self.renderer.render()
            
            # 获取位姿
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            cam_pos = self.data.cam_xpos[cam_id]
            cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3)
            
            # 修正矩阵 (MuJoCo -> CV)
            correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            cam_mat_cv = cam_mat @ correction
            
            cam_pose_world = np.eye(4)
            cam_pose_world[:3, :3] = cam_mat_cv
            cam_pose_world[:3, 3] = cam_pos
            
            pts_world = self.depth_to_pointcloud(depth, self.cam_intrinsics[name], cam_pose_world)
            all_points.append(pts_world)
            
        self.renderer.disable_depth_rendering()
            
        if not all_points:
            return np.zeros((0, 3), dtype=np.float32)
            
        full_cloud_world = np.concatenate(all_points, axis=0)
        
        # 转换到 Link0
        ones = np.ones((full_cloud_world.shape[0], 1))
        full_cloud_world_h = np.hstack([full_cloud_world, ones])
        full_cloud_local_h = (T_world_to_link0 @ full_cloud_world_h.T).T
        
        return full_cloud_local_h[:, :3].astype(np.float32)

    def get_observation_with_ar(self, n_points=4096):
        return np.zeros((n_points, 3))