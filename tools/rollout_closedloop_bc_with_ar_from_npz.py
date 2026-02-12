#!/usr/bin/env python
# tools/rollout_closedloop_bc_with_ar_from_npz.py
import os
import time
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco
import mujoco.viewer as mjviewer
from scipy.spatial.transform import Rotation as R
from src.training.train_bc_with_ar_transformer_multistep import TransformerPolicyWithAR


# ----------------- world -> link0（与训练/采集时保持一致） -----------------
def world_to_link0(object_pos_world, goal_pos_world, robot_base):
    """
    输入:
        object_pos_world: (3,) 世界坐标
        goal_pos_world:   (3,) 世界坐标
        robot_base:       (3,) [base_x, base_y, base_theta]
    输出:
        obj_link0:  (3,)
        goal_link0:(3,)
    """
    object_pos_world = object_pos_world.astype(np.float32)
    goal_pos_world   = goal_pos_world.astype(np.float32)
    robot_base       = robot_base.astype(np.float32)

    base_x, base_y, base_theta = robot_base
    # 与 mobile_panda.xml 一致：mobile_base.pos.z = 0.15
    base_pos = np.array([base_x, base_y, 0.15], dtype=np.float32)
    # link0 相对 base 的偏移：<body name="link0" pos="0.1 0 0.2">
    offset = np.array([0.1, 0.0, 0.2], dtype=np.float32)

    cos_t = np.cos(base_theta)
    sin_t = np.sin(base_theta)
    Rz = np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0],
        ],
        dtype=np.float32,
    )

    link0_pos_world = base_pos + Rz @ offset

    obj_link0  = Rz.T @ (object_pos_world - link0_pos_world)
    goal_link0 = Rz.T @ (goal_pos_world   - link0_pos_world)
    return obj_link0, goal_link0


# ----------------- 深度图 -> 点云（与 MujocoSimEnv 一致） -----------------
def depth_to_pointcloud(depth, K, cam_pose_world):
    """
    depth: (H, W), float32
    K: (3, 3)
    cam_pose_world: (4, 4)
    """
    rows, cols = depth.shape
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    u = u.flatten()
    v = v.flatten()
    d = depth.flatten()

    valid = (d > 0.01) & (d < 3.0)
    u = u[valid]
    v = v[valid]
    d = d[valid]
    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)  # (N,3)
    ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
    points_cam_h = np.hstack([points_cam, ones])                             # (N,4)
    points_world_h = (cam_pose_world @ points_cam_h.T).T                     # (N,4)
    return points_world_h[:, :3].astype(np.float32)


def get_fused_pointcloud_link0(model, data, renderer, cam_names, cam_intrinsics,
                               link0_pos, link0_mat, height, width):
    """
    从当前仿真状态下，多相机深度融合得到点云，并转换到 link0 坐标系。
    """
    all_points = []

    # world -> link0
    R_inv = link0_mat.T
    t_inv = -R_inv @ link0_pos
    T_world_to_link0 = np.eye(4, dtype=np.float32)
    T_world_to_link0[:3, :3] = R_inv
    T_world_to_link0[:3, 3] = t_inv

    renderer.enable_depth_rendering()

    for name in cam_names:
        if name not in cam_intrinsics:
            continue
        try:
            renderer.update_scene(data, camera=name)
            depth = renderer.render()  # (H,W), float32
        except Exception as e:
            print(f"[PointCloud] Render error for camera {name}: {e}")
            continue

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0:
            continue
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()

        # MuJoCo -> CV 坐标修正
        correction = np.array([[1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]], dtype=np.float32)
        cam_mat_cv = cam_mat @ correction

        cam_pose_world = np.eye(4, dtype=np.float32)
        cam_pose_world[:3, :3] = cam_mat_cv
        cam_pose_world[:3, 3] = cam_pos

        K = cam_intrinsics[name]
        pts_world = depth_to_pointcloud(depth, K, cam_pose_world)
        all_points.append(pts_world)

    renderer.disable_depth_rendering()

    if not all_points:
        return np.zeros((0, 3), dtype=np.float32)

    full_world = np.concatenate(all_points, axis=0)   # (M,3)
    ones = np.ones((full_world.shape[0], 1), dtype=np.float32)
    full_world_h = np.hstack([full_world, ones])      # (M,4)
    full_link0_h = (T_world_to_link0 @ full_world_h.T).T  # (M,4)
    return full_link0_h[:, :3].astype(np.float32)


# ----------------- BC Policy 模型（与 train_bc_with_ar.py 一致） -----------------
class PointNetEncoderBC(nn.Module):
    def __init__(self, input_dim=3, feat_dim=128):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, feat_dim)

    def forward(self, x):
        # x: (B,N,3)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        x = x.max(dim=1)[0]
        return x


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, feat_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, feat_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BCPolicyWithAR(nn.Module):
    """
    输入:
        pc_scene: (B,N,3)
        q_curr:   (B,7)
        ar_vec:   (B,6) = [obj_link0(3), goal_link0(3)]
    输出:
        action:   (B,7) = q_{t+1} - q_t
    """
    def __init__(self,
                 action_dim=7,
                 state_dim=7,
                 ar_dim=6,
                 scene_feat_dim=128,
                 state_feat_dim=64,
                 ar_feat_dim=64,
                 hidden_dim=256):
        super().__init__()
        self.scene_encoder = PointNetEncoderBC(3, scene_feat_dim)
        self.state_encoder = MLPEncoder(state_dim, state_feat_dim)
        self.ar_encoder    = MLPEncoder(ar_dim, ar_feat_dim)

        fusion_dim = scene_feat_dim + state_feat_dim + ar_feat_dim
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, pc_scene, q_curr, ar_vec):
        sf = self.scene_encoder(pc_scene)
        st = self.state_encoder(q_curr)
        af = self.ar_encoder(ar_vec)
        fused = torch.cat([sf, st, af], dim=-1)
        action = self.policy_head(fused)
        return action


# ----------------- 从 scene_xml 构建 Mujoco 模型 -----------------
def model_from_scene_xml(scene_xml_str: str):
    model = mujoco.MjModel.from_xml_string(scene_xml_str)
    data = mujoco.MjData(model)
    return model, data


# ----------------- 主函数：闭环 roll-out，使用 GT AR（来自 npz） -----------------
def rollout_closedloop_bc_with_ar_from_npz(
    data_dir,
    scenario,
    policy_ckpt,
    num_points=2048,
    device="cuda",
    control_dt=0.05,
    max_steps=None,
    grab_xy_thresh=0.05,
    grab_z_min=0.05,
    release_xy_thresh=0.05,
    release_z_min=0.05,
    img_width=320,
    img_height=240,
):
    if device == "cuda" and not torch.cuda.is_available():
        print("[Rollout] CUDA 不可用，自动切到 CPU")
        device = "cpu"
    device = torch.device(device)

    # 1. 加载策略
    print(f"[Rollout] Loading policy checkpoint: {policy_ckpt}")
    policy_state = torch.load(policy_ckpt, map_location=device)
    policy = TransformerPolicyWithAR(
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        action_dim=7,
        state_dim=7,
        ar_dim=6,
    ).to(device)
    policy.load_state_dict(policy_state)
    policy.eval()

    # 2. 从指定场景目录中随机选一条 npz
    if scenario is not None:
        scen_dir = os.path.join(data_dir, scenario)
        if not os.path.exists(scen_dir):
            raise RuntimeError(f"[Rollout] Scenario dir not found: {scen_dir}")
        pattern = os.path.join(scen_dir, "*.npz")
        print(f"[Rollout] Searching npz in scenario dir: {scen_dir}")
        recursive = False
    else:
        pattern = os.path.join(data_dir, "**", "*.npz")
        print(f"[Rollout] No scenario specified, searching npz in: {data_dir} (recursive)")
        recursive = True

    files = glob.glob(pattern, recursive=recursive)
    if len(files) == 0:
        raise RuntimeError(f"[Rollout] No npz files found for pattern: {pattern}")

    traj_path = np.random.choice(files)
    print(f"[Rollout] Using trajectory: {traj_path}")

    with np.load(traj_path, allow_pickle=True) as d:
        q_seq  = d["q_arm"]          # (T, 7)
        object_pos_world = d["object_pos"].astype(np.float32)  # (3,)
        goal_pos_world   = d["goal_pos"].astype(np.float32)    # (3,)
        robot_base       = d["robot_base"].astype(np.float32)  # (3,)
        scene_xml_arr    = d["scene_xml"]

        if isinstance(scene_xml_arr, np.ndarray):
            scene_xml = str(scene_xml_arr.item())
        else:
            scene_xml = str(scene_xml_arr)

    T = len(q_seq)
    if T <= 1:
        raise RuntimeError(f"[Rollout] Trajectory too short T={T}")

    if max_steps is None:
        max_steps = T - 1

    # 3. 计算 GT AR 在 link0 下的坐标（固定目标）
    obj_link0_gt, goal_link0_gt = world_to_link0(
        object_pos_world, goal_pos_world, robot_base
    )
    print(f"[Rollout] GT obj_link0={obj_link0_gt}, goal_link0={goal_link0_gt}")

    ar_vec_gt = np.concatenate([obj_link0_gt, goal_link0_gt], axis=0).astype(np.float32)
    ar_vec_tensor = torch.from_numpy(ar_vec_gt).unsqueeze(0).to(device)  # (1,6)

    # 4. 根据 scene_xml 构建 Mujoco 模型
    model, data = model_from_scene_xml(scene_xml)

    ARM_QPOS_START = 3
    ARM_DOF = 7
    FINGER_QPOS_START = 10

    data.qvel[:] = 0.0
    data.qpos[0:3] = robot_base
    data.qpos[ARM_QPOS_START:ARM_QPOS_START + ARM_DOF] = q_seq[0]
    data.qpos[FINGER_QPOS_START:FINGER_QPOS_START + 2] = 0.04
    mujoco.mj_forward(model, data)

    # link0 / tcp / obj / goal_viz
    try:
        link0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    except Exception:
        link0_id = -1
        print("[Rollout] Warning: body 'link0' not found.")

    try:
        tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    except Exception:
        tcp_id = -1
        print("[Rollout] Warning: site 'tcp_site' not found.")

    try:
        obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    except Exception:
        obj_body_id = -1
        print("[Rollout] Warning: body 'obj' not found.")

    if obj_body_id != -1:
        obj_jnt_id = model.body_jntadr[obj_body_id]
        obj_qpos_adr = model.jnt_qposadr[obj_jnt_id]
        object_pos_world_sim = data.xpos[obj_body_id].copy()
        print(f"[Rollout] Initial object_pos_world (sim) = {object_pos_world_sim}")
    else:
        obj_qpos_adr = None
        object_pos_world_sim = None

    obj_ar_geom_ids = []
    if obj_body_id != -1:
        for g_id in range(model.body_geomadr[obj_body_id],
                          model.body_geomadr[obj_body_id] + model.body_geomnum[obj_body_id]):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g_id)
            if geom_name and geom_name.startswith("obj_grasp"):
                obj_ar_geom_ids.append(g_id)

    try:
        goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_viz")
    except Exception:
        goal_body_id = -1
        print("[Rollout] Warning: body 'goal_viz' not found.")

    if goal_body_id != -1:
        goal_pos_world_sim = data.xpos[goal_body_id].copy()
        print(f"[Rollout] Initial goal_pos_world (sim) = {goal_pos_world_sim}")
        goal_ar_geom_ids = list(range(model.body_geomadr[goal_body_id],
                                      model.body_geomadr[goal_body_id] + model.body_geomnum[goal_body_id]))
    else:
        goal_pos_world_sim = None
        goal_ar_geom_ids = []

    # 5. 渲染器 + 相机内参（与 MujocoSimEnv 一致）
    renderer = mujoco.Renderer(model, img_height, img_width)
    cam_names = ["cam_left", "cam_right", "cam_wrist"]
    cam_intrinsics = {}
    for name in cam_names:
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        except Exception:
            cam_id = -1
        if cam_id < 0:
            print(f"[Rollout] Camera {name} not found, skip.")
            continue
        fovy = model.cam_fovy[cam_id]
        f = 0.5 * img_height / np.tan(np.deg2rad(fovy) / 2.0)
        cx = img_width / 2.0
        cy = img_height / 2.0
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float32)
        cam_intrinsics[name] = K
    cam_names = [n for n in cam_names if n in cam_intrinsics]

    # 6. 假抓取 / 放置 状态
    grabbed = False
    released = False

    # 7. 闭环 roll-out（完全由模型自驱；专家轨迹仅用于误差统计）
    q_curr = q_seq[0].copy()

    with mjviewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        print(
            f"[Rollout] Closed-loop (GT AR) started. "
            f"Scenario={scenario or 'ALL'}, "
            "policy input uses explicit GT AR from npz."
        )

        for step in range(max_steps):
            if not viewer.is_running():
                print("[Rollout] Viewer closed by user.")
                break

            # 7.1 写入当前 q_curr 到仿真
            data.qpos[0:3] = robot_base
            data.qpos[ARM_QPOS_START:ARM_QPOS_START + ARM_DOF] = q_curr
            data.qpos[FINGER_QPOS_START:FINGER_QPOS_START + 2] = 0.04
            mujoco.mj_forward(model, data)

            # 7.2 计算 link0 姿态
            if link0_id != -1:
                link0_pos = data.xpos[link0_id].copy()
                link0_mat = data.xmat[link0_id].reshape(3, 3).copy()
            else:
                link0_pos = np.zeros(3, dtype=np.float32)
                link0_mat = np.eye(3, dtype=np.float32)

            # 7.3 获取当前 fused 点云（link0）
            pc_link0 = get_fused_pointcloud_link0(
                model, data, renderer, cam_names, cam_intrinsics,
                link0_pos, link0_mat, img_height, img_width
            )

            if len(pc_link0) >= num_points:
                idx_pc = np.random.choice(len(pc_link0), num_points, replace=False)
                pc = pc_link0[idx_pc]
            else:
                if len(pc_link0) > 0:
                    pad = np.repeat(pc_link0[-1][None, :], num_points - len(pc_link0), axis=0)
                    pc = np.concatenate([pc_link0, pad], axis=0)
                else:
                    pc = np.zeros((num_points, 3), dtype=np.float32)

            pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)
            q_tensor  = torch.from_numpy(q_curr).float().unsqueeze(0).to(device)

            # 7.4 Policy：点云 + 当前 q + 固定 GT AR → Δq
            with torch.no_grad():
                action_pred = policy(pc_tensor, q_tensor, ar_vec_tensor)[0].cpu().numpy()

            # 可选：对 Δq 限幅，减缓爆炸（如每步不超过 0.05 rad）
            max_step_rad = 0.05
            action_pred = np.clip(action_pred, -max_step_rad, max_step_rad)

            q_next_pred = q_curr + action_pred

            # 7.5 假抓取 / 放置（吸附）
            if tcp_id != -1:
                tcp_pos = data.site_xpos[tcp_id].copy()
                tcp_mat = data.site_xmat[tcp_id].reshape(3, 3)
                r_tcp = R.from_matrix(tcp_mat)
                quat_xyzw = r_tcp.as_quat()
                tcp_quat = np.array(
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                    dtype=np.float64,
                )
            else:
                tcp_pos = None
                tcp_quat = None

            if tcp_pos is not None and obj_qpos_adr is not None and object_pos_world is not None:
                # 抓取判定：在物体上方一定范围内
                if not grabbed:
                    d_obj = tcp_pos - object_pos_world
                    xy_dist = np.linalg.norm(d_obj[:2])
                    z_diff  = d_obj[2]
                    if xy_dist < grab_xy_thresh and z_diff > grab_z_min:
                        grabbed = True
                        print(f"[Rollout] Fake GRAB at step={step}, xy_dist={xy_dist:.3f}, z_diff={z_diff:.3f}")
                        for g_id in obj_ar_geom_ids:
                            model.geom_rgba[g_id][3] = 0.0  # 隐藏物体 AR

                # 放置判定：在目标上方一定范围内
                if grabbed and not released and goal_pos_world is not None:
                    d_goal = tcp_pos - goal_pos_world
                    xy_dist_g = np.linalg.norm(d_goal[:2])
                    z_diff_g  = d_goal[2]
                    if xy_dist_g < release_xy_thresh and z_diff_g > release_z_min:
                        released = True
                        print(f"[Rollout] Fake RELEASE at step={step}, xy_dist={xy_dist_g:.3f}, z_diff={z_diff_g:.3f}")
                        for g_id in goal_ar_geom_ids:
                            model.geom_rgba[g_id][3] = 0.0  # 隐藏放置 AR 球

                # 物体吸附 TCP
                if grabbed and not released:
                    data.qpos[obj_qpos_adr:obj_qpos_adr + 3] = tcp_pos
                    if tcp_quat is not None:
                        data.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = tcp_quat
                    mujoco.mj_forward(model, data)

            # 7.6 误差统计（对专家下一步，仅日志用）
            t_gt = min(step + 1, T - 1)
            q_next_gt = q_seq[t_gt]
            err = np.linalg.norm(q_next_pred - q_next_gt)
            print(
                f"[Rollout] step={step}/{max_steps-1}, "
                f"||q_next_pred - q_next_gt|| = {err:.4f} rad, "
                f"grabbed={grabbed}, released={released}"
            )

            q_curr = q_next_pred.copy()

            viewer.sync()
            time.sleep(control_dt)

        print("[Rollout] Closed-loop rollout with GT AR finished.")


# ----------------- 命令行 -----------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop rollout of BC-AR policy using explicit GT AR from npz "
            "(no AR segmentation), with fake grasp/placement."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of dataset_v2 containing scenario subfolders.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name (e.g., living_room, kitchen, storage, corner). "
             "If omitted, sample from all scenarios.",
    )
    parser.add_argument(
        "--policy-ckpt",
        type=str,
        required=True,
        help="Path to BC-AR checkpoint (.pth).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of points to subsample from each point cloud frame.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max rollout steps. Default: T-1 of chosen expert traj.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.05,
        help="Time delay (s) between two control steps in the viewer.",
    )
    parser.add_argument(
        "--grab-xy-thresh",
        type=float,
        default=0.05,
        help="Fake grasp XY threshold (m).",
    )
    parser.add_argument(
        "--grab-z-min",
        type=float,
        default=0.05,
        help="Fake grasp minimum Z offset (m).",
    )
    parser.add_argument(
        "--release-xy-thresh",
        type=float,
        default=0.05,
        help="Fake release XY threshold (m).",
    )
    parser.add_argument(
        "--release-z-min",
        type=float,
        default=0.05,
        help="Fake release minimum Z offset (m).",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=320,
        help="Offscreen render width for depth maps.",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=240,
        help="Offscreen render height for depth maps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rollout_closedloop_bc_with_ar_from_npz(
        data_dir=args.data_dir,
        scenario=args.scenario,
        policy_ckpt=args.policy_ckpt,
        num_points=args.num_points,
        device=args.device,
        control_dt=args.control_dt,
        max_steps=args.max_steps,
        grab_xy_thresh=args.grab_xy_thresh,
        grab_z_min=args.grab_z_min,
        release_xy_thresh=args.release_xy_thresh,
        release_z_min=args.release_z_min,
        img_width=args.img_width,
        img_height=args.img_height,
    )