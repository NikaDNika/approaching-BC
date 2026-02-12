# src/training/train_bc_with_ar_transformer_multistep.py
import os
import time
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ----------------- world -> link0（与采集/AR训练保持一致） -----------------
def world_to_link0(object_pos_world, goal_pos_world, robot_base):
    object_pos_world = object_pos_world.astype(np.float32)
    goal_pos_world   = goal_pos_world.astype(np.float32)
    robot_base       = robot_base.astype(np.float32)

    base_x, base_y, base_theta = robot_base
    base_pos = np.array([base_x, base_y, 0.15], dtype=np.float32)  # mobile_base.pos.z = 0.15
    offset  = np.array([0.1, 0.0, 0.2], dtype=np.float32)          # link0 相对 base 的偏移

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


# ----------------- Dataset: 多步窗口 + 预计算 AR（优先用 ar_pred_link0） -----------------
class BCMultiStepARDataset(Dataset):
    """
    从数据集构造多步训练样本:
      必须包含:
        pc_ar      : (T, N_raw, 3) in link0
        q_arm      : (T, 7)
        object_pos : (3,) in world
        goal_pos   : (3,) in world
        robot_base : (3,) [base_x, base_y, base_theta]

      如果存在:
        ar_pred_link0 : (6,) = [obj_pred_link0, goal_pred_link0]
      则使用它作为 AR 输入（推荐：来自 precompute_ar_pred_for_bc.py）；
      否则退回用 world_to_link0(object_pos, goal_pos, robot_base) 计算 GT AR。

    对每条轨迹，按窗口长度 K+1 构造:
      窗口起点 t0: 0..T-K-1
      返回:
        pcs   : (K, num_points, 3)  # t0..t0+K-1 的点云
        qs    : (K+1, 7)            # t0..t0+K 的关节
        ar    : (6,)                # 本轨迹固定 AR = ar_pred_link0 或 GT AR
    """
    def __init__(self, data_dir, num_points=1024, K=5):
        self.num_points = num_points
        self.K = K

        self.files = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        if len(self.files) == 0:
            raise RuntimeError(f"[BCMultiStepARDataset] No npz files found in {data_dir}")
        print(f"[BCMultiStepARDataset] Found {len(self.files)} trajectories.")

        self.windows = []  # list of (file_idx, t_start)

        for fi, path in enumerate(self.files):
            try:
                with np.load(path) as d:
                    q_seq = d["q_arm"]  # (T,7)
                T = len(q_seq)
                if T < self.K + 1:
                    continue
                for t0 in range(0, T - self.K):
                    self.windows.append((fi, t0))
            except Exception as e:
                print(f"[BCMultiStepARDataset] Warning: skip file {path}: {e}")
                continue

        if len(self.windows) == 0:
            raise RuntimeError("[BCMultiStepARDataset] No valid windows found.")
        print(f"[BCMultiStepARDataset] Total windows: {len(self.windows)}")

        self.data_dir = data_dir

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fi, t0 = self.windows[idx]
        path = self.files[fi]

        try:
            with np.load(path) as d:
                pc_seq = d["pc_ar"]          # (T, N_raw, 3)
                q_seq  = d["q_arm"]          # (T, 7)

                # 如果存在 ar_pred_link0，优先使用它
                if "ar_pred_link0" in d.files:
                    ar_vec = d["ar_pred_link0"].astype(np.float32)  # (6,)
                else:
                    # 退回到根据 GT object/goal 计算 AR（兼容老数据）
                    object_pos_world = d["object_pos"].astype(np.float32)
                    goal_pos_world   = d["goal_pos"].astype(np.float32)
                    robot_base       = d["robot_base"].astype(np.float32)
                    obj_link0, goal_link0 = world_to_link0(
                        object_pos_world, goal_pos_world, robot_base
                    )
                    ar_vec = np.concatenate([obj_link0, goal_link0], axis=0).astype(np.float32)  # (6,)

        except Exception as e:
            print(f"[BCMultiStepARDataset] Error loading {path}: {e}")
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        T = len(q_seq)
        if T < self.K + 1:
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        t1 = t0 + self.K  # inclusive index for q

        # 点云窗口 t0..t0+K-1
        pcs_list = []
        for t in range(t0, t0 + self.K):
            pc_t = pc_seq[t]  # (N_raw, 3)
            if len(pc_t) >= self.num_points:
                idx_pc = np.random.choice(len(pc_t), self.num_points, replace=False)
            else:
                idx_pc = np.random.choice(len(pc_t), self.num_points, replace=True)
            pcs_list.append(pc_t[idx_pc].astype(np.float32))
        pcs = np.stack(pcs_list, axis=0)  # (K, num_points, 3)

        # 关节窗口 t0..t0+K
        qs = q_seq[t0:t1+1].astype(np.float32)  # (K+1,7)

        return (
            torch.from_numpy(pcs),        # (K,N,3)
            torch.from_numpy(qs),         # (K+1,7)
            torch.from_numpy(ar_vec),     # (6,)
        )


# ----------------- Transformer Policy -----------------
class TransformerPolicyWithAR(nn.Module):
    """
    Transformer 风格策略:
      - 把每个点云点作为一个 token: embed: R^3 -> R^d
      - 把 [q, AR] 作为一个 state token: R^(7+6) -> R^d
      - tokens = [state_token, pc_tokens...] 共 (N+1) 个
      - 通过 TransformerEncoder 多层 self-attn
      - 取 state token 的输出做一个 MLP head -> Δq

    参考 HumanX 中 token 化的思路，但这里 token 是 {state, points}。
    """
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        action_dim=7,
        state_dim=7,
        ar_dim=6,
    ):
        super().__init__()
        self.d_model = d_model

        # 点云点的线性嵌入
        self.point_embed = nn.Linear(3, d_model)

        # [q, AR] 的 state token 嵌入
        self.state_embed = nn.Linear(state_dim + ar_dim, d_model)

        # 简单“坐标位置编码”（MLP 处理 xyz 后加到点 token 上）
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B,S,E)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 用 state token 输出做 head
        self.head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, pc_scene, q_curr, ar_vec):
        """
        pc_scene: (B,N,3)
        q_curr  : (B,7)
        ar_vec  : (B,6)
        """
        B, N, _ = pc_scene.shape

        # 1) 点云 token
        pc_embed = self.point_embed(pc_scene)         # (B,N,d)
        coord_enc = self.coord_mlp(pc_scene)         # (B,N,d)
        pc_tokens = pc_embed + coord_enc             # (B,N,d)

        # 2) state token
        state_input = torch.cat([q_curr, ar_vec], dim=-1)   # (B,13)
        state_token = self.state_embed(state_input).unsqueeze(1)  # (B,1,d)

        # 3) 拼接 tokens: [state, p1, p2, ..., pN]
        tokens = torch.cat([state_token, pc_tokens], dim=1)  # (B, N+1, d)

        # 4) Transformer Encoder
        enc_out = self.encoder(tokens)                     # (B,N+1,d)

        # 5) 取 state token 的输出 -> MLP head
        state_out = enc_out[:, 0]                          # (B,d)
        delta_q = self.head(state_out)                     # (B,7)
        return delta_q


# ----------------- 训练：多步 + Scheduled Sampling -----------------
def train_bc_with_ar_transformer_multistep():
    config = {
        "exp_name": "bc_with_ar_transformer_multistep_predAR",
        # 建议指向包含 ar_pred_link0 的数据集
        "data_dir": "/media/j16/8deb00a4-cceb-e842-a899-55532424da473/dataset_v2_with_ar_pred",
        "checkpoint_root": "checkpoints",

        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 100,

        "num_points": 1024,   # Transformer 负载较大，可适当减点
        "K": 5,               # 窗口长度

        # Transformer 结构参数
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 512,
        "dropout": 0.1,

        "num_workers": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(
        config["checkpoint_root"],
        f"{config['exp_name']}_{timestamp}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[BC-TX] Output dir: {ckpt_dir}")

    # Dataset & Loader
    dataset = BCMultiStepARDataset(
        data_dir=config["data_dir"],
        num_points=config["num_points"],
        K=config["K"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    device = torch.device(config["device"])

    model = TransformerPolicyWithAR(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        action_dim=7,
        state_dim=7,
        ar_dim=6,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
    )

    best_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        total_batches = 0

        # Scheduled Sampling: teacher forcing 概率线性从 1.0 -> 0.1
        tf_start = 1.0
        tf_end   = 0.1
        progress = epoch / max(1, (config["epochs"] - 1))
        teacher_prob = float(tf_start + (tf_end - tf_start) * progress)
        print(f"[BC-TX] Epoch {epoch+1}, teacher forcing prob={teacher_prob:.3f}")

        for i, (pcs, qs, ar_vec) in enumerate(loader):
            # pcs: (B,K,N,3)
            # qs : (B,K+1,7)
            # ar_vec: (B,6)
            pcs = pcs.to(device)        # (B,K,N,3)
            qs  = qs.to(device)         # (B,K+1,7)
            ar_vec = ar_vec.to(device)  # (B,6)

            B, K, N, _ = pcs.shape
            q_gt = qs                   # (B,K+1,7)
            q_hat = q_gt[:, 0].clone()  # (B,7) 初始为 q_t0

            loss = 0.0

            for k in range(K):
                pc_k = pcs[:, k]        # (B,N,3)

                # Scheduled Sampling: 按 teacher_prob 决定用 GT 还是 q_hat
                if teacher_prob >= 1.0:
                    q_in = q_gt[:, k]
                elif teacher_prob <= 0.0:
                    q_in = q_hat
                else:
                    mask = (torch.rand(B, 1, device=device) < teacher_prob).float()
                    q_in = mask * q_gt[:, k] + (1.0 - mask) * q_hat

                delta_hat = model(pc_k, q_in, ar_vec)   # (B,7)

                # 可选：限幅，减缓训练中梯度爆炸
                max_step = 0.1  # rad
                delta_hat = torch.clamp(delta_hat, -max_step, max_step)

                q_hat = q_in + delta_hat  # 下一步预测的关节 (B,7)

                # 监督 q_hat 接近 q_gt[:, k+1]
                loss_step = F.mse_loss(q_hat, q_gt[:, k+1])
                loss += loss_step

            loss = loss / K

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if i % 20 == 0:
                print(
                    f"[BC-TX][Epoch {epoch+1}/{config['epochs']}] "
                    f"[{i}/{len(loader)}] Loss={loss.item():.6f}"
                )

        scheduler.step()

        avg_loss = total_loss / max(1, total_batches)
        print(f"[BC-TX] Epoch {epoch+1} Avg Loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[BC-TX]   Saved new best model: {best_path}")

        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_path)
            print(f"[BC-TX]   Saved checkpoint: {epoch_path}")

    print("[BC-TX] Training complete.")


if __name__ == "__main__":
    train_bc_with_ar_transformer_multistep()