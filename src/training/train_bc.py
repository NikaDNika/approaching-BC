import os
import time
import yaml  # 确保安装: pip install pyyaml
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# 根据你的项目结构调整 import 路径
# 如果在根目录运行: from dataset import ApproachDataset
# 如果作为 package 安装: from src.data.dataset import ApproachDataset
try:
    from src.data.dataset import ApproachDataset 
    from src.models.impact import IMPACTPolicy
except ImportError:
    # Fallback for flat directory structure
    from data.dataset import ApproachDataset
    from models.impact import IMPACTPolicy

def train_drp():
    # --- 1. 实验配置 (Hyperparameters) ---
    config = {
        "exp_name": "drp_bc_baseline",  # 实验名称
        "data_dir": "/media/j16/8deb00a4-cceb-e842-a899-55532424da473/dataset_v2",
        "checkpoint_root": "checkpoints", # 根保存路径
        
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        
        "chunk_size": 10,
        "action_dim": 7,
        "state_dim": 7,
        "target_dim": 7,
        
        "num_points": 2048,
        "num_robot_points": 512,
        
        "num_workers": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # --- 2. 创建实验目录 ---
    # 格式: checkpoints/{exp_name}_{timestamp}/
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(config['checkpoint_root'], f"{config['exp_name']}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"\n{'='*40}")
    print(f"Experiment: {config['exp_name']}")
    print(f"Output Dir: {ckpt_dir}")
    print(f"{'='*40}\n")
    
    # 保存配置到 yaml
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # --- 3. 数据加载 ---
    print("Loading Dataset...")
    train_set = ApproachDataset(
        data_dir=config['data_dir'], 
        chunk_size=config['chunk_size'], 
        num_points=config['num_points'],
        num_robot_points=config['num_robot_points']
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    
    # --- 4. 模型构建 ---
    print("Building IMPACT Model...")
    model = IMPACTPolicy(
        chunk_size=config['chunk_size'], 
        action_dim=config['action_dim'], 
        state_dim=config['state_dim'], 
        target_dim=config['target_dim']
    ).to(config['device'])
    
    # --- 5. 优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # --- 6. 训练循环 ---
    print(f"Start Training on {config['device']}...")
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (pc, robot_pc, q, target, gt_action, is_pad) in enumerate(train_loader):
            # Move to device
            pc = pc.to(config['device'])
            robot_pc = robot_pc.to(config['device'])
            q = q.to(config['device'])
            target = target.to(config['device'])
            gt_action = gt_action.to(config['device'])
            is_pad = is_pad.to(config['device'])
            
            # Forward
            pred_action = model(pc, robot_pc, q, target) 
            
            # Loss Calculation (Masked MSE)
            # reduction='none' -> (B, Chunk, 7)
            loss = F.mse_loss(pred_action, gt_action, reduction='none')
            # Average over action dim -> (B, Chunk)
            loss = loss.mean(dim=2) 
            
            # Apply Mask (Ignore padded steps)
            mask = 1.0 - is_pad
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print log
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
        
        # Scheduler step
        scheduler.step()
        
        # Epoch Summary
        avg_loss = total_loss / len(train_loader)
        print(f">>> Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.6f}")
        
        # --- 7. 模型保存策略 ---
        
        # A. 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"    [Saved] New best model! (Loss: {best_loss:.6f})")
            
        # B. 定期保存 (每10轮)
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_path)
            print(f"    [Saved] Checkpoint: {epoch_path}")
            
        # C. 保存最新状态 (用于断点续训)
        last_path = os.path.join(ckpt_dir, "last.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'config': config
        }, last_path)

    print("\nTraining Complete.")

if __name__ == "__main__":
    train_drp()