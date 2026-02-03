import torch
import torch.nn as nn
from src.models.modules import PointNet2Encoder

class IMPACTPolicy(nn.Module):
    def __init__(self, 
                 chunk_size=10, 
                 action_dim=7, 
                 state_dim=7, 
                 target_dim=7, # [注意] 目标是关节角，维度是7
                 embed_dim=256, 
                 n_heads=4, 
                 n_layers=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        # 1. 场景点云编码器 (Ps Encoder)
        self.scene_pc_encoder = PointNet2Encoder(feature_dim=embed_dim)
        
        # 2. [新增] 机器人点云编码器 (Pr Encoder)
        self.robot_pc_encoder = PointNet2Encoder(feature_dim=embed_dim)
        
        # 3. 状态编码器 (qc)
        self.state_emb = nn.Sequential(
            nn.Linear(state_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, embed_dim)
        )
        
        # 4. 目标编码器 (q_mg)
        self.target_emb = nn.Sequential(
            nn.Linear(target_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, embed_dim)
        )
        
        # Action Queries & Positional Embedding
        self.action_queries = nn.Parameter(torch.zeros(chunk_size, embed_dim))
        self.query_pos = nn.Parameter(torch.rand(chunk_size, embed_dim))
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 输出头
        self.action_head = nn.Linear(embed_dim, action_dim)

    def forward(self, ps, pr, q_curr, q_goal):
        """
        ps: Scene Point Cloud (B, Ns, 3)
        pr: Robot Point Cloud (B, Nr, 3)
        q_curr: Current Joint (B, 7)
        q_goal: Goal Joint (B, 7)
        """
        batch_size = ps.shape[0]
        
        # --- Encoding ---
        # 1. Point Cloud Features
        ps_feat = self.scene_pc_encoder(ps) # (B, dim)
        pr_feat = self.robot_pc_encoder(pr) # (B, dim) [New]
        
        # 2. State & Target
        q_feat = self.state_emb(q_curr)     # (B, dim)
        g_feat = self.target_emb(q_goal)    # (B, dim)
        
        # --- Fusion ---
        # Memory tokens: [Scene_PC, Robot_PC, Curr_Joint, Goal_Joint]
        # Transformer 的 memory 序列长度为 4
        memory = torch.stack([ps_feat, pr_feat, q_feat, g_feat], dim=1) # (B, 4, dim)
        
        # --- Decoding ---
        # 准备 Queries: (Chunk, Dim) -> (B, Chunk, Dim)
        tgt = self.action_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        tgt_pos = self.query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer Decoder
        # tgt: Queries (Output positions)
        # memory: Context (PC + State + Target)
        output = self.transformer_decoder(tgt + tgt_pos, memory) # (B, Chunk, embed_dim)
        
        # 预测动作
        pred_actions = self.action_head(output) # (B, Chunk, 7)
        
        return pred_actions