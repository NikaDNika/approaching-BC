import torch
import torch.nn as nn

class TransformerPolicyWithAR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, action_dim=7, state_dim=7, ar_dim=6):
        super().__init__()
        self.point_embed = nn.Linear(3, d_model)
        self.state_embed = nn.Linear(state_dim + ar_dim, d_model)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, action_dim),
        )

    def forward(self, pc_scene, q_curr, ar_vec):
        B, N, _ = pc_scene.shape
        pc_tokens = self.point_embed(pc_scene) + self.coord_mlp(pc_scene)
        state_input = torch.cat([q_curr, ar_vec], dim=-1)
        state_token = self.state_embed(state_input).unsqueeze(1)
        tokens = torch.cat([state_token, pc_tokens], dim=1)
        enc_out = self.encoder(tokens)
        return self.head(enc_out[:, 0]) # 返回 delta_q