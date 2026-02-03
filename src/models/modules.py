import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """ 计算点对距离，用于最远点采样 (FPS) 和 球查询 (Ball Query) """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """ 根据索引取点 """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """ 最远点采样 (FPS) """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """ 球查询: 找邻居 """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    """ PointNet++ 的核心 Set Abstraction Layer """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3) 坐标
        points: (B, N, C) 特征 (可以为None)
        """
        xyz = xyz.permute(0, 2, 1) # -> (B, 3, N)
        if points is not None:
            points = points.permute(0, 2, 1) # -> (B, C, N)

        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 3, 1).to(xyz.device)
            grouped_xyz = xyz.view(xyz.shape[0], 3, 1, -1)
            if points is not None:
                new_points = torch.cat([grouped_xyz, points.view(points.shape[0], -1, 1, points.shape[2])], dim=1)
            else:
                new_points = grouped_xyz
        else:
            # FPS Sampling
            xyz_flipped = xyz.permute(0, 2, 1)
            new_xyz_idx = farthest_point_sample(xyz_flipped, self.npoint)
            new_xyz = index_points(xyz_flipped, new_xyz_idx).permute(0, 2, 1)
            
            # Grouping
            idx = query_ball_point(self.radius, self.nsample, xyz_flipped, new_xyz.permute(0, 2, 1))
            grouped_xyz = index_points(xyz_flipped, idx).permute(0, 3, 1, 2) # [B, 3, npoint, nsample]
            grouped_xyz -= new_xyz.view(new_xyz.shape[0], 3, self.npoint, 1) # 相对坐标
            
            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), idx).permute(0, 3, 1, 2)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=1)
            else:
                new_points = grouped_xyz

        # PointNet Layer
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 3)[0] # Max Pooling
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        
        return new_xyz, new_points

class PointNet2Encoder(nn.Module):
    """ 简化的 PointNet++ 编码器，输出全局特征向量 """
    def __init__(self, normal_channel=False, feature_dim=256):
        super().__init__()
        in_channel = 3 if normal_channel else 0
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel+3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, feature_dim], group_all=True)

    def forward(self, xyz):
        # xyz: (B, N, 3)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l3_points: (B, 1, feature_dim)
        return l3_points.squeeze(1)