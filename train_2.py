"""
train_adaptive_alpha_with_pyg.py
修正版训练脚本（针对预处理后的点云 .pt 文件优化），采用批量化 knn 奖励计算。
依赖: torch, torch_geometric (pyg), torch_cluster, torch_scatter, pytorch3d (可选用于原始 mesh 采样), trimesh (原始 mesh 解析可选)
"""

import os
import glob
import random
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 如果需要 knn_graph / scatter_mean
try:
    from torch_cluster import knn_graph
except Exception as e:
    knn_graph = None
    # We'll check and warn later.

try:
    from torch_scatter import scatter_mean
except Exception:
    scatter_mean = None

# Optional utilities for fallback sampling from meshes (if user didn't preprocess)
try:
    import trimesh
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
except Exception:
    trimesh = None

# ---------------------------
# Config (你可以在运行前调整这些路径)
# ---------------------------
train_data_dir = "train_data_dir"   # 你的原始 ShapeNet mesh 或数据文件夹（和你原来一致）
val_data_dir = "val_data_dir"
preprocessed_train_dir = "preprocessed_train"  # 预处理后文件输出目录（.pt）
preprocessed_val_dir = "preprocessed_val"
USE_PREPROCESSED = True  # 若 True 则训练脚本从 preprocessed_train/val 加载 .pt
NUM_POINTS = 1024
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_ALPHA = 1e-6

# ---------------------------
# Small helpers
# ---------------------------
def create_mlp(in_channels, out_channels_list):
    """
    稳定版本 create_mlp：输出列表里最后一层是线性(no ReLU)，中间层带 ReLU。
    out_channels_list: list of integers (隐层和输出)
    """
    layers = []
    for i, out_ch in enumerate(out_channels_list):
        layers.append(nn.Linear(in_channels, out_ch))
        if i != len(out_channels_list) - 1:
            layers.append(nn.ReLU(inplace=True))
        in_channels = out_ch
    return nn.Sequential(*layers)


def safe_normalize(t: torch.Tensor, eps=1e-8):
    mn = float(torch.min(t))
    mx = float(torch.max(t))
    rng = mx - mn
    if rng < 1e-8:
        return torch.zeros_like(t)
    return (t - mn) / (rng + eps)


# ---------------------------
# Dataset (加载预处理后的 .pt 或 fallback)
# ---------------------------
class PreprocessedPointCloudDataset(Dataset):
    def __init__(self, folder):
        # expects many .pt files each containing a tensor of shape [num_points,3]
        self.files = sorted(glob.glob(os.path.join(folder, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {folder}")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])  # tensor shape [N,3]
        if data.ndim == 2 and data.shape[1] >= 3:
            pts = data[:, :3].float()
        else:
            raise RuntimeError("preprocessed file format incorrect")
        return pts


# Fallback dataset (slow): 给出原始 mesh 文件夹，用 trimesh + pytorch3d 临时采样
class MeshOnTheFlyDataset(Dataset):
    def __init__(self, folder, num_points=1024):
        self.files = sorted(glob.glob(os.path.join(folder, "*.ply")) + glob.glob(os.path.join(folder, "*.obj")))
        self.num_points = num_points
        if len(self.files) == 0:
            raise RuntimeError(f"No mesh files (.ply/.obj) found in {folder}")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        mesh_path = self.files[idx]
        if trimesh is None:
            raise RuntimeError("trimesh/pytorch3d not available for on-the-fly sampling")
        tm = trimesh.load(mesh_path, force='mesh')
        if tm.is_empty:
            raise RuntimeError(f"mesh empty: {mesh_path}")
        # convert to pytorch3d Meshes
        verts = torch.tensor(tm.vertices, dtype=torch.float32).unsqueeze(0)
        faces = torch.tensor(tm.faces, dtype=torch.int64).unsqueeze(0)
        mesh = Meshes(verts=verts, faces=faces)
        pts = sample_points_from_meshes(mesh, self.num_points)[0]  # (num_points,3)
        return pts


# ---------------------------
# Model (示例简单 encoder -> mlp -> 输出每点 alpha)
# ---------------------------
class SimplePointEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden=[64,128,256], out_dim=128):
        super().__init__()
        layers = []
        last = in_channels
        for h in hidden:
            layers.append(create_mlp(last, [h]))
            last = h
        self.mlp = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = create_mlp(last, [out_dim])  # global feature dim

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        x = x.view(B*N, -1)
        x = self.mlp(x)         # (B*N, feat)
        feat_dim = x.shape[1]
        x = x.view(B, N, feat_dim).permute(0,2,1)  # (B,feat,N)
        g = self.global_pool(x).squeeze(-1)        # (B,feat)
        g = self.fc_out(g)                         # (B,out_dim)
        return g


class AlphaPolicyNet(nn.Module):
    def __init__(self, point_feat_dim=128, hidden=[128, 128], max_pts=1024):
        super().__init__()
        self.point_feat_dim = point_feat_dim
        self.max_pts = max_pts
        # a small decoder that will expand global feature to per-point means
        self.decoder = create_mlp(point_feat_dim, hidden + [max_pts])
        # produce a per-point std (learnable scalar)
        self.log_std = nn.Parameter(torch.log(torch.ones(1) * 0.1))

    def forward(self, global_feat):
        # global_feat: (B, feat)
        loc = self.decoder(global_feat)  # (B, max_pts)
        loc = F.softplus(loc) + MIN_ALPHA  # ensure > 0
        std = torch.exp(self.log_std)
        return loc.unsqueeze(1), std  # loc (B,1,max_pts), std scalar


# ---------------------------
# Reward function (batch化、knn)
# ---------------------------
def calculate_reward_batch(alphas_batch, points_batch, k=8, weights=None, device='cpu'):
    """
    alphas_batch: (B, N)  or (B,1,N)
    points_batch: (B, N, 3)
    returns: rewards: (B,)
    使用 knn_graph（batch 支持）并用 scatter_mean 聚合距离，避免 full cdist.
    """
    if weights is None:
        weights = {'w_correlation': 1.0, 'w_diversity': 0.5, 'w_magnitude': 0.2}

    B, N, _ = points_batch.shape
    device = points_batch.device
    alphas_batch = alphas_batch.view(B, N)

    if knn_graph is None or scatter_mean is None:
        # fallback: approximate using pairwise cdist but do per-batch sequentially (slow)
        rewards = []
        for b in range(B):
            pts = points_batch[b]  # (N,3)
            alphas = alphas_batch[b]
            d = torch.cdist(pts, pts)  # (N,N) - heavy for large N
            k_small = min(k+1, N)
            vals, _ = torch.topk(d, k=k_small, largest=False)  # (N, k+1)
            mean_d = vals.mean(dim=1)
            corr = -F.mse_loss(safe_normalize(alphas), safe_normalize(mean_d))
            diversity = torch.std(alphas)
            magnitude_pen = -torch.log1p(torch.mean(alphas))
            rew = weights['w_correlation']*corr + weights['w_diversity']*diversity + weights['w_magnitude']*magnitude_pen
            rewards.append(float(rew))
        return torch.tensor(rewards, device=device)

    # knn_graph (batch aware)
    # flatten positions and make batch index
    pts_flat = points_batch.view(B*N, 3)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, N).view(-1)  # (B*N,)
    # knn_graph returns edges for the flattened set when batch provided
    edge_index = knn_graph(pts_flat, k=k+1, batch=batch_idx, loop=True)  # [2, E]

    src = edge_index[0]
    dst = edge_index[1]
    diffs = pts_flat[src] - pts_flat[dst]
    dists = torch.sqrt((diffs ** 2).sum(dim=1) + 1e-12)  # (E,)

    # aggregate mean distance per source (flattened)
    mean_dists_flat = scatter_mean(dists, src, dim=0, dim_size=B*N)  # (B*N,)
    mean_dists = mean_dists_flat.view(B, N)

    rewards = []
    for b in range(B):
        alphas = alphas_batch[b]
        mean_d = mean_dists[b]
        corr = -F.mse_loss(safe_normalize(alphas), safe_normalize(mean_d))
        diversity = torch.std(alphas)
        magnitude_pen = -torch.log1p(torch.mean(alphas))
        rew = weights['w_correlation']*corr + weights['w_diversity']*diversity + weights['w_magnitude']*magnitude_pen
        rewards.append(rew)
    rewards = torch.stack(rewards).to(device)
    return rewards


# ---------------------------
# Training loop skeleton
# ---------------------------
def collate_pad(batch):
    # batch is list of (N,3) tensors (all same N after preprocess)
    xs = torch.stack(batch, dim=0)
    return xs

def train():
    # dataset selection
    if USE_PREPROCESSED:
        train_ds = PreprocessedPointCloudDataset(preprocessed_train_dir)
        val_ds = PreprocessedPointCloudDataset(preprocessed_val_dir)
    else:
        train_ds = MeshOnTheFlyDataset(train_data_dir, num_points=NUM_POINTS)
        val_ds = MeshOnTheFlyDataset(val_data_dir, num_points=NUM_POINTS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_pad, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_pad, pin_memory=True)

    encoder = SimplePointEncoder(in_channels=3, hidden=[64,128,256], out_dim=128).to(DEVICE)
    policy_net = AlphaPolicyNet(point_feat_dim=128, hidden=[128,128], max_pts=NUM_POINTS).to(DEVICE)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(policy_net.parameters()), lr=1e-4)
    epochs = 100

    for epoch in range(epochs):
        encoder.train(); policy_net.train()
        running_loss = 0.0
        for i, pts in enumerate(train_loader):
            pts = pts.to(DEVICE)  # (B, N, 3)
            B, N, _ = pts.shape
            # forward
            global_feat = encoder(pts)  # (B, feat)
            loc, std = policy_net(global_feat)  # loc: (B,1,max_pts)
            # ensure shapes match
            loc = loc[:, 0, :N] if loc.shape[-1] >= N else F.pad(loc[:,0,:], (0, N - loc.shape[-1]), 'constant', MIN_ALPHA)
            # use reparameterization for lower variance
            eps = torch.randn_like(loc)
            sampled_alphas = loc + eps * std
            sampled_alphas = F.softplus(sampled_alphas) + MIN_ALPHA  # ensure >0

            # compute reward (batch)
            rewards = calculate_reward_batch(sampled_alphas, pts, k=8, device=DEVICE)
            # loss: negative reward (we maximize reward)
            loss = -torch.mean(rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            if i % 20 == 0:
                print(f"Epoch {epoch} iter {i} loss {running_loss/(i+1):.4f}")

        # optional validation pass
        encoder.eval(); policy_net.eval()
        val_rewards = []
        with torch.no_grad():
            for pts in val_loader:
                pts = pts.to(DEVICE)
                global_feat = encoder(pts)
                loc, std = policy_net(global_feat)
                loc = loc[:, 0, :pts.shape[1]] if loc.shape[-1] >= pts.shape[1] else F.pad(loc[:,0,:], (0, pts.shape[1]-loc.shape[-1]), 'constant', MIN_ALPHA)
                sampled_alphas = loc  # deterministic eval
                rewards = calculate_reward_batch(sampled_alphas, pts, k=8, device=DEVICE)
                val_rewards.append(rewards)
            if len(val_rewards) > 0:
                val_mean = torch.cat(val_rewards).mean().item()
                print(f"Epoch {epoch} validation mean reward: {val_mean:.4f}")

    # save models
    torch.save(encoder.state_dict(), "encoder_fixed.pth")
    torch.save(policy_net.state_dict(), "policy_net_fixed.pth")
    print("Training finished, models saved.")


if __name__ == "__main__":
    # quick sanity checks
    if USE_PREPROCESSED:
        if not os.path.isdir(preprocessed_train_dir) or not os.path.isdir(preprocessed_val_dir):
            raise RuntimeError("USE_PREPROCESSED=True but preprocessed directories not found. Run preprocessing first or set USE_PREPROCESSED=False")
    else:
        if trimesh is None:
            print("Warning: trimesh/pytorch3d not available; set USE_PREPROCESSED=True to avoid on-the-fly sampling failure.")
    if knn_graph is None or scatter_mean is None:
        print("Warning: torch_cluster or torch_scatter not available. Reward will fallback to slower cdist-based computation.")
    train()
