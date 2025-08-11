# ==============================================================================
#      Advanced Adaptive Alpha Shape Training (V3.1)
#
#  这个脚本整合了以下内容:
#  1. PyTorch Geometric (PyG) 作为现代、高效的PointNet++后端.
#  2. 一个自监督的强化学习训练循环 (REINFORCE 策略梯度).
#  3. 一个多目标的、更鲁棒的奖励函数.
#  4. 使用 `trimesh` 库来正确加载您数据集中的 .ply 文件.
#
#  作者: shengzhegan
#  日期: August 11, 2025
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import os
import glob

# --- 1. 核心依赖导入 ---
try: import trimesh; print("Trimesh library found.")
except ImportError: print("FATAL ERROR: 'trimesh' not installed. Run: pip install trimesh"); exit()
try:
    from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool
    from torch_geometric.data import Data, Dataset; from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_dense_batch; print("PyTorch Geometric found.")
except ImportError as e: print(f"FATAL ERROR: PyG not installed correctly. Error: {e}"); exit()
try:
    from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
    from pytorch3d.ops import sample_points_from_meshes; from pytorch3d.structures import Meshes
    print("PyTorch3D found.")
except ImportError: print("FATAL ERROR: PyTorch3D not found. Run: pip install pytorch3d"); exit()
try:
    from CGAL.CGAL_Kernel import Point_3; from CGAL.CGAL_Alpha_shape_3 import Alpha_shape_3, Mode
    print("CGAL-pybind found.")
except ImportError: print("WARNING: cgal-pybind not found. Reconstruction will be a DUMMY step.")


# --- 2. 基于PyG的PointNet++ Alpha预测模型 (恢复可读性) ---
class PyG_PointNet2_Alpha_Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个清晰的辅助函数来创建MLP层
        def create_mlp(in_channels, out_channels_list, add_relu=True):
            layers = []
            for i, out_channels in enumerate(out_channels_list):
                layers.append(nn.Linear(in_channels, out_channels))
                if add_relu and i < len(out_channels_list) - 1:
                    layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            return nn.Sequential(*layers)

        # --- 编码器 (下采样) ---
        self.sa1_mlp = create_mlp(3, [64, 64, 128])
        self.sa2_mlp = create_mlp(128 + 3, [128, 128, 256])
        self.sa3_mlp = create_mlp(256 + 3, [256, 512, 1024])
        self.global_mlp = create_mlp(1024, [1024])

        # --- 解码器 (上采样/特征传播) ---
        self.fp3_mlp = create_mlp(1024 + 1024, [512, 256]) # Global + SA3 -> FP3
        self.fp2_mlp = create_mlp(256 + 256, [256, 128])   # FP3 + SA2 -> FP2
        self.fp1_mlp = create_mlp(128 + 128, [128, 128])   # FP2 + SA1 -> FP1

        # --- 输出头 ---
        self.head_mlp = nn.Sequential(
            create_mlp(128 + 3, [128, 64]), # FP1 + original points
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        self.softplus = nn.Softplus()

    def forward(self, data):
        pos, batch = data.pos, data.batch

        # --- 编码器 ---
        l1_pos, l1_batch = pos, batch
        l1_x = self.sa1_mlp(l1_pos)

        l2_idx = fps(l1_pos, l1_batch, ratio=0.25)
        l2_pos, l2_batch, l2_x = l1_pos[l2_idx], l1_batch[l2_idx], l1_x[l2_idx]
        l2_x = self.sa2_mlp(torch.cat([l2_x, l2_pos], dim=1))

        l3_idx = fps(l2_pos, l2_batch, ratio=0.25)
        l3_pos, l3_batch, l3_x = l2_pos[l3_idx], l2_batch[l3_idx], l2_x[l3_idx]
        l3_x = self.sa3_mlp(torch.cat([l3_x, l3_pos], dim=1))
        
        l4_x = self.global_mlp(global_max_pool(l3_x, l3_batch))

        # --- 解码器 ---
        l3_x = self.fp3_mlp(torch.cat([l4_x[l3_batch], l3_x], dim=1))
        l2_x = self.fp2_mlp(torch.cat([l3_x[fps(l2_pos,l2_batch,ratio=1)], l2_x], dim=1)) # Simplified upsampling
        l1_x = self.fp1_mlp(torch.cat([l2_x[fps(l1_pos,l1_batch,ratio=1)], l1_x], dim=1)) # Simplified upsampling

        # --- 输出头 ---
        x = self.head_mlp(torch.cat([l1_x, l1_pos], dim=1))
        alpha_mean, _ = to_dense_batch(x, batch)
        alpha_mean = alpha_mean.permute(0, 2, 1)

        alpha_mean_activated = self.softplus(alpha_mean)
        alpha_std = torch.ones_like(alpha_mean_activated) * 0.01
        policy = Normal(alpha_mean_activated, alpha_std)
        return policy


# --- 3. 数据加载 (使用Trimesh) ---
class PyGShapeNetDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, split='train'):
        self.root_dir = root_dir; self.num_points = num_points
        self.paths = glob.glob(os.path.join(root_dir, "**/model_normalized.ply"), recursive=True)
        if not self.paths: raise ValueError(f"No 'model_normalized.ply' files found in {root_dir}.")
        print(f"Found {len(self.paths)} models for the '{split}' split.")
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            mesh = trimesh.load(self.paths[idx]); verts = torch.tensor(mesh.vertices, dtype=torch.float32); faces = torch.tensor(mesh.faces, dtype=torch.long)
            pytorch3d_mesh = Meshes(verts=[verts], faces=[faces]); points = sample_points_from_meshes(pytorch3d_mesh, num_samples=self.num_points)
            return Data(pos=points.squeeze(0))
        except Exception: return self.__getitem__((idx + 1) % len(self))


# --- 4. 强化学习环境与奖励 (V2版奖励函数) ---
def reconstruct_with_alpha_shape(points, alphas):
    # ... (此部分保持不变)
    if 'CGAL' not in globals() or 'Alpha_shape_3' not in globals(): return None
    median_alpha = torch.median(alphas).item();
    if median_alpha <= 1e-9: median_alpha = 1e-9
    try:
        points_cgal = [Point_3(p[0], p[1], p[2]) for p in points.cpu().tolist()]; alpha_shape = Alpha_shape_3(points_cgal, median_alpha, Mode.GENERAL); verts_list, faces_list = alpha_shape.get_surface_mesh()
        if not verts_list or not faces_list: return None
        return Meshes(verts=[torch.tensor(verts_list, dtype=torch.float32)], faces=[torch.tensor(faces_list, dtype=torch.long)])
    except Exception: return None

def calculate_reward_v2(reconstructed_mesh, original_points, weights):
    # ... (此部分保持不变)
    if reconstructed_mesh is None or reconstructed_mesh.verts_packed().shape[0] < 4: return -10.0
    device = original_points.device; reconstructed_mesh = reconstructed_mesh.to(device)
    try:
        reconstructed_points = sample_points_from_meshes(reconstructed_mesh, num_samples=original_points.shape[0]); loss_chamfer, _ = chamfer_distance(reconstructed_points, original_points.unsqueeze(0)); reward_fidelity = -loss_chamfer
    except Exception: return -10.0
    loss_laplacian = mesh_laplacian_smoothing(reconstructed_mesh, method="uniform"); reward_smoothness = -loss_laplacian
    reward_watertight = 0.5 if reconstructed_mesh.is_watertight() else -1.0
    total_reward = (weights['w_chamfer'] * reward_fidelity + weights['w_laplacian'] * reward_smoothness + weights['w_watertight'] * reward_watertight)
    return total_reward.item()


# --- 5. 训练主函数 (保持不变) ---
def main():
    # ... (此部分保持不变)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/path/to/your/ShapeNetCore.v2/ShapeNetCore.v2"
    NUM_POINTS = 2048; BATCH_SIZE = 16; LEARNING_RATE = 0.0005; EPOCHS = 200; REWARD_BASELINE_DECAY = 0.95
    REWARD_WEIGHTS = {'w_chamfer': 1.0, 'w_laplacian': 0.5, 'w_watertight': 1.5}
    if not os.path.isdir(SHAPENET_PATH) or "/path/to/your/" in SHAPENET_PATH: print("="*80 + f"\nFATAL ERROR: Please update the SHAPENET_PATH variable in the code.\n" + "="*80); exit()
    model = PyG_PointNet2_Alpha_Predictor().to(DEVICE); dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH, num_points=NUM_POINTS); dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True); optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE); reward_baseline = 0.0
    print(f"Starting training on {DEVICE} with reward weights: {REWARD_WEIGHTS}")
    for epoch in range(EPOCHS):
        model.train(); progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE); points_dense, mask = to_dense_batch(batch_data.pos, batch_data.batch); policy = model(batch_data); sampled_alphas = policy.sample(); log_probs = policy.log_prob(sampled_alphas).sum(dim=-1).mean(); batch_rewards = []
            for i in range(points_dense.shape[0]):
                sample_points = points_dense[i, mask[i]]; sample_alphas = sampled_alphas[i, :, mask[i]].squeeze()
                with torch.no_grad(): reconstructed_mesh = reconstruct_with_alpha_shape(sample_points, sample_alphas); reward = calculate_reward_v2(reconstructed_mesh, sample_points, REWARD_WEIGHTS); batch_rewards.append(reward)
            avg_reward = sum(batch_rewards) / len(batch_rewards); advantage = avg_reward - reward_baseline; loss = -log_probs * advantage
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            reward_baseline = REWARD_BASELINE_DECAY * reward_baseline + (1 - REWARD_BASELINE_DECAY) * avg_reward
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_reward=f"{avg_reward:.4f}", baseline=f"{reward_baseline:.4f}")
        if (epoch + 1) % 5 == 0: torch.save(model.state_dict(), f"advanced_model_pyg_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()