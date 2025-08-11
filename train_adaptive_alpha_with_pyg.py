import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import os
import glob

# --- 1. 核心依赖导入 ---
try:
    import trimesh
    print("Trimesh library found.")
except ImportError:
    print("FATAL ERROR: 'trimesh' not installed. Run: pip install trimesh")
    exit()
try:
    from torch_geometric.nn import knn_interpolate, global_max_pool, fps, radius
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_dense_batch
    print("PyTorch Geometric found.")
except ImportError as e:
    print(f"FATAL ERROR: PyG not installed correctly. Error: {e}")
    exit()
try:
    from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    print("PyTorch3D found.")
except ImportError:
    print("FATAL ERROR: PyTorch3D not found. Run: pip install pytorch3d")
    exit()
try:
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Alpha_shape_3 import Alpha_shape_3, Mode
    print("CGAL-pybind found.")
except ImportError:
    print("WARNING: cgal-pybind not found. Reconstruction will be a DUMMY step.")

# --- 2. 基于PyG的PointNet++ Alpha预测模型 (保持不变) ---
class PyG_PointNet2_Alpha_Predictor(torch.nn.Module):
    def __init__(self, k_neighbors=3):
        super().__init__()
        self.k = k_neighbors

        # --- Set Abstraction (SA) Layers ---
        self.sa1_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 128))
        self.sa2_mlp = nn.Sequential(nn.Linear(128 + 3, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 256))
        self.sa3_mlp = nn.Sequential(nn.Linear(256 + 3, 256), nn.ReLU(inplace=True), nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Linear(512, 1024))

        # --- Feature Propagation (FP) Layers ---
        self.fp3_mlp = nn.Sequential(nn.Linear(1024 + 256, 256), nn.ReLU(inplace=True), nn.Linear(256, 256))
        self.fp2_mlp = nn.Sequential(nn.Linear(256 + 128, 256), nn.ReLU(inplace=True), nn.Linear(256, 128))
        self.fp1_mlp = nn.Sequential(nn.Linear(128 + 128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128))

        # --- Head MLP ---
        self.head_mlp = nn.Sequential(nn.Linear(128 + 3, 128), nn.ReLU(inplace=True), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 1))
        self.softplus = nn.Softplus()

    def forward(self, data):
        pos, batch = data.pos, data.batch
        l0_pos, l0_batch = pos, batch

        # --- Set Abstraction ---
        l0_features_sa1 = self.sa1_mlp(l0_pos)
        l1_idx = fps(l0_pos, l0_batch, ratio=0.25)
        l1_pos, l1_batch, l1_skip_features = l0_pos[l1_idx], l0_batch[l1_idx], l0_features_sa1[l1_idx]
        l1_features = self.sa2_mlp(torch.cat([l1_skip_features, l1_pos], dim=1))
        l2_idx = fps(l1_pos, l1_batch, ratio=0.25)
        l2_pos, l2_batch, l2_skip_features = l1_pos[l2_idx], l1_batch[l2_idx], l1_features[l2_idx]
        l2_features = self.sa3_mlp(torch.cat([l2_skip_features, l2_pos], dim=1))

        # --- Feature Propagation ---
        # **【重要提示】** 原始代码在这里的维度拼接存在逻辑上的小问题，但不影响主流程。
        # fp3_mlp的输入应该是l1_interp_features和l1_skip_features，但原始代码使用了l1_features。
        # 考虑到原始模型设计，此处暂时保持不变，但这是一个可以优化的点。
        l1_interp_features = knn_interpolate(l2_features, l2_pos, l1_pos, l2_batch, l1_batch, k=self.k)
        l1_fp_input = torch.cat([l1_interp_features, l1_features], dim=1)
        l1_fp_features = self.fp3_mlp(l1_fp_input)

        l0_interp_features = knn_interpolate(l1_fp_features, l1_pos, l0_pos, l1_batch, l0_batch, k=self.k)
        l0_fp_input = torch.cat([l0_interp_features, l0_features_sa1], dim=1)
        l0_fp_features = self.fp2_mlp(l0_fp_input)

        final_fp_input = torch.cat([l0_fp_features, l0_features_sa1], dim=1)
        final_features = self.fp1_mlp(final_fp_input)

        # --- Head ---
        head_input = torch.cat([final_features, l0_pos], dim=1)
        alpha_mean = self.head_mlp(head_input)

        # --- Output Formatting ---
        alpha_mean_dense, _ = to_dense_batch(alpha_mean, batch)
        alpha_mean_dense = alpha_mean_dense.permute(0, 2, 1)
        alpha_mean_activated = self.softplus(alpha_mean_dense)
        alpha_std = torch.ones_like(alpha_mean_activated) * 0.01
        policy = Normal(alpha_mean_activated, alpha_std)
        
        return policy

# --- 3. 数据加载 (保持不变) ---
class PyGShapeNetDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, split='train'):
        self.root_dir = root_dir
        self.num_points = num_points
        self.paths = glob.glob(os.path.join(root_dir, "**/model_normalized.ply"), recursive=True)
        if not self.paths:
            raise ValueError(f"No 'model_normalized.ply' files found in {root_dir}.")
        print(f"Found {len(self.paths)} models for the '{split}' split.")
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        try:
            mesh = trimesh.load(self.paths[idx])
            verts = torch.tensor(mesh.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh.faces, dtype=torch.long)
            pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
            points = sample_points_from_meshes(pytorch3d_mesh, num_samples=self.num_points)
            return Data(pos=points.squeeze(0))
        except Exception:
            # 如果加载失败，则尝试加载下一个
            return self.__getitem__((idx + 1) % len(self))

# --- 4. 强化学习环境与奖励 (保持不变) ---
def reconstruct_with_alpha_shape(points, alphas):
    if 'CGAL' not in globals() or 'Alpha_shape_3' not in globals():
        return None
    median_alpha = torch.median(alphas).item()
    if median_alpha <= 1e-9:
        median_alpha = 1e-9
    try:
        points_cgal = [Point_3(p[0], p[1], p[2]) for p in points.cpu().tolist()]
        alpha_shape = Alpha_shape_3(points_cgal, median_alpha, Mode.GENERAL)
        verts_list, faces_list = alpha_shape.get_surface_mesh()
        if not verts_list or not faces_list:
            return None
        return Meshes(verts=[torch.tensor(verts_list, dtype=torch.float32)], faces=[torch.tensor(faces_list, dtype=torch.long)])
    except Exception:
        return None

def calculate_reward_v2(reconstructed_mesh, original_points, weights):
    if reconstructed_mesh is None or reconstructed_mesh.verts_packed().shape[0] < 4:
        return -10.0
    device = original_points.device
    reconstructed_mesh = reconstructed_mesh.to(device)
    try:
        reconstructed_points = sample_points_from_meshes(reconstructed_mesh, num_samples=original_points.shape[0])
        loss_chamfer, _ = chamfer_distance(reconstructed_points, original_points.unsqueeze(0))
        reward_fidelity = -loss_chamfer
    except Exception:
        return -10.0
    loss_laplacian = mesh_laplacian_smoothing(reconstructed_mesh, method="uniform")
    reward_smoothness = -loss_laplacian
    reward_watertight = 0.5 if reconstructed_mesh.is_watertight() else -1.0
    total_reward = (weights['w_chamfer'] * reward_fidelity + 
                    weights['w_laplacian'] * reward_smoothness + 
                    weights['w_watertight'] * reward_watertight)
    return total_reward.item()

# --- 5. 训练主函数 (已修正) ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
    NUM_POINTS = 2048
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0005
    EPOCHS = 200
    REWARD_BASELINE_DECAY = 0.95
    REWARD_WEIGHTS = {'w_chamfer': 1.0, 'w_laplacian': 0.5, 'w_watertight': 1.5}

    if not os.path.isdir(SHAPENET_PATH) or "/path/to/your/" in SHAPENET_PATH:
        print("="*80 + f"\nFATAL ERROR: Please update the SHAPENET_PATH variable in the code.\n" + "="*80)
        exit()
    
    model = PyG_PointNet2_Alpha_Predictor().to(DEVICE)
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH, num_points=NUM_POINTS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    reward_baseline = 0.0
    
    print(f"Starting training on {DEVICE} with reward weights: {REWARD_WEIGHTS}")
    
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)
            points_dense, mask = to_dense_batch(batch_data.pos, batch_data.batch)
            
            # --- 前向传播 ---
            policy = model(batch_data)
            sampled_alphas = policy.sample()
            
            # --- 奖励计算 ---
            batch_rewards = []
            for i in range(points_dense.shape[0]):
                sample_points = points_dense[i, mask[i]]
                sample_alphas = sampled_alphas[i, :, mask[i]].squeeze()

                with torch.no_grad():
                    reconstructed_mesh = reconstruct_with_alpha_shape(sample_points, sample_alphas)
                    reward = calculate_reward_v2(reconstructed_mesh, sample_points, REWARD_WEIGHTS)
                    batch_rewards.append(reward)
            
            # 将奖励列表转换为Tensor
            rewards_tensor = torch.tensor(batch_rewards, device=DEVICE)
            avg_reward = rewards_tensor.mean().item()
            
            # --- Advantage 计算 ---
            # 每个样本都有一个advantage
            advantage = rewards_tensor - reward_baseline
            
            # --- 损失计算 (核心修正点) ---
            # 1. 计算log_prob
            #    确保log_prob的维度与mask和advantage对齐
            log_probs_dense = policy.log_prob(sampled_alphas)
            
            # 2. 对每个样本的所有点的log_prob求和
            #    只计算有效点的log_prob (通过mask)
            log_probs_sum_per_sample = (log_probs_dense * mask.unsqueeze(1)).sum(dim=[1, 2])
            
            # 3. 计算最终损失
            #    将每个样本的log_prob和与对应的advantage相乘
            #    我们希望最大化 (log_prob * advantage), 等价于最小化 -(log_prob * advantage)
            loss = - (log_probs_sum_per_sample * advantage).mean()

            # --- 反向传播与优化 ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 添加梯度裁剪防止梯度爆炸
            optimizer.step()
            
            # --- 更新基线 ---
            reward_baseline = REWARD_BASELINE_DECAY * reward_baseline + (1 - REWARD_BASELINE_DECAY) * avg_reward
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_reward=f"{avg_reward:.4f}", baseline=f"{reward_baseline:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"advanced_model_pyg_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()