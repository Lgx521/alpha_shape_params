# train_final_v8.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import os
import glob
from torch.utils.tensorboard import SummaryWriter

# ==============================================================================
# --- 版本与实验配置 ---
# ==============================================================================
training_version = 'v8'
'''
版本历史:
v7: 采用基于CGAL的真实重建奖励。
v8: 将重建引擎从CGAL替换为易于安装的alphashape包，以解决安装问题。
    - 牺牲了部分计算速度，换取了开发和部署的便利性。
'''

# --- 1. 核心依赖导入 ---
# (省略了之前已经写过的部分，直接进入新依赖)
try:
    import alphashape
    print("alphashape 库已找到 (V8奖励函数必需)。")
except ImportError:
    print("致命错误: 'alphashape' 未安装。请运行: python -m pip install alphashape")
    exit()

# V8版本依然需要这些库来计算奖励
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch_geometric.nn import knn_interpolate, fps, knn_graph
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_directory = os.path.join(project_root, f'checkpoints_{training_version}')
os.makedirs(save_directory, exist_ok=True)


# --- 2. 模型架构 (保持不变) ---
class PointNetAlphaUNet(torch.nn.Module):
    # ... (此处省略与之前完全相同的模型代码) ...
    def __init__(self, k_interp=3):
        super().__init__()
        self.k = k_interp
        def create_mlp(in_channels, out_channels_list, last_relu=True):
            layers = []
            for out_channels in out_channels_list:
                layers.append(nn.Linear(in_channels, out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if not last_relu: return nn.Sequential(*layers[:-1])
            return nn.Sequential(*layers)
        self.sa1_mlp = create_mlp(3 + 3, [64, 128])
        self.sa2_mlp = create_mlp(128 + 3, [128, 256])
        self.sa3_mlp = create_mlp(256 + 3, [256, 512])
        self.fp2_mlp = create_mlp(512 + 256, [256, 256])
        self.fp1_mlp = create_mlp(256 + 128, [256, 128])
        self.fp0_mlp = create_mlp(128 + 3 + 3, [128, 128])
        self.head_mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 1))
        self.softplus = nn.Softplus()
    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        l0_pos, l0_x, l0_batch = pos, x, batch
        l0_features = self.sa1_mlp(torch.cat([l0_pos, l0_x], dim=1))
        l1_idx = fps(l0_pos, l0_batch, ratio=0.25)
        l1_pos, l1_batch = l0_pos[l1_idx], l0_batch[l1_idx]
        l1_agg_features = knn_interpolate(l0_features, l0_pos, l1_pos, l0_batch, l1_batch, k=16)
        l1_features = self.sa2_mlp(torch.cat([l1_agg_features, l1_pos], dim=1))
        l2_idx = fps(l1_pos, l1_batch, ratio=0.25)
        l2_pos, l2_batch = l1_pos[l2_idx], l1_batch[l2_idx]
        l2_agg_features = knn_interpolate(l1_features, l1_pos, l2_pos, l1_batch, l2_batch, k=16)
        l2_features = self.sa3_mlp(torch.cat([l2_agg_features, l2_pos], dim=1))
        l1_up_features = knn_interpolate(l2_features, l2_pos, l1_pos, l2_batch, l1_batch, k=self.k)
        l1_fp_features = self.fp2_mlp(torch.cat([l1_up_features, l1_features], dim=1))
        l0_up_features = knn_interpolate(l1_fp_features, l1_pos, l0_pos, l1_batch, l0_batch, k=self.k)
        l0_fp_features = self.fp1_mlp(torch.cat([l0_up_features, l0_features], dim=1))
        final_features = self.fp0_mlp(torch.cat([l0_fp_features, l0_pos, l0_x], dim=1))
        alpha_mean = self.head_mlp(final_features)
        alpha_mean_dense, _ = to_dense_batch(alpha_mean, batch)
        alpha_mean_dense = alpha_mean_dense.permute(0, 2, 1)
        MIN_ALPHA = 0.01
        alpha_mean_activated = self.softplus(alpha_mean_dense) + MIN_ALPHA
        policy = Normal(alpha_mean_activated, torch.ones_like(alpha_mean_activated))
        return policy

# --- 3. 数据加载器 (保持不变) ---
class PyGShapeNetDataset(Dataset):
    # ... (此处省略与V7完全相同的数据加载器代码) ...
    def __init__(self, root_dir, num_points=2048, split='train'):
        super().__init__(root_dir)
        self.processed_data_folder = os.path.join(self.root, "processed_points_with_normals")
        self.num_points = num_points
        self.paths = glob.glob(os.path.join(self.processed_data_folder, "**/*.pt"), recursive=True)
        if not self.paths: raise ValueError(f"在 '{self.processed_data_folder}' 中未找到预处理的 '.pt' 文件。")
        print(f"为 '{split}' 分割找到了 {len(self.paths)} 个预处理好的模型。")
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            data_dict = torch.load(self.paths[idx], weights_only=False)
            points, normals = data_dict['pos'], data_dict['x']
            center = points.mean(dim=0)
            points_centered = points - center
            scale = (points_centered.norm(p=2, dim=1)).max()
            points_normalized = points_centered / scale
            return Data(pos=points_normalized, x=normals)
        except Exception as e:
            print(f"警告：加载或处理文件 {self.paths[idx]} 失败: {e}")
            return self.__getitem__((idx + 1) % len(self))

# --- 4. 全新奖励函数 (V8 - 基于alphashape) ---
def calculate_reconstruction_reward_v8_alphashape(alphas, points, weights, device):
    try:
        points_np = points.cpu().numpy()
        median_alpha = torch.median(alphas).item()
        if median_alpha <= 1e-9: median_alpha = 1e-9
        mesh_alphashape = alphashape.alphashape(points_np, median_alpha)
        if not hasattr(mesh_alphashape, 'faces') or len(mesh_alphashape.faces) == 0:
            return torch.tensor(-10.0, device=device)
        verts_tensor = torch.tensor(mesh_alphashape.vertices, dtype=torch.float32, device=device)
        faces_tensor = torch.tensor(mesh_alphashape.faces, dtype=torch.long, device=device)
        reconstructed_mesh = Meshes(verts=[verts_tensor], faces=[faces_tensor])
    except Exception:
        return torch.tensor(-10.0, device=device)

    with torch.no_grad():
        sampled_points_from_recon = sample_points_from_meshes(reconstructed_mesh, num_samples=points.shape[0])
        chamfer_loss, _ = chamfer_distance(sampled_points_from_recon, points.unsqueeze(0))
        reward_fidelity = -chamfer_loss
        smoothness_penalty = mesh_laplacian_smoothing(reconstructed_mesh)
        reward_smoothness = -smoothness_penalty
    total_reward = (weights['w_fidelity'] * reward_fidelity + weights['w_smoothness'] * reward_smoothness)
    return total_reward

# --- 5. 训练主函数 (适配V8) ---
def main():
    # --- 超参数配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
    NUM_POINTS = 2048
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-5
    EPOCHS = 100
    REWARD_BASELINE_DECAY = 0.95
    EXPLORATION_DECAY = 0.98
    REWARD_WEIGHTS = {'w_fidelity': 100.0, 'w_smoothness': 1.0}

    writer = SummaryWriter(f'runs/pointnet_alpha_{training_version}_experiment')
    model = PointNetAlphaUNet().to(DEVICE)
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH, num_points=NUM_POINTS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * EPOCHS, eta_min=1e-6)

    START_EPOCH, reward_baseline, global_step = 0, 0.0, 0
    print(f"在 {DEVICE} 上开始训练 (版本: {training_version})")
    print(f"Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, 奖励权重: {REWARD_WEIGHTS}")
    
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        current_std = max(0.20 * (EXPLORATION_DECAY**epoch), 0.01)

        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)
            points_dense, mask = to_dense_batch(batch_data.pos, batch_data.batch)
            policy = model(batch_data)
            policy.scale = torch.ones_like(policy.loc) * current_std 
            sampled_alphas_dense = policy.sample()

            batch_rewards = []
            for i in range(points_dense.shape[0]):
                sample_points = points_dense[i, mask[i]]
                sample_alphas = sampled_alphas_dense[i, :, mask[i]].squeeze()
                # 调用V8奖励函数
                reward = calculate_reconstruction_reward_v8_alphashape(sample_alphas, sample_points, REWARD_WEIGHTS, DEVICE)
                batch_rewards.append(reward)
            rewards_tensor = torch.stack(batch_rewards)

            rewards_tensor.clamp_(-20, 0)
            avg_reward = rewards_tensor.mean().item()
            advantage = rewards_tensor - reward_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            log_probs_dense = policy.log_prob(sampled_alphas_dense)
            log_probs_sum_per_sample = (log_probs_dense * mask.unsqueeze(1)).sum(dim=[1, 2])
            loss = - (log_probs_sum_per_sample * advantage).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            reward_baseline = REWARD_BASELINE_DECAY * reward_baseline + (1 - REWARD_BASELINE_DECAY) * avg_reward
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_reward=f"{avg_reward:.4f}", baseline=f"{reward_baseline:.4f}")

            if global_step % 10 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Reward/average_reward', avg_reward, global_step)
                writer.add_scalar('Reward/baseline', reward_baseline, global_step)
                writer.add_scalar('Hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            save_file_name = f"pointnet_alpha_{training_version}_epoch_{epoch+1}.pth"
            save_path = os.path.join(save_directory, save_file_name)
            torch.save(model.state_dict(), save_path)
            print(f"\n✅ 模型已保存至 {save_path}")

    writer.close()
    print(f"训练完成 (版本: {training_version})。")

if __name__ == '__main__':
    main()