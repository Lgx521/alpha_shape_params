import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import os
import glob
from torch.utils.tensorboard import SummaryWriter



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_directory = os.path.join(project_root, 'checkpoints')


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
except ImportError as e:
    print("FATAL ERROR: PyTorch3D not found. Run: pip install pytorch3d")
    print(e)
    exit()
try:
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Alpha_shape_3 import Alpha_shape_3, Mode
    print("CGAL-pybind found.")
except ImportError:
    print("WARNING: cgal-pybind not found. Reconstruction will be a DUMMY step.")


# --- 2. 基于PyG的PointNet++ Alpha预测模型 (V5 - 最终修正版) ---
# 这个版本拥有一个逻辑正确、维度匹配、层次分明的U-Net架构。
class PyG_PointNet2_Alpha_Predictor(torch.nn.Module):
    def __init__(self, k_neighbors=3):
        super().__init__()
        self.k = k_neighbors

        # 定义一个清晰的辅助函数来创建MLP层
        def create_mlp(in_channels, out_channels_list):
            layers = []
            for out_channels in out_channels_list:
                layers.append(nn.Linear(in_channels, out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            # 移除最后一个ReLU以获得原始的logits
            return nn.Sequential(*layers[:-1])

        # --- 编码器 (SA) Layers ---
        # 每一层MLP处理的是 [(上一层特征), (当前层坐标)]
        self.sa1_mlp = create_mlp(3, [64, 64, 128])
        self.sa2_mlp = create_mlp(128 + 3, [128, 128, 256])
        self.sa3_mlp = create_mlp(256 + 3, [256, 512, 1024])

        # --- 解码器 (FP) Layers ---
        # 每一层MLP处理的是 [(插值后的上层特征), (本层跳跃连接的特征)]
        # [V5 修正] 维度与forward函数中的标准U-Net逻辑完全匹配
        self.fp3_mlp = create_mlp(1024 + 256, [256, 256])  # l3_up(1024) + l2_skip(256)
        self.fp2_mlp = create_mlp(256 + 128, [256, 128])   # l2_fp(256) + l1_skip(128)
        self.fp1_mlp = create_mlp(128 + 3, [128, 128, 128])# l1_fp(128) + l0_coords(3)

        # --- 输出头 ---
        self.head_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        self.softplus = nn.Softplus()

    def forward(self, data):
        pos, batch = data.pos, data.batch

        # --- 编码器 (Encoder) ---
        # 保存每一层的特征和坐标，用于后续的跳跃连接
        
        # Level 0 (原始输入)
        l0_pos, l0_batch = pos, batch
        
        # Level 1
        l1_idx = fps(l0_pos, l0_batch, ratio=0.25)
        l1_pos, l1_batch = l0_pos[l1_idx], l0_batch[l1_idx]
        # l1_skip_features 来自于对原始坐标的处理
        l1_skip_features = F.relu(self.sa1_mlp(l1_pos))

        # Level 2
        l2_idx = fps(l1_pos, l1_batch, ratio=0.25)
        l2_pos, l2_batch = l1_pos[l2_idx], l1_batch[l2_idx]
        # l2_skip_features 来自于对l1特征的处理
        l2_skip_features = F.relu(self.sa2_mlp(torch.cat([l1_skip_features[l2_idx], l2_pos], dim=1)))
        
        # Level 3 (最深层)
        l3_idx = fps(l2_pos, l2_batch, ratio=0.25)
        l3_pos, l3_batch = l2_pos[l3_idx], l2_batch[l3_idx]
        l3_features = F.relu(self.sa3_mlp(torch.cat([l2_skip_features[l3_idx], l3_pos], dim=1)))
        
        # --- 解码器 (Decoder) ---
        
        # FP for Level 2
        l2_interp_features = knn_interpolate(l3_features, l3_pos, l2_pos, l3_batch, l2_batch, k=self.k)
        l2_fp_features = F.relu(self.fp3_mlp(torch.cat([l2_interp_features, l2_skip_features], dim=1)))
        
        # FP for Level 1
        l1_interp_features = knn_interpolate(l2_fp_features, l2_pos, l1_pos, l2_batch, l1_batch, k=self.k)
        l1_fp_features = F.relu(self.fp2_mlp(torch.cat([l1_interp_features, l1_skip_features], dim=1)))
        
        # FP for Level 0 (原始点)
        l0_interp_features = knn_interpolate(l1_fp_features, l1_pos, l0_pos, l1_batch, l0_batch, k=self.k)
        # 最后一层与原始坐标拼接
        l0_fp_features = F.relu(self.fp1_mlp(torch.cat([l0_interp_features, l0_pos], dim=1)))

        # --- 输出头 ---
        alpha_mean = self.head_mlp(l0_fp_features)
        
        # --- 输出格式化 ---
        alpha_mean_dense, _ = to_dense_batch(alpha_mean, batch)
        alpha_mean_dense = alpha_mean_dense.permute(0, 2, 1)

        MIN_ALPHA = 0.01 

        alpha_mean_activated = self.softplus(alpha_mean_dense) + MIN_ALPHA
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


def calculate_reward_v3(reconstructed_mesh, original_points, alphas, weights, device):
    """
    V3版奖励函数，结合了网格质量和alpha值本身的属性。
    """
    # --- Part 1: Alpha值本身的启发式奖励 (即使网格重建失败也能计算) ---

    # 1a. 局部Alpha一致性奖励: 鼓励局部区域的alpha值平滑
    # 我们使用KNN找到每个点的邻居，并计算alpha值的局部方差
    with torch.no_grad():
        # 使用PyG的radius函数找到每个点半径范围内的邻居
        # 这个半径需要根据你的点云尺度进行调整
        radius_graph = radius(original_points, original_points, r=0.1, max_num_neighbors=16)
        row, col = radius_graph
        # 计算每个点与其邻居alpha值的差的平方的均值，作为局部方差的代理
        local_variance = (alphas[row] - alphas[col])**2
        # 我们希望方差小，所以奖励是负方差
        reward_alpha_consistency = -torch.mean(local_variance)

    # 1b. Alpha值幅度惩罚: 惩罚极端值
    # 使用log惩罚来温和地惩罚过大的alpha值
    penalty_alpha_magnitude = -torch.log(1 + torch.mean(alphas))

    # 1c. 多样性奖励: 奖励alpha值的标准差，防止模型输出常数
    reward_alpha_diversity = torch.std(alphas)

    # --- Part 2: 基于重建网格质量的奖励 (核心目标) ---
    
    reward_fidelity = -10.0  # Chamfer Loss, 保真度
    reward_smoothness = -2.0 # Laplacian Loss, 平滑度
    reward_watertight = -1.0 # 水密性
    
    if reconstructed_mesh is not None and reconstructed_mesh.verts_packed().shape[0] >= 4:
        reconstructed_mesh = reconstructed_mesh.to(device)
        try:
            # 保真度奖励 (Chamfer距离)
            reconstructed_points = sample_points_from_meshes(reconstructed_mesh, num_samples=original_points.shape[0])
            loss_chamfer, _ = chamfer_distance(reconstructed_points, original_points.unsqueeze(0))
            # 使用负的Chamfer距离作为奖励，并进行缩放以控制其影响范围
            # reward_fidelity = -torch.clamp(loss_chamfer, 0, 10) 
            reward_fidelity = -torch.log(1.0 + torch.clamp(loss_chamfer, 0, 100))

            
            # 平滑度奖励
            loss_laplacian = mesh_laplacian_smoothing(reconstructed_mesh, method="uniform")
            reward_smoothness = -torch.clamp(loss_laplacian, 0, 2)

            # 水密性奖励
            if reconstructed_mesh.is_watertight():
                reward_watertight = 1.0 # 成功构建水密网格应获得显著奖励
        except Exception:
            # 如果在计算过程中出错，则使用默认的惩罚值
            pass
            
    # --- Part 3: 组合总奖励 ---
    total_reward = (weights['w_fidelity'] * reward_fidelity +
                    weights['w_smoothness'] * reward_smoothness +
                    weights['w_watertight'] * reward_watertight +
                    weights['w_alpha_consistency'] * reward_alpha_consistency +
                    weights['w_alpha_magnitude'] * penalty_alpha_magnitude +
                    weights['w_alpha_diversity'] * reward_alpha_diversity)
    
    return total_reward.item()

# --- 5. 全新奖励函数 (V5 - 可靠且高效) ---
def calculate_reward_v5(alphas, original_points, k, weights, device):
    """
    V5版奖励函数：不再依赖于重建，而是直接奖励alpha值与局部几何特征的相关性。
    此函数绝对可靠，总能提供平滑的梯度。

    Args:
        alphas (torch.Tensor): 模型生成的alpha值 (N,)
        original_points (torch.Tensor): 原始点云 (N, 3)
        k (int): 用于计算局部密度的邻居数量
        weights (dict): 奖励各部分的权重
        device (torch.Tensor): 计算设备

    Returns:
        float: 计算出的总奖励值
    """
    with torch.no_grad():
        # --- Part 1: 计算每个点的局部几何特征 (邻居平均距离) ---
        # 这是一个完美的“几何复杂度”代理：
        # - 稀疏区域 -> 邻居距离远
        # - 密集区域 -> 邻居距离近

        # 计算所有点对之间的距离矩阵 (N, N)
        dist_matrix = torch.cdist(original_points.unsqueeze(0), original_points.unsqueeze(0)).squeeze(0)

        # 找到每个点最近的k个邻居的距离（topk(k+1)因为包括了自身，距离为0）
        # 我们使用 largest=False 来获取最小的距离
        knn_dists = torch.topk(dist_matrix, k + 1, dim=1, largest=False).values

        # 计算到k个邻居的平均距离（忽略自身，所以从第1个索引开始）
        # 添加一个小的epsilon防止除以0（虽然不太可能）
        local_geom_feature = torch.mean(knn_dists[:, 1:], dim=1)

        # --- Part 2: 标准化，让alpha和几何特征具有可比性 ---
        # 将两个张量都进行min-max标准化到[0, 1]区间，消除尺度差异
        def normalize(tensor):
            return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

        norm_alphas = normalize(alphas)
        norm_geom_feature = normalize(local_geom_feature)

    # --- Part 3: 计算核心奖励 ---

    # 3a. 相关性奖励 (核心！):
    # 我们希望 norm_alphas 和 norm_geom_feature 的分布尽可能一致。
    # 使用负的均方误差(MSE)来奖励它们之间的相似性。MSE越小，奖励越高。
    # 这是最直接、最强大的学习信号。
    reward_correlation = -F.mse_loss(norm_alphas, norm_geom_feature)
    
    # 或者，可以使用余弦相似度，它更关注方向上的一致性（推荐）
    # reward_correlation = F.cosine_similarity(norm_alphas, norm_geom_feature, dim=0)


    # 3b. Alpha多样性奖励:
    # 鼓励模型不要输出一个恒定的alpha值，而是根据几何形状进行探索。
    # 标准差越大，说明模型输出的alpha值越丰富。
    reward_diversity = torch.std(alphas)

    # 3c. Alpha幅度温和惩罚:
    # 防止alpha值爆炸性增长。使用log来温和地惩罚过大的均值。
    penalty_magnitude = -torch.log(1 + torch.mean(alphas))

    # --- Part 4: 组合总奖励 ---
    total_reward = (weights['w_correlation'] * reward_correlation +
                    weights['w_diversity'] * reward_diversity +
                    weights['w_magnitude'] * penalty_magnitude)

    return total_reward.item()


# --- 5. 训练主函数 (使用V5奖励) ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
    NUM_POINTS = 2048
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0003  # 可以从 2e-4 到 5e-4 之间尝试
    EPOCHS = 50
    REWARD_BASELINE_DECAY = 0.95

    # --- V5版奖励权重 (全新，更可靠) ---
    REWARD_WEIGHTS_V5 = {
        'w_correlation': 2.0,  # 主要目标：让alpha分布匹配几何特征
        'w_diversity': 0.5,    # 次要目标：鼓励alpha值的多样性，防止坍缩
        'w_magnitude': 0.2,    # 启发式：温和地惩罚过大的alpha值
    }
    # V5奖励函数中K邻居参数
    K_NEIGHBORS_FOR_REWARD = 16

    if not os.path.isdir(SHAPENET_PATH) or "/path/to/your/" in SHAPENET_PATH:
        print("="*80 + f"\nFATAL ERROR: Please update the SHAPENET_PATH variable in the code.\n" + "="*80); exit()

    # 设置检查点加载逻辑
    START_EPOCH = 0
    file_name = f"advanced_model_v3_epoch_{START_EPOCH}.pth" # 可以更新命名方案为v5
    CHECKPOINT_PATH = os.path.join(save_directory, file_name)
    
    # Tensorboard Visualizer
    writer = SummaryWriter('runs/adaptive_alpha_v5_experiment')
    
    model = PyG_PointNet2_Alpha_Predictor().to(DEVICE)
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH, num_points=NUM_POINTS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ Resuming training from checkpoint: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print(f"🟡 Checkpoint file '{CHECKPOINT_PATH}' not found. Starting training from scratch.")
        START_EPOCH = 0

    # [重要] V5奖励的期望值更接近0，所以从0开始更合理
    reward_baseline = 0.0
    global_step = 0
    
    print(f"Starting training on {DEVICE} with V5 reward weights: {REWARD_WEIGHTS_V5}")
    
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # 探索衰减: 初始标准差可以设得高一些以鼓励探索
        current_std = max(0.20 * (0.96**epoch), 0.01)

        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)
            points_dense, mask = to_dense_batch(batch_data.pos, batch_data.batch)
            
            policy = model(batch_data)
            policy.scale = torch.ones_like(policy.loc) * current_std 
            sampled_alphas_dense = policy.sample()

            batch_rewards = []
            for i in range(points_dense.shape[0]):
                # 提取出当前样本的有效点和对应的alpha值
                sample_points = points_dense[i, mask[i]]
                sample_alphas = sampled_alphas_dense[i, :, mask[i]].squeeze()

                # [!!! 核心变化 !!!]
                # 不再进行耗时且不稳定的重建
                # 直接调用 V5 奖励函数
                reward = calculate_reward_v5(sample_alphas, 
                                             sample_points, 
                                             K_NEIGHBORS_FOR_REWARD, 
                                             REWARD_WEIGHTS_V5, 
                                             DEVICE)
                batch_rewards.append(reward)
            
            rewards_tensor = torch.tensor(batch_rewards, device=DEVICE, dtype=torch.float32)
            avg_reward = rewards_tensor.mean().item()
            advantage = rewards_tensor - reward_baseline
            
            log_probs_dense = policy.log_prob(sampled_alphas_dense)
            # 根据mask确保只计算有效点的log_prob
            log_probs_sum_per_sample = (log_probs_dense * mask.unsqueeze(1)).sum(dim=[1, 2])
            loss = - (log_probs_sum_per_sample * advantage).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            reward_baseline = REWARD_BASELINE_DECAY * reward_baseline + (1 - REWARD_BASELINE_DECAY) * avg_reward
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_reward=f"{avg_reward:.4f}", baseline=f"{reward_baseline:.4f}")

            # --- Tensorboard Logging ---
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Reward/average_reward', avg_reward, global_step)
            writer.add_scalar('Reward/baseline', reward_baseline, global_step)
            writer.add_scalar('Hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Hyperparameters/exploration_std', current_std, global_step)
            writer.add_histogram('Alphas/sampled_distribution', sampled_alphas_dense[mask.unsqueeze(1).expand_as(sampled_alphas_dense)], global_step)
            writer.add_histogram('Reward/advantage_distribution', advantage, global_step)

            global_step += 1
        
        scheduler.step()
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            # 建议更新模型命名以反映新的策略
            save_file_name = f"advanced_model_v5_epoch_{epoch+1}.pth"
            save_path = os.path.join(save_directory, save_file_name)
            torch.save(model.state_dict(), save_path)
            print(f"\n✅ Model saved to {save_path}")

    writer.close()
    print("Training finished. TensorBoard logs saved.")


if __name__ == '__main__':
    main()