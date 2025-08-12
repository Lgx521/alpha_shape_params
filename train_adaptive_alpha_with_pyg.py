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

# def calculate_reward_v4(reconstructed_mesh, original_points, weights, device):
#     """
#     V4版奖励函数，基于拓扑分析，奖励有意义的几何组件。
#     """
#     # --- 1. 重建失败惩罚 ---
#     if reconstructed_mesh is None or reconstructed_mesh.verts_packed().shape[0] < 4:
#         return -10.0

#     reconstructed_mesh = reconstructed_mesh.to(device)

#     # --- 2. 拆分网格为连通组件 ---
#     # 我们巧妙地使用to_mitsuba函数，它内部会进行连通组件拆分
#     # 这是一个比自己写循环更高效、更鲁棒的方法
#     try:
#         # 这个函数会返回一个包含所有独立连通组件的列表
#         components = to_mitsuba(reconstructed_mesh, "component")
#     except Exception:
#         # 如果拆分失败，说明网格质量极差
#         return -10.0

#     if not components:
#         return -10.0

#     # --- 3. 组件数量惩罚 ---
#     # 如果网格过于破碎，给予温和惩罚。我们不希望有几百个小碎片。
#     # 使用log来让惩罚不至于太剧烈
#     reward_fragmentation = -torch.log(1.0 + torch.tensor(len(components), device=device))

#     # --- 4. 找到并分析最大的组件 ---
#     # 找到包含顶点数最多的那个组件
#     largest_component = max(components, key=lambda m: m.verts_packed().shape[0])
    
#     # a. 尺寸奖励: 直接奖励最大组件的尺寸
#     # 我们希望模型生成大的、有意义的结构
#     reward_size = torch.log(1.0 + torch.tensor(largest_component.verts_packed().shape[0], device=device))

#     # b. 保真度奖励 (只针对最大组件)
#     try:
#         component_points = sample_points_from_meshes(largest_component, num_samples=original_points.shape[0])
#         loss_chamfer, _ = chamfer_distance(component_points, original_points.unsqueeze(0))
#         # 用log代替线性惩罚，对小的误差更宽容，对大的误差惩罚更重
#         reward_fidelity = -torch.log(1.0 + 10.0 * loss_chamfer) 
#     except Exception:
#         reward_fidelity = -5.0 # 如果采样失败，给予一个固定惩罚

#     # c. 平滑度奖励 (只针对最大组件)
#     loss_laplacian = mesh_laplacian_smoothing(largest_component, method="uniform")
#     reward_smoothness = -torch.log(1.0 + loss_laplacian)
    
#     # --- 5. 组合总奖励 ---
#     total_reward = (weights['w_fidelity'] * reward_fidelity +
#                     weights['w_smoothness'] * reward_smoothness +
#                     weights['w_size'] * reward_size +
#                     weights['w_fragmentation'] * reward_fragmentation)

#     return total_reward.item()



def calculate_reward_v4(reconstructed_mesh, original_points, weights, device):
    """
    V4.1版奖励函数，使用手动实现的连通组件拆分，以兼容旧版PyTorch3D。
    """
    # --- 1. 重建失败惩罚 ---
    if reconstructed_mesh is None or reconstructed_mesh.verts_packed().shape[0] < 4:
        return -10.0

    reconstructed_mesh = reconstructed_mesh.to(device)

    # --- 2. [重要更新] 手动拆分网格为连通组件 ---
    # 这是您ROS代码中 cluster_connected_triangles 的PyTorch3D等价实现
    try:
        # .get_mesh_verts_faces(0) 获取批次中第一个（也是唯一一个）网格的顶点和面
        verts = reconstructed_mesh.verts_list()[0]
        faces = reconstructed_mesh.faces_list()[0]

        # 使用trimesh来执行连通组件分析，因为它非常鲁棒
        # 我们将PyTorch3D的网格数据临时转换成trimesh对象
        mesh_trimesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())
        
        # split()函数返回一个包含所有独立连通组件的trimesh对象列表
        components_trimesh = mesh_trimesh.split(only_watertight=False)
        
        if not components_trimesh:
            return -10.0 # 如果trimesh无法拆分或找不到组件
        
        # 将trimesh组件列表转换回PyTorch3D的Meshes对象列表
        components = []
        for comp_tm in components_trimesh:
            comp_verts = torch.tensor(comp_tm.vertices, dtype=torch.float32, device=device)
            comp_faces = torch.tensor(comp_tm.faces, dtype=torch.long, device=device)
            components.append(Meshes(verts=[comp_verts], faces=[comp_faces]))

    except Exception:
        # 如果在拆分过程中出错，说明网格质量极差
        return -10.0

    # --- 3. 组件数量惩罚 ---
    reward_fragmentation = -torch.log(1.0 + torch.tensor(len(components), device=device))

    # --- 4. 找到并分析最大的组件 ---
    try:
        largest_component = max(components, key=lambda m: m.verts_packed().shape[0])
    except ValueError:
        return -10.0 # 如果组件列表为空

    # a. 尺寸奖励
    reward_size = torch.log(1.0 + torch.tensor(largest_component.verts_packed().shape[0], device=device))

    # b. 保真度奖励 (只针对最大组件)
    try:
        component_points = sample_points_from_meshes(largest_component, num_samples=original_points.shape[0])
        loss_chamfer, _ = chamfer_distance(component_points, original_points.unsqueeze(0))
        reward_fidelity = -torch.log(1.0 + 10.0 * loss_chamfer) 
    except Exception:
        reward_fidelity = -5.0

    # c. 平滑度奖励 (只针对最大组件)
    loss_laplacian = mesh_laplacian_smoothing(largest_component, method="uniform")
    reward_smoothness = -torch.log(1.0 + loss_laplacian)
    
    # --- 5. 组合总奖励 ---
    total_reward = (weights['w_fidelity'] * reward_fidelity +
                    weights['w_smoothness'] * reward_smoothness +
                    weights['w_size'] * reward_size +
                    weights['w_fragmentation'] * reward_fragmentation)

    return total_reward.item()


# --- 5. 训练主函数 (V3版) ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
    NUM_POINTS = 2048
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0002
    EPOCHS = 50
    REWARD_BASELINE_DECAY = 0.95

    # # --- V3版奖励权重 (这是新的关键超参数，需要仔细调整) ---
    # REWARD_WEIGHTS_V3 = {
    #     'w_fidelity': 0.6,           # 主要目标：网格与点云的相似度
    #     'w_smoothness': 0.5,         # 次要目标：网格表面平滑
    #     'w_watertight': 1.0,         # 重要目标：网格的拓扑正确性
    #     'w_alpha_consistency': 1.5,  # 启发式：鼓励alpha场平滑
    #     'w_alpha_magnitude': 0.4,    # 启发式：惩罚过大的alpha值
    #     'w_alpha_diversity': 1.0     # 启发式：鼓励模型探索不同的alpha值
    # }

    # --- V4版奖励权重，专注于拓扑和主要组件 ---
    REWARD_WEIGHTS_V4 = {
        'w_fidelity': 1.5,      # 主要目标：最大组件与点云的相似度
        'w_smoothness': 0.5,    # 次要目标：最大组件的表面平滑
        'w_size': 1.0,          # 重要目标：奖励生成更大的主体结构
        'w_fragmentation': 0.3, # 启发式：温和地惩罚过于破碎的网格
    }

    # 设置要加载的检查点文件路径。如果文件不存在，则从头训练。
    START_EPOCH = 0 # <-- 请修改为加载模型的epoch数
    file_name = f"advanced_model_v3_epoch_{START_EPOCH}.pth"
    CHECKPOINT_PATH = os.path.join(save_directory, file_name)


    if not os.path.isdir(SHAPENET_PATH) or "/path/to/your/" in SHAPENET_PATH:
        print("="*80 + f"\nFATAL ERROR: Please update the SHAPENET_PATH variable in the code.\n" + "="*80); exit()

    # Tensorboard Visualizer
    writer = SummaryWriter('runs/adaptive_alpha_v3_experiment')
    
    model = PyG_PointNet2_Alpha_Predictor().to(DEVICE)
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH, num_points=NUM_POINTS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 引入学习率调度器，可以进一步稳定训练
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ Resuming training from checkpoint: {CHECKPOINT_PATH}")
        # 加载模型的状态字典 (权重)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("🟡 Checkpoint file not found. Starting training from scratch.")
        # 如果找不到文件，就从epoch 0开始
        START_EPOCH = 0

    reward_baseline = -5.0 # 初始化一个更现实的基线

    # Tensorboard Visualizer
    global_step = 0
    
    # print(f"Starting training on {DEVICE} with V3 reward weights: {REWARD_WEIGHTS_V3}")
    print(f"Starting training on {DEVICE} with V4 reward weights: {REWARD_WEIGHTS_V4}")
    
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # --- 探索衰减 ---
        # 动态调整策略的标准差，实现从探索到利用的过渡
        # 初始std为0.1，最终衰减到0.01
        current_std = max(0.15 * (0.96**epoch), 0.01)

        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)
            points_dense, mask = to_dense_batch(batch_data.pos, batch_data.batch)
            
            # --- 前向传播 ---
            # 修改模型forward的调用方式，传入std
            policy = model(batch_data) # model的forward不需要改动
            # 在采样前手动修改策略的std
            policy.scale = torch.ones_like(policy.loc) * current_std 
            sampled_alphas_dense = policy.sample()

            batch_rewards = []
            for i in range(points_dense.shape[0]):
                sample_points = points_dense[i, mask[i]]
                # 从稠密张量中提取对应样本的alpha值
                sample_alphas = sampled_alphas_dense[i, :, mask[i]].squeeze()

                with torch.no_grad():
                    reconstructed_mesh = reconstruct_with_alpha_shape(sample_points, sample_alphas)
                    # 使用V3奖励函数
                    # reward = calculate_reward_v3(reconstructed_mesh, sample_points, sample_alphas, REWARD_WEIGHTS_V3, DEVICE)
                    reward = calculate_reward_v4(reconstructed_mesh, sample_points, REWARD_WEIGHTS_V4, DEVICE)
                    batch_rewards.append(reward)
            
            rewards_tensor = torch.tensor(batch_rewards, device=DEVICE, dtype=torch.float32)
            avg_reward = rewards_tensor.mean().item()
            advantage = rewards_tensor - reward_baseline
            
            # --- 损失计算 (使用修正后的正确方法) ---
            log_probs_dense = policy.log_prob(sampled_alphas_dense)
            log_probs_sum_per_sample = (log_probs_dense * mask.unsqueeze(1)).sum(dim=[1, 2])
            loss = - (log_probs_sum_per_sample * advantage).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            reward_baseline = REWARD_BASELINE_DECAY * reward_baseline + (1 - REWARD_BASELINE_DECAY) * avg_reward
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_reward=f"{avg_reward:.4f}", baseline=f"{reward_baseline:.4f}", std=f"{current_std:.3f}")

            # --- 3. <<< TENSORBOARD >>> 在每一步记录关键指标 ---
            # 使用 global_step 作为 X 轴，确保图表连续
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Reward/average_reward', avg_reward, global_step)
            writer.add_scalar('Reward/baseline', reward_baseline, global_step)
            writer.add_scalar('Hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Hyperparameters/exploration_std', current_std, global_step)
            
            # 记录分布情况，对于调试非常有用
            writer.add_histogram('Alphas/sampled_distribution', sampled_alphas_dense, global_step)
            writer.add_histogram('Reward/advantage_distribution', advantage, global_step)

            global_step += 1 # 更新全局步数
        
        scheduler.step() # 更新学习率
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            file_name = f"advanced_model_v3_epoch_{epoch+1}.pth"
            save_path = os.path.join(save_directory, file_name)
            torch.save(model.state_dict(), save_path)

    # --- 4. <<< TENSORBOARD >>> 训练结束后关闭writer ---
    writer.close()
    print("Training finished. TensorBoard logs saved.")

if __name__ == '__main__':
    main()