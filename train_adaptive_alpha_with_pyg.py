# ==============================================================================
#                 Adaptive Alpha Shape Training with PyG & RL
#
#  这个脚本整合了以下内容:
#  1. PyTorch Geometric (PyG) 作为现代、高效的PointNet++后端.
#  2. 一个自监督的强化学习训练循环 (REINFORCE 策略梯度).
#  3. 使用 PyTorch3D 计算奖励 (倒角距离).
#  4. 调用 CGAL 执行(模拟的)Alpha Shape重建.
#
#  作者: Gemini (Google AI)
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

# PyTorch Geometric (PyG)
try:
    from torch_geometric.nn import PointNetConv, fps, radius
    from torch_geometric.nn.glob import global_max_pool
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_dense_batch
    print("PyTorch Geometric found and successfully imported.")
except ImportError as e:
    print("="*80)
    print("FATAL ERROR: PyTorch Geometric is not installed correctly.")
    print("Please follow the installation instructions at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    print(f"Import Error: {e}")
    print("="*80)
    exit()

# PyTorch3D
try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.io import load_obj
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    print("PyTorch3D found.")
except ImportError:
    print("FATAL ERROR: PyTorch3D not found. Please install with 'pip install pytorch3d'")
    exit()

# CGAL
try:
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Alpha_shape_3 import Alpha_shape_3, Mode
    print("CGAL-pybind found.")
except ImportError:
    print("WARNING: cgal-pybind not found. Reconstruction will be a DUMMY step.")


# --- 2. 基于PyG的PointNet++ Alpha预测模型 ---

class PyG_PointNet2_Alpha_Predictor(torch.nn.Module):
    """
    一个现代化的PointNet++模型，使用PyG实现。
    它采用U-Net（编码器-解码器）架构来为每个点预测alpha值。
    """
    def __init__(self):
        super().__init__()
        # 为了简化，我们使用MLP作为PointNetConv的内部网络
        def create_mlp(in_channels, out_channels_list):
            layers = []
            for out_channels in out_channels_list:
                layers.append(nn.Linear(in_channels, out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            return nn.Sequential(*layers)

        # --- 编码器 (下采样) ---
        self.sa1_conv = PointNetConv(create_mlp(3, [64, 64, 128]))
        self.sa2_conv = PointNetConv(create_mlp(128 + 3, [128, 128, 256]))
        self.sa3_conv = PointNetConv(create_mlp(256 + 3, [256, 512, 1024]))

        # --- 解码器 (上采样/特征传播) ---
        self.fp3_mlp = create_mlp(1024 + 256, [256, 256])
        self.fp2_mlp = create_mlp(256 + 128, [128, 128])
        self.fp1_mlp = create_mlp(128 + 3, [128, 128]) # 初始特征是3维坐标

        # --- 输出头 ---
        self.head_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)  # 输出alpha分布的均值
        )
        self.softplus = nn.Softplus()

    def forward(self, data):
        pos, batch = data.pos, data.batch

        # --- 编码器 ---
        # SA1
        idx1 = fps(pos, batch, ratio=0.25)
        row1, col1 = radius(pos, pos[idx1], r=0.2, batch_x=batch, batch_y=batch[idx1], max_num_neighbors=64)
        edge_index1 = torch.stack([col1, row1], dim=0)
        x1 = self.sa1_conv(pos, pos[idx1], edge_index1)
        pos1, batch1 = pos[idx1], batch[idx1]

        # SA2
        idx2 = fps(pos1, batch1, ratio=0.25)
        row2, col2 = radius(pos1, pos1[idx2], r=0.4, batch_x=batch1, batch_y=batch1[idx2], max_num_neighbors=64)
        edge_index2 = torch.stack([col2, row2], dim=0)
        x2 = self.sa2_conv(x1, x1[idx2], edge_index2)
        pos2, batch2 = pos1[idx2], batch1[idx2]
        
        # SA3
        idx3 = fps(pos2, batch2, ratio=0.25)
        row3, col3 = radius(pos2, pos2[idx3], r=0.8, batch_x=batch2, batch_y=batch2[idx3], max_num_neighbors=64)
        edge_index3 = torch.stack([col3, row3], dim=0)
        x3 = self.sa3_conv(x2, x2[idx3], edge_index3)
        pos3, batch3 = pos2[idx3], batch2[idx3]
        
        # --- 解码器 ---
        # FP3: 最近邻插值 (简化但有效)
        x2_up = F.interpolate(x3.unsqueeze(0), size=x2.size(0), mode='nearest').squeeze(0) # 简化插值
        x2_fp = self.fp3_mlp(torch.cat([x2_up, x2], dim=1))

        # FP2
        x1_up = F.interpolate(x2_fp.unsqueeze(0), size=x1.size(0), mode='nearest').squeeze(0)
        x1_fp = self.fp2_mlp(torch.cat([x1_up, x1], dim=1))

        # FP1
        x0_up = F.interpolate(x1_fp.unsqueeze(0), size=pos.size(0), mode='nearest').squeeze(0)
        x0_fp = self.fp1_mlp(torch.cat([x0_up, pos], dim=1))

        # --- 输出头 ---
        alpha_mean = self.head_mlp(x0_fp)  # [Total_Points, 1]

        # 转换为稠密格式以匹配RL流程
        alpha_mean_dense, _ = to_dense_batch(alpha_mean, batch) # [B, N, 1]
        alpha_mean_dense = alpha_mean_dense.permute(0, 2, 1)    # [B, 1, N]

        # 创建策略分布
        alpha_mean_activated = self.softplus(alpha_mean_dense)
        alpha_std = torch.ones_like(alpha_mean_activated) * 0.01 # 固定标准差
        policy = Normal(alpha_mean_activated, alpha_std)

        return policy


# --- 3. 数据加载 ---

class PyGShapeNetDataset(Dataset):
    """一个适配PyG的数据集，加载ShapeNet模型并返回Data对象"""
    def __init__(self, root_dir, num_points=2048, split='train'):
        self.root_dir = root_dir
        self.num_points = num_points
        # 根据官方推荐的划分获取文件列表
        # 这里简化为直接读取所有文件，实际项目中应使用官方的train/val/test划分
        self.paths = glob.glob(os.path.join(root_dir, "**/*.obj"), recursive=True)
        if not self.paths:
             raise ValueError(f"No .obj files found in {root_dir}. Check the path.")
        print(f"Found {len(self.paths)} models for the '{split}' split.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            verts, faces, _ = load_obj(self.paths[idx])
            mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
            points = sample_points_from_meshes(mesh, num_samples=self.num_points)
            return Data(pos=points.squeeze(0))
        except Exception as e:
            # print(f"Warning: Skipping corrupted file {self.paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self)) # 加载下一个


# --- 4. 强化学习环境与奖励 ---

def reconstruct_with_alpha_shape(points, alphas):
    """
    (模拟的)环境交互步骤。
    使用预测的alpha值和CGAL来重建网格。这是一个不可微分的黑箱。
    """
    # 如果CGAL未安装，返回一个由输入点构成的“伪网格”用于演示
    if 'CGAL' not in globals() or not 'Alpha_shape_3' in globals():
        # print("CGAL not found, returning dummy mesh.")
        dummy_faces = torch.arange(len(points)).reshape(1, -1)
        # 确保面片数量合理以避免Pytorch3D错误
        if dummy_faces.shape[1] > 3:
            dummy_faces = dummy_faces[:, :3]
        elif dummy_faces.shape[1] < 3:
             return None # 无法构成面片
        return Meshes(verts=[points.to(torch.float32)], faces=[dummy_faces])

    # 简化的自适应alpha应用：使用alpha值的中值作为全局alpha进行演示
    # 这是一个重要的简化，高级实现可能需要更复杂的局部alpha应用
    median_alpha = torch.median(alphas).item()
    if median_alpha <= 0: median_alpha = 1e-6 # 保证alpha为正

    points_cgal = [Point_3(p[0], p[1], p[2]) for p in points.cpu().tolist()]
    alpha_shape = Alpha_shape_3(points_cgal, median_alpha, Mode.GENERAL)
    
    # 尝试从alpha shape中提取表面网格
    try:
        verts_list, faces_list = alpha_shape.get_surface_mesh()
        if not verts_list or not faces_list:
            return None # 无法形成表面
        
        verts_tensor = torch.tensor(verts_list, dtype=torch.float32)
        faces_tensor = torch.tensor(faces_list, dtype=torch.long)
        return Meshes(verts=[verts_tensor], faces=[faces_tensor])
    except Exception:
        return None # CGAL提取失败

def calculate_reward(reconstructed_mesh, original_points):
    """计算奖励：奖励 = -倒角距离"""
    if reconstructed_mesh is None:
        return -10.0 # 对无法重建的情况给予巨大惩罚

    device = original_points.device
    reconstructed_mesh = reconstructed_mesh.to(device)

    # 从重建网格采样点以计算距离
    try:
        reconstructed_points = sample_points_from_meshes(
            reconstructed_mesh,
            num_samples=original_points.shape[0]
        )
    except Exception:
        return -10.0 # 采样失败也给予惩罚

    loss_chamfer, _ = chamfer_distance(
        reconstructed_points,
        original_points.unsqueeze(0)
    )
    
    # 奖励是损失的负数。倒角距离越小，奖励越高。
    return -loss_chamfer.item()


# --- 5. 训练主函数 ---

def main():
    # --- 超参数设置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
    # ---------------------------------------------------------
    
    NUM_POINTS = 2048
    BATCH_SIZE = 16 # 根据你的GPU显存调整
    LEARNING_RATE = 0.001
    EPOCHS = 100
    REWARD_BASELINE_DECAY = 0.95 # 用于更新奖励基线的衰减率

    print(f"Starting training on device: {DEVICE}")
    if not os.path.isdir(SHAPENET_PATH):
        print("="*80)
        print(f"FATAL ERROR: The path '{SHAPENET_PATH}' does not exist.")
        print("Please download the ShapeNetCore.v2 dataset and update the SHAPENET_PATH variable.")
        print("="*80)
        exit()

    # --- 初始化模型、数据和优化器 ---
    model = PyG_PointNet2_Alpha_Predictor().to(DEVICE)
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH, num_points=NUM_POINTS)
    # 使用PyG的DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    reward_baseline = 0.0 # 奖励基线，用于稳定训练

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)
            
            # 从PyG的Batch对象中获取稠密表示，以便逐样本处理
            points_dense, mask = to_dense_batch(batch_data.pos, batch_data.batch)
            
            # --- (a) 执行策略并采样动作 ---
            policy = model(batch_data)
            # .sample()是RL的关键，它引入了探索
            sampled_alphas = policy.sample() # [B, 1, N]
            
            batch_loss = 0
            current_rewards = []
            
            # --- 逐样本计算奖励和损失 ---
            for i in range(points_dense.shape[0]):
                # 获取单个样本的点云和对应的alpha值
                sample_points = points_dense[i, mask[i]]
                sample_alphas = sampled_alphas[i, :, mask[i]].squeeze()

                # --- (b) 环境交互（不可微分）---
                with torch.no_grad():
                    reconstructed_mesh = reconstruct_with_alpha_shape(sample_points, sample_alphas)
                
                # --- (c) 计算奖励 ---
                reward = calculate_reward(reconstructed_mesh, sample_points)
                current_rewards.append(reward)

                # --- (d) 计算策略梯度损失 ---
                # REINFORCE算法核心: Loss = -log_prob * Advantage
                advantage = reward - reward_baseline 
                
                # policy.log_prob()是可微分的
                log_prob = policy.log_prob(sampled_alphas).mean()
                
                # 我们希望最大化(log_prob * advantage), 所以梯度下降时对它取负
                loss = -log_prob * advantage
                batch_loss += loss

            # --- (e) 更新网络权重 ---
            optimizer.zero_grad()
            final_loss = batch_loss / BATCH_SIZE
            final_loss.backward()
            optimizer.step()
            
            # --- 更新奖励基线 ---
            avg_reward = sum(current_rewards) / len(current_rewards)
            reward_baseline = REWARD_BASELINE_DECAY * reward_baseline + (1 - REWARD_BASELINE_DECAY) * avg_reward

            progress_bar.set_postfix(loss=f"{final_loss.item():.4f}", avg_reward=f"{avg_reward:.4f}", baseline=f"{reward_baseline:.4f}")

        # 每隔几个周期保存一次模型
        if (epoch + 1) % 5 == 0:
            save_path = f"adaptive_alpha_model_pyg_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()