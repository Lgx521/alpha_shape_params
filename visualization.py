# ==============================================================================
#      Adaptive Alpha Shape Inference and Visualization (Pure Open3D Version)
#
#  这个脚本做了以下事情:
#  1. 加载您训练好的模型和一个测试用的点云数据。
#  2. 对点云进行推理，得到逐点的alpha预测值。
#  3. [重要更新] 使用纯Open3D内置的Alpha Shape功能重建三维网格，不再需要CGAL。
#  4. 使用Open3D将原始点云、Alpha热力图和重建网格并排显示。
#
#  作者: Gemini (Google AI)
#  日期: August 11, 2025 (已修正)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import argparse
import random

# --- 1. 核心依赖导入 ---
try:
    import open3d as o3d
    print("Open3D library found.")
except ImportError:
    print("FATAL ERROR: 'open3d' not installed. Run: pip install open3d"); exit()
try:
    import trimesh
    print("Trimesh library found.")
except ImportError:
    print("FATAL ERROR: 'trimesh' not installed. Run: pip install trimesh"); exit()
try:
    from torch_geometric.data import Data
    # [修正] 添加 to_dense_batch 导入
    from torch_geometric.utils import to_dense_batch
    from torch_geometric.nn import knn_interpolate, fps, knn_graph
    print("PyTorch Geometric found.")
except ImportError as e:
    print(f"FATAL ERROR: PyG not installed correctly. Error: {e}"); exit()
try:
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    print("PyTorch3D found.")
except ImportError as e:
    print(f"FATAL ERROR: PyTorch3D not found. Error: {e}"); exit()
try:
    # 尝试导入matplotlib以获得更好的颜色映射，如果失败则优雅降级
    import matplotlib.pyplot as plt
    print("Matplotlib found for color mapping.")
except ImportError:
    plt = None
    print("Matplotlib not found, using a simpler heatmap color.")

# [修正] 添加 Normal (正态分布) 导入
from torch.distributions import Normal


class PyG_PointNet2_Alpha_Predictor_2(torch.nn.Module):
    def __init__(self, k_interp=3):
        super().__init__()
        self.k = k_interp

        def create_mlp(in_channels, out_channels_list, last_relu=True):
            layers = []
            for out_channels in out_channels_list:
                layers.append(nn.Linear(in_channels, out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if not last_relu:
                return nn.Sequential(*layers[:-1])
            return nn.Sequential(*layers)

        # --- 編碼器 (下採樣) ---
        # 處理原始點 (坐標+法線)
        self.sa1_mlp = create_mlp(3 + 3, [64, 128])
        # 處理上一層的特徵
        self.sa2_mlp = create_mlp(128 + 3, [128, 256])
        # 瓶頸層
        self.sa3_mlp = create_mlp(256 + 3, [256, 512])

        # --- 解碼器 (上採樣與特徵融合) ---
        self.fp2_mlp = create_mlp(512 + 256, [256, 256])  # 上採樣(l2->l1) + 跳躍連接(l1)
        self.fp1_mlp = create_mlp(256 + 128, [256, 128])   # 上採樣(l1->l0) + 跳躍連接(l0)
        
        # 輸出前的最終特徵融合層
        self.fp0_mlp = create_mlp(128 + 3 + 3, [128, 128]) # 融合特徵 + 原始坐標 + 原始法線

        # --- 輸出頭 ---
        self.head_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        self.softplus = nn.Softplus()

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        # --- 編碼器 ---
        # Level 0 (原始點)
        l0_pos, l0_x, l0_batch = pos, x, batch
        l0_features = self.sa1_mlp(torch.cat([l0_pos, l0_x], dim=1))

        # Level 1
        l1_idx = fps(l0_pos, l0_batch, ratio=0.25)
        l1_pos, l1_batch = l0_pos[l1_idx], l0_batch[l1_idx]
        l1_agg_features = knn_interpolate(l0_features, l0_pos, l1_pos, l0_batch, l1_batch, k=16)
        l1_features = self.sa2_mlp(torch.cat([l1_agg_features, l1_pos], dim=1))

        # Level 2 (瓶頸)
        l2_idx = fps(l1_pos, l1_batch, ratio=0.25)
        l2_pos, l2_batch = l1_pos[l2_idx], l1_batch[l2_idx]
        l2_agg_features = knn_interpolate(l1_features, l1_pos, l2_pos, l1_batch, l2_batch, k=16)
        l2_features = self.sa3_mlp(torch.cat([l2_agg_features, l2_pos], dim=1))
        
        # --- 解碼器 ---
        # 從 Level 2 上採樣到 Level 1
        l1_up_features = knn_interpolate(l2_features, l2_pos, l1_pos, l2_batch, l1_batch, k=self.k)
        l1_fp_features = self.fp2_mlp(torch.cat([l1_up_features, l1_features], dim=1))
        
        # 從 Level 1 上採樣到 Level 0
        l0_up_features = knn_interpolate(l1_fp_features, l1_pos, l0_pos, l1_batch, l0_batch, k=self.k)
        l0_fp_features = self.fp1_mlp(torch.cat([l0_up_features, l0_features], dim=1))

        # 最終特徵融合
        final_features = self.fp0_mlp(torch.cat([l0_fp_features, l0_pos, l0_x], dim=1))

        # --- 輸出頭 ---
        alpha_mean = self.head_mlp(final_features)
        alpha_mean_dense, _ = to_dense_batch(alpha_mean, batch)
        alpha_mean_dense = alpha_mean_dense.permute(0, 2, 1)

        MIN_ALPHA = 0.01 
        alpha_mean_activated = self.softplus(alpha_mean_dense) + MIN_ALPHA
        
        # 返回一個正態分佈策略
        policy = Normal(alpha_mean_activated, torch.ones_like(alpha_mean_activated))
        return policy

# --- 2. 基于PyG的PointNet++ Alpha预测模型 (V9 - 最终、最可靠版) ---
class PyG_PointNet2_Alpha_Predictor_1(torch.nn.Module):
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
            return nn.Sequential(*layers[:-1])

        # --- 编码器 (SA) Layers ---
        self.sa1_mlp = create_mlp(6, [64, 64, 128])
        self.sa2_mlp = create_mlp(128 + 3, [128, 128, 256])
        self.sa3_mlp = create_mlp(256 + 3, [256, 512, 1024])

        # --- 解码器 (FP) Layers ---
        self.fp3_mlp = create_mlp(1024 + 256, [256, 256])
        self.fp2_mlp = create_mlp(256 + 128, [256, 128])
        self.fp1_mlp = create_mlp(128 + 6, [128, 128, 128])

        # --- 输出头 ---
        self.head_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 1)
        )
        self.softplus = nn.Softplus()

    def forward(self, data):
        # [V9 修正] 采用清晰、标准的U-Net前向传播逻辑
        x, pos, batch = data.x, data.pos, data.batch

        # --- 编码器 (Encoder) ---
        # Level 0 -> Level 1
        l1_idx = fps(pos, batch, ratio=0.25)
        l1_pos, l1_batch = pos[l1_idx], batch[l1_idx]
        l1_x = F.relu(self.sa1_mlp(x[l1_idx]))

        # Level 1 -> Level 2
        l2_idx = fps(l1_pos, l1_batch, ratio=0.25)
        # ========================= [BUG FIX 关键修复] =========================
        # 错误原因: l2_batch 应该从 l1_batch 中索引, 而不是最原始的 batch
        # 原始错误代码: l2_pos, l2_batch = l1_pos[l2_idx], batch[l2_idx]
        l2_pos, l2_batch = l1_pos[l2_idx], l1_batch[l2_idx]
        # ====================================================================
        l2_x = F.relu(self.sa2_mlp(torch.cat([l1_x[l2_idx], l2_pos], dim=1)))

        # Level 2 -> Level 3 (最深层)
        l3_idx = fps(l2_pos, l2_batch, ratio=0.25)
        # ========================= [BUG FIX 关键修复] =========================
        # 错误原因: l3_batch 应该从 l2_batch 中索引, 而不是其他
        # 原始错误代码: l3_pos, l3_batch = l2_pos[l3_idx], batch[l2_idx]
        l3_pos, l3_batch = l2_pos[l3_idx], l2_batch[l3_idx]
        # ====================================================================
        l3_features = F.relu(self.sa3_mlp(torch.cat([l2_x[l3_idx], l3_pos], dim=1)))

        # --- 解码器 (Decoder) ---
        # FP for Level 3 -> 2
        l2_interp = knn_interpolate(l3_features, l3_pos, l2_pos, l3_batch, l2_batch, k=self.k)
        l2_fp = F.relu(self.fp3_mlp(torch.cat([l2_interp, l2_x], dim=1))) # 跳跃连接用l2_x

        # FP for Level 2 -> 1
        l1_interp = knn_interpolate(l2_fp, l2_pos, l1_pos, l2_batch, l1_batch, k=self.k)
        l1_fp = F.relu(self.fp2_mlp(torch.cat([l1_interp, l1_x], dim=1))) # 跳跃连接用l1_x

        # FP for Level 1 -> 0
        l0_interp = knn_interpolate(l1_fp, l1_pos, pos, l1_batch, batch, k=self.k)
        l0_fp = F.relu(self.fp1_mlp(torch.cat([l0_interp, x], dim=1)))

        # --- 输出头 ---
        alpha_mean = self.head_mlp(l0_fp)
        
        # --- 输出格式化 ---
        alpha_mean_dense, _ = to_dense_batch(alpha_mean, batch)
        alpha_mean_dense = alpha_mean_dense.permute(0, 2, 1)
        alpha_mean_activated = self.softplus(alpha_mean_dense)
        alpha_std = torch.ones_like(alpha_mean_activated) * 0.01
        policy = Normal(alpha_mean_activated, alpha_std)
        
        return policy

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
            return nn.Sequential(*layers[:-1]) # 移除最后一个ReLU

        # --- 编码器 (SA) Layers ---
        self.sa1_mlp = create_mlp(6, [64, 128])
        self.sa2_mlp = create_mlp(128 + 3, [128, 256])
        self.sa3_mlp = create_mlp(256 + 3, [256, 512])

        # --- 解码器 (FP) Layers ---
        self.fp2_mlp = create_mlp(512 + 256, [256, 256])
        self.fp1_mlp = create_mlp(256 + 128, [256, 128])
        self.fp0_mlp = create_mlp(128 + 6, [128, 128])
        
        # --- 输出头 ---
        self.head_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        self.softplus = nn.Softplus()

    def forward(self, data):
        # 这个forward逻辑是根据上面的__init__结构推导出的标准U-Net流程
        x, pos, batch = data.x, data.pos, data.batch

        # --- 编码器 (Encoder) ---
        # Level 0 -> Level 1
        l1_idx = fps(pos, batch, ratio=0.25)
        l1_pos, l1_batch = pos[l1_idx], batch[l1_idx]
        l1_skip_features = F.relu(self.sa1_mlp(x[l1_idx]))

        # Level 1 -> Level 2
        l2_idx = fps(l1_pos, l1_batch, ratio=0.25)
        l2_pos, l2_batch = l1_pos[l2_idx], l1_batch[l2_idx]
        l2_skip_features = F.relu(self.sa2_mlp(torch.cat([l1_skip_features[l2_idx], l2_pos], dim=1)))
        
        # Level 2 -> Level 3 (最深层)
        l3_idx = fps(l2_pos, l2_batch, ratio=0.25)
        l3_pos, l3_batch = l2_pos[l3_idx], l2_batch[l3_idx]
        l3_features = F.relu(self.sa3_mlp(torch.cat([l2_skip_features[l3_idx], l3_pos], dim=1)))
        
        # --- 解码器 (Decoder) ---
        # FP for Level 3 -> 2
        l2_interp = knn_interpolate(l3_features, l3_pos, l2_pos, l3_batch, l2_batch, k=self.k)
        l2_fp = F.relu(self.fp2_mlp(torch.cat([l2_interp, l2_skip_features], dim=1)))

        # FP for Level 2 -> 1
        l1_interp = knn_interpolate(l2_fp, l2_pos, l1_pos, l2_batch, l1_batch, k=self.k)
        l1_fp = F.relu(self.fp1_mlp(torch.cat([l1_interp, l1_skip_features], dim=1)))
        
        # FP for Level 1 -> 0 (原始点)
        l0_interp = knn_interpolate(l1_fp, l1_pos, pos, l1_batch, batch, k=self.k)
        l0_fp = F.relu(self.fp0_mlp(torch.cat([l0_interp, x], dim=1)))
        
        # --- 输出头 ---
        alpha_mean = self.head_mlp(l0_fp)
        
        # --- 输出格式化 ---
        alpha_mean_dense, _ = to_dense_batch(alpha_mean, batch)
        alpha_mean_dense = alpha_mean_dense.permute(0, 2, 1)
        alpha_mean_activated = self.softplus(alpha_mean_dense)
        alpha_std = torch.ones_like(alpha_mean_activated) * 0.01
        policy = Normal(alpha_mean_activated, alpha_std)
        
        return policy
    

# --- 3. [新] 纯Open3D重建函数 ---
def reconstruct_with_open3d(points_np, alphas_tensor):
    """
    [纯Open3D实现]
    使用Open3D内置的Alpha Shape功能进行网格重建。

    Args:
        points_np (np.ndarray): NumPy格式的点云 (N, 3)
        alphas_tensor (torch.Tensor): PyTorch格式的alpha预测值 (N,)

    Returns:
        open3d.geometry.TriangleMesh or None: 重建出的网格，如果失败则返回None。
    """
    if points_np.shape[0] < 4:
        return None

    # a) 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # b) 决定全局Alpha值
    median_alpha = torch.median(alphas_tensor).item()
    
    # c) 设置一个合理的alpha下限，防止Open3D因值过小而失败
    if median_alpha <= 1e-6:
        median_alpha = 0.01
    
    # d) 调用Open3D的核心函数进行重建
    reconstructed_mesh = None
    try:
        reconstructed_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=median_alpha
        )
        if not reconstructed_mesh.has_vertices() or not reconstructed_mesh.has_triangles():
            print(f"WARNING: Open3D with alpha={median_alpha:.4f} produced an empty mesh.")
            return None
            
    except Exception as e:
        print(f"WARNING: Open3D Alpha Shape failed with alpha={median_alpha:.4f}. Error: {e}")
        return None

    return reconstructed_mesh


def visualize_inference(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 1. 加载训练好的模型 ---
    if not os.path.exists(args.model_path):
        print(f"FATAL ERROR: Model file not found at {args.model_path}"); exit()
    print(f"Loading trained model from: {args.model_path}")
    
    try:
        # ======================= [逻辑修正] =======================
        # 实例化已修复的、标记为 "V9-最终版" 的模型
        model = PyG_PointNet2_Alpha_Predictor().to(DEVICE)
        # ========================================================
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model state_dict. Ensure the model definition in this script matches the one used for training.")
        print(f"Error details: {e}")
        exit()

    # --- 2. 数据加载与推理 ---
    print(f"Scanning for models in: {args.shapenet_path}")
    paths = glob.glob(os.path.join(args.shapenet_path, "**/model_normalized.ply"), recursive=True)
    if not paths: raise ValueError("No .ply files found in the specified ShapeNet path.")

    if args.sample_index is not None:
        if 0 <= args.sample_index < len(paths): test_file_path = paths[args.sample_index]
        else: print(f"Error: Index {args.sample_index} is out of bounds."); return
    else:
        test_file_path = random.choice(paths); print("No index provided, choosing a random model.")
    
    print(f"Visualizing sample: {test_file_path}")

    try:
        # --- 数据加载 ---
        mesh_trimesh = trimesh.load(test_file_path)
        verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh_trimesh.faces, dtype=torch.long)
        pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
        
        points_tensor, normals_tensor = sample_points_from_meshes(
            pytorch3d_mesh,
            num_samples=args.num_points,
            return_normals=True
        )
        points_tensor = points_tensor.squeeze(0)
        normals_tensor = normals_tensor.squeeze(0)

        input_features = torch.cat([points_tensor, normals_tensor], dim=1)
        
        # --- [代码健壮性] ---
        # 推荐在创建Data对象时就传入所有属性, 包括batch
        batch_vector = torch.zeros(points_tensor.shape[0], dtype=torch.long)
        data = Data(pos=points_tensor, x=input_features, batch=batch_vector).to(DEVICE)

        points_np = points_tensor.cpu().numpy()

        # --- 推理 ---
        print("Running inference to predict alpha values...")
        with torch.no_grad():
            policy = model(data)
            predicted_alphas = policy.loc.squeeze().cpu()

        print(f"Inference complete. Predicted alpha range: [{predicted_alphas.min():.4f}, {predicted_alphas.max():.4f}]")

        # --- 重建与可视化 ---
        print("Reconstructing mesh using Open3D...")
        reconstructed_mesh_o3d = reconstruct_with_open3d(points_np, predicted_alphas)

        print("Preparing geometries for visualization...")
        # a) 原始点云 (灰色)
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(points_np)
        pcd_original.paint_uniform_color([0.7, 0.7, 0.7])

        # b) Alpha值热力图点云
        alphas_normalized = (predicted_alphas - predicted_alphas.min()) / (predicted_alphas.max() - predicted_alphas.min() + 1e-9)
        if plt:
            colors = plt.get_cmap('viridis')(alphas_normalized.numpy())[:, :3]
        else:
            # Note: This Open3D method does not exist. Using a manual gradient as a fallback.
            heatmap = np.zeros((len(alphas_normalized), 3))
            heatmap[:, 0] = alphas_normalized # Red channel
            heatmap[:, 1] = 1 - alphas_normalized # Green channel
            colors = heatmap

        pcd_alpha_heatmap = o3d.geometry.PointCloud()
        pcd_alpha_heatmap.points = o3d.utility.Vector3dVector(points_np)
        pcd_alpha_heatmap.colors = o3d.utility.Vector3dVector(colors)

        # c) 准备重建网格
        geometries_to_draw = []
        pcd_alpha_heatmap.translate([-1.2, 0, 0])
        geometries_to_draw.extend([pcd_original, pcd_alpha_heatmap])
        
        if reconstructed_mesh_o3d is not None:
            reconstructed_mesh_o3d.compute_vertex_normals()
            reconstructed_mesh_o3d.paint_uniform_color([0.1, 0.9, 0.1])
            reconstructed_mesh_o3d.translate([1.2, 0, 0])
            geometries_to_draw.append(reconstructed_mesh_o3d)
            
        print("Launching Open3D visualizer...")
        o3d.visualization.draw_geometries(
            geometries_to_draw,
            window_name="Input (Center) | Alpha Heatmap (Left) | Reconstruction (Right)",
            width=1200, height=800, mesh_show_wireframe=True
        )
        print("Visualization window closed.")

    except Exception as e:
        print(f"\nFATAL ERROR during processing sample {test_file_path}.")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the input, alpha heatmap, and reconstruction result of a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--shapenet_path', type=str, required=True, help="Path to the root of the ShapeNetCore.v2 dataset.")
    parser.add_argument('--num_points', type=int, default=2048, help="Number of points to sample from the mesh (should match training).")
    parser.add_argument('--sample_index', type=int, default=None, help="Index of the sample to visualize. If not provided, a random one is chosen.")
    args = parser.parse_args()
    
    # 移除关于 torch.load 的 FutureWarning 警告
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    visualize_inference(args)
