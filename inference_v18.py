# visualize_reconstruction.py
import torch
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from tqdm import tqdm
import os
import glob
import argparse

# ==============================================================================
# 1. 从训练脚本中复制核心类定义
#    为了使此脚本能独立运行，直接将模型和数据集类的定义复制于此。
# ==============================================================================

# --- 模型定义: SDF自解码器 (SDF Auto-Decoder) ---
class SDFAutoDecoder(torch.nn.Module):
    def __init__(self, num_shapes, latent_dim=256, mlp_hidden_dim=256):
        super().__init__()
        self.latent_codes = torch.nn.Embedding(num_shapes, latent_dim)
        torch.nn.init.normal_(self.latent_codes.weight.data, 0.0, 0.01)
        self.sdf_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 3, mlp_hidden_dim), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_hidden_dim, mlp_hidden_dim), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_hidden_dim, 1)
        )

    # def forward(self, shape_indices, query_points):
    #     B = 1 if shape_indices.dim() == 0 else shape_indices.shape[0]
    #     num_queries = query_points.shape[1]
        
    #     latent_z = self.latent_codes(shape_indices)
    #     if B > 1:
    #          latent_z_expanded = latent_z.unsqueeze(1).expand(-1, num_queries, -1)
    #     else: # 处理单个样本的情况
    #          latent_z_expanded = latent_z.unsqueeze(0).expand(num_queries, -1)
    #          query_points = query_points.squeeze(0)

    #     sdf_head_input = torch.cat([latent_z_expanded, query_points], dim=-1)
    #     predicted_sdf = self.sdf_head(sdf_head_input)
        
    #     # 如果是单个样本，保持输出形状一致性
    #     if B == 1 and predicted_sdf.dim() > 1:
    #         predicted_sdf = predicted_sdf.unsqueeze(0)
            
    #     return predicted_sdf

    def forward(self, shape_indices, query_points):
        """
        Args:
            shape_indices (Tensor): 形状的索引, shape (B,).
            query_points (Tensor): 查询点, shape (B, num_queries, 3).
        """
        # 这个实现同时适用于训练 (B > 1) 和推理 (B = 1)
        B, num_queries, _ = query_points.shape
        
        # 从“密码本”中查找对应的隐编码
        # shape_indices: (B,) -> latent_z: (B, latent_dim)
        latent_z = self.latent_codes(shape_indices) 
        
        # 将隐编码扩展以匹配查询点的数量
        # latent_z: (B, latent_dim) -> unsqueeze(1) -> (B, 1, latent_dim)
        # -> expand(...) -> (B, num_queries, latent_dim)
        latent_z_expanded = latent_z.unsqueeze(1).expand(-1, num_queries, -1)
        
        # 拼接隐编码和查询点坐标
        # 输入形状: (B, num_queries, latent_dim + 3)
        sdf_head_input = torch.cat([latent_z_expanded, query_points], dim=-1)
        
        predicted_sdf = self.sdf_head(sdf_head_input)
        
        # 输出形状: (B, num_queries, 1)
        return predicted_sdf

# --- 数据加载器定义 (仅用于获取原始点云) ---
class PyGShapeNetDatasetWithIdx(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.processed_data_folder = os.path.join(root_dir, "processed_points_with_normals")
        self.paths = sorted(glob.glob(os.path.join(self.processed_data_folder, "**/*.pt"), recursive=True))
        if not self.paths: raise ValueError(f"在 '{self.processed_data_folder}' 中未找到预处理的 '.pt' 文件。")
        print(f"为可视化找到了 {len(self.paths)} 个模型。")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data_dict = torch.load(self.paths[idx], weights_only=False)
        points, normals = data_dict['pos'], data_dict['x']
        center = points.mean(dim=0)
        points_centered = points - center
        scale = (points_centered.norm(p=2, dim=1)).max()
        points_normalized = points_centered / scale
        return {'pos': points_normalized, 'x': normals, 'idx': idx}

# ==============================================================================
# 2. 核心可视化函数
# ==============================================================================

def get_mesh_from_sdf(model, shape_idx, device, resolution=128, batch_size=65536):
    """
    使用Marching Cubes算法从模型的SDF预测中提取网格。

    Args:
        model (SDFAutoDecoder): 训练好的模型。
        shape_idx (int): 要重建的形状在数据集中的索引。
        device (torch.device): 计算设备 (cuda/cpu)。
        resolution (int): 网格分辨率，越高越精细。
        batch_size (int): 推理时的批处理大小，防止显存溢出。

    Returns:
        open3d.geometry.TriangleMesh: 重建出的网格。
    """
    # 创建一个标准化的三维坐标网格
    grid_range = [-1.0, 1.0]
    grid_pts = np.linspace(grid_range[0], grid_range[1], resolution)
    x, y, z = np.meshgrid(grid_pts, grid_pts, grid_pts, indexing='ij')
    query_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    query_points_tensor = torch.tensor(query_points, dtype=torch.float32).to(device)

    # 将形状索引转换为Tensor
    shape_idx_tensor = torch.tensor([shape_idx], device=device)

    sdf_values = []
    
    print(f"正在查询 {query_points_tensor.shape[0]} 个点的SDF值...")
    # 分批次进行推理，避免OOM
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, query_points_tensor.shape[0], batch_size), desc="SDF预测"):
            batch_points = query_points_tensor[i : i + batch_size].unsqueeze(0) # 增加Batch维度
            
            # 模型需要 (B, num_queries, 3) 形状的输入
            predicted_sdf = model(shape_idx_tensor, batch_points).squeeze(0)
            sdf_values.append(predicted_sdf.cpu().numpy())

    sdf_grid = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)

    print("正在使用Marching Cubes算法生成网格...")
    try:
        # 运行marching cubes算法，提取0等值面
        verts, faces, normals, _ = marching_cubes(sdf_grid, level=0.0, spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution))
        # 将顶点坐标从[0, 2]范围平移到[-1, 1]范围
        verts -= 1.0
    except (RuntimeError, ValueError) as e:
        print(f"Marching Cubes执行失败: {e}. 可能SDF场中没有过零点。")
        return None

    # 创建Open3D网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return mesh

def main():
    parser = argparse.ArgumentParser(description="SDF自解码器模型重建与可视化脚本")
    parser.add_argument('--checkpoint', type=str, default="/home/sgan/alpha_shape_params/checkpoints_v18/autodecoder_v18_epoch_10.pth", help="训练好的模型检查点文件路径 (.pth)")
    parser.add_argument('--shapenet_path', type=str, default="./ShapeNetCore.v2/ShapeNetCore.v2", help="ShapeNet数据集根目录")
    parser.add_argument('--shape_idx', type=int, default=10, help="要可视化的数据在数据集中的索引")
    parser.add_argument('--resolution', type=int, default=128, help="Marching Cubes算法的分辨率")
    args = parser.parse_args()

    # --- 1. 环境与设备配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件未找到: {args.checkpoint}")

    # --- 2. 加载数据集和模型 ---
    dataset = PyGShapeNetDatasetWithIdx(root_dir=args.shapenet_path)
    num_shapes = len(dataset)
    
    model = SDFAutoDecoder(num_shapes=num_shapes).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    print(f"成功从 '{args.checkpoint}' 加载模型权重。")

    # --- 3. 从SDF重建网格 ---
    print(f"\n开始重建索引为 {args.shape_idx} 的形状...")
    reconstructed_mesh = get_mesh_from_sdf(model, args.shape_idx, DEVICE, resolution=args.resolution)

    if reconstructed_mesh is None:
        print("网格重建失败，退出程序。")
        return

    # --- 4. 准备原始点云用于对比 ---
    ground_truth_data = dataset[args.shape_idx]
    ground_truth_points = ground_truth_data['pos'].numpy()
    
    # 创建Open3D点云对象
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(ground_truth_points)
    gt_pcd.paint_uniform_color([1, 0, 0]) # 设置为红色

    # --- 5. 可视化 ---
    print("\n可视化对比: 左侧为重建网格 (蓝色), 右侧为原始点云 (红色)")
    
    # 将原始点云向右平移，以便于并排比较
    gt_pcd.translate((2.2, 0, 0))
    
    # 给重建的网格上色
    reconstructed_mesh.paint_uniform_color([0.5, 0.7, 1.0]) # 淡蓝色

    o3d.visualization.draw_geometries(
        [reconstructed_mesh, gt_pcd],
        window_name=f"重建对比 (索引: {args.shape_idx})",
        width=1600,
        height=800
    )

if __name__ == '__main__':
    main()