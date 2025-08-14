# inference.py
import torch
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from tqdm import tqdm
import os

# [重要] 确保这里的模型定义和数据加载器与您的训练脚本完全一致
from final_train_v14_SDF_RL_correct import PointNetSDF, PyGShapeNetDataset

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
# 选择一个您训练好的模型权重文件
CHECKPOINT_PATH = "/root/autodl-tmp/code/checkpoints_v14_SDF_RL_correct/pointnet_sdf_v14_SDF_RL_correct_epoch_30.pth"
# 选择一个测试样本的索引
SAMPLE_IDX = 100 
# 重建的分辨率 (64对于快速预览足够了, 128或256可以获得更精细的结果)
RESOLUTION = 128

def main():
    # 1. 加载模型
    model = PointNetSDF().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print(f"模型已从 {CHECKPOINT_PATH} 加载。")

    # 2. 加载一个测试数据样本
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH, split='test')
    data_sample = dataset[SAMPLE_IDX].to(DEVICE)
    
    # 原始点云 (用于对比可视化)
    original_points = data_sample.pos.cpu().numpy()

    # 3. 创建查询网格
    grid_min, grid_max = -1.1, 1.1
    coords = np.linspace(grid_min, grid_max, RESOLUTION)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    query_grid = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    query_grid_torch = torch.tensor(query_grid, dtype=torch.float32, device=DEVICE)

    # 4. 批量查询SDF值
    print(f"正在查询 {RESOLUTION**3} 个点的SDF值...")
    sdf_values = []
    with torch.no_grad():
        # 编码场景一次
        scene_feature = model.encode_scene(data_sample)
        
        # 将查询点分批以避免显存溢出
        for points_batch in tqdm(torch.split(query_grid_torch, 1024 * 64, dim=0)):
            # query_sdf需要 (B, N, 3) 的输入
            sdf_batch = model.query_sdf(scene_feature.unsqueeze(0), points_batch.unsqueeze(0))
            sdf_values.append(sdf_batch.squeeze(0).cpu())
    
    sdf_values = torch.cat(sdf_values, dim=0).numpy().reshape(RESOLUTION, RESOLUTION, RESOLUTION)

    # 5. 使用 Marching Cubes 提取零等值面
    print("正在使用 Marching Cubes 提取网格...")
    try:
        verts, faces, normals, _ = marching_cubes(sdf_values, level=0.0)
        # 将顶点坐标从[0, RESOLUTION]范围映射回[-1.1, 1.1]范围
        verts = verts * (grid_max - grid_min) / (RESOLUTION - 1) + grid_min
    except ValueError:
        print("错误：在SDF场中未能找到level=0的表面。模型可能没有学好。")
        return

    if len(verts) == 0:
        print("警告：提取出的网格没有顶点。")
        return

    # 6. 可视化
    print("正在准备可视化...")
    reconstructed_mesh = o3d.geometry.TriangleMesh()
    reconstructed_mesh.vertices = o3d.utility.Vector3dVector(verts)
    reconstructed_mesh.triangles = o3d.utility.Vector3iVector(faces)
    reconstructed_mesh.compute_vertex_normals() # 计算法线以获得更好的着色效果
    reconstructed_mesh.paint_uniform_color([0.7, 0.7, 0.7]) # 灰色

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    original_pcd.paint_uniform_color([1, 0, 0]) # 红色

    # 在一个窗口中同时显示原始点云和重建的网格
    o3d.visualization.draw_geometries([reconstructed_mesh, original_pcd])
    
    # (可选) 保存网格
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"recon_sample_{SAMPLE_IDX}.ply")
    o3d.io.write_triangle_mesh(output_path, reconstructed_mesh)
    print(f"重建的网格已保存至 {output_path}")

if __name__ == '__main__':
    # 您可能需要先安装 open3d: python -m pip install open3d
    # 您可能需要先安装 scikit-image: python -m pip install scikit-image
    main()