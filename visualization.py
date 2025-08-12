# ==============================================================================
#      Adaptive Alpha Shape Inference and Full Visualization Script
#
#  这个脚本做了以下事情:
#  1. 加载您训练好的模型和一个测试用的点云数据。
#  2. 对点云进行推理，得到逐点的alpha预测值。
#  3. 使用CGAL和预测的alpha值重建出三维网格。
#  4. [核心功能] 使用Open3D将以下三个对象并排显示，以便对比：
#     - 原始点云 (输入)
#     - Alpha值热力图点云 (模型"思考"过程)
#     - 重建后的网格 (输出)
#
#  作者: Gemini (Google AI)
#  日期: August 11, 2025
# ==============================================================================

import torch
import numpy as np
import os
import glob
import argparse
import random

# --- 核心依赖导入 ---
try: import open3d as o3d; print("Open3D library found.")
except ImportError: print("FATAL ERROR: 'open3d' not installed. Run: pip install open3d"); exit()
try: import trimesh; print("Trimesh library found.")
except ImportError: print("FATAL ERROR: 'trimesh' not installed. Run: pip install trimesh"); exit()
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader # 虽然只处理单个，但保持一致性
    print("PyTorch Geometric found.")
except ImportError as e: print(f"FATAL ERROR: PyG not installed correctly. Error: {e}"); exit()
try:
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    print("PyTorch3D found.")
except ImportError as e: print(f"FATAL ERROR: PyTorch3D not found. Error: {e}"); exit()
try:
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Alpha_shape_3 import Alpha_shape_3, Mode
    print("CGAL-pybind found.")
except ImportError: print("FATAL ERROR: cgal-pybind is required for visualization. Please install it."); exit()

# --- 导入模型定义 ---
# 确保这个脚本可以找到你的训练脚本中定义的模型类
# 如果它们在同一个目录下，这行代码就可以工作
try:
    from train_adaptive_alpha_with_pyg import PyG_PointNet2_Alpha_Predictor
except ImportError:
    print("FATAL ERROR: Could not import model definition. Ensure 'train_adaptive_alpha_with_pyg.py' is in the same directory.")
    exit()


def visualize_inference(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 1. 加载训练好的模型 ---
    if not os.path.exists(args.model_path):
        print(f"FATAL ERROR: Model file not found at {args.model_path}"); exit()
    print(f"Loading trained model from: {args.model_path}")
    model = PyG_PointNet2_Alpha_Predictor().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval() # 非常重要！设置为评估模式

    # --- 2. 加载一个测试用的点云 ---
    print(f"Scanning for models in: {args.shapenet_path}")
    paths = glob.glob(os.path.join(args.shapenet_path, "**/model_normalized.ply"), recursive=True)
    if not paths: raise ValueError("No .ply files found in the specified ShapeNet path.")

    if args.sample_index is not None:
        if 0 <= args.sample_index < len(paths): test_file_path = paths[args.sample_index]
        else: print(f"Error: Index {args.sample_index} is out of bounds."); return
    else:
        test_file_path = random.choice(paths)
        print("No index provided, choosing a random model.")
    
    print(f"Visualizing sample: {test_file_path}")
    
    mesh_trimesh = trimesh.load(test_file_path)
    verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh_trimesh.faces, dtype=torch.long)
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    points_tensor = sample_points_from_meshes(pytorch3d_mesh, num_samples=args.num_points).squeeze(0)
    data = Data(pos=points_tensor).to(DEVICE)
    points_np = points_tensor.cpu().numpy()

    # --- 3. 执行推理 ---
    print("Running inference to predict alpha values...")
    with torch.no_grad():
        policy = model(data)
        predicted_alphas = policy.loc.squeeze().cpu() # [N]

    print(f"Inference complete. Predicted alpha range: [{predicted_alphas.min():.4f}, {predicted_alphas.max():.4f}]")

    # --- 4. 使用CGAL进行重建 ---
    print("Reconstructing mesh using CGAL with predicted alphas...")
    median_alpha = torch.median(predicted_alphas).item()
    if median_alpha <= 1e-9: median_alpha = 1e-9
    
    points_cgal = [Point_3(p[0], p[1], p[2]) for p in points_np]
    alpha_shape = Alpha_shape_3(points_cgal, median_alpha, Mode.GENERAL)
    
    reconstructed_mesh_o3d = None
    try:
        verts_list, faces_list = alpha_shape.get_surface_mesh()
        if verts_list and faces_list:
            reconstructed_verts = np.array(verts_list)
            reconstructed_faces = np.array(faces_list)
            reconstructed_mesh_o3d = o3d.geometry.TriangleMesh()
            reconstructed_mesh_o3d.vertices = o3d.utility.Vector3dVector(reconstructed_verts)
            reconstructed_mesh_o3d.triangles = o3d.utility.Vector3iVector(reconstructed_faces)
            reconstructed_mesh_o3d.compute_vertex_normals()
            reconstructed_mesh_o3d.paint_uniform_color([0.1, 0.9, 0.1]) # 绿色
        else:
            print("WARNING: CGAL could not reconstruct a surface. Only point clouds will be shown.")
    except Exception as e:
        print(f"WARNING: An error occurred during CGAL reconstruction: {e}")

    # --- 5. 使用Open3D进行可视化 ---
    print("Launching Open3D visualizer...")

    # a) 原始点云 (灰色)
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points_np)
    pcd_original.paint_uniform_color([0.7, 0.7, 0.7])

    # b) Alpha值热力图点云
    alphas_normalized = (predicted_alphas - predicted_alphas.min()) / (predicted_alphas.max() - predicted_alphas.min() + 1e-9)
    # 使用Matplotlib的viridis色彩映射，它对色盲更友好
    try:
        import matplotlib.pyplot as plt
        colors = plt.get_cmap('viridis')(alphas_normalized.numpy())[:, :3]
    except ImportError:
        print("Matplotlib not found, using a simpler heatmap.")
        colors = torch.tensor(o3d.utility.Vector3dVector(np.array(o3d.geometry.TriangleMesh.get_vertex_colors_from_heatmap(alphas_normalized))))
    
    pcd_alpha_heatmap = o3d.geometry.PointCloud()
    pcd_alpha_heatmap.points = o3d.utility.Vector3dVector(points_np)
    pcd_alpha_heatmap.colors = o3d.utility.Vector3dVector(colors)

    # c) 将三个对象放置到不同的位置以便观察
    # 原始点云在中间
    # Alpha热力图在左边
    pcd_alpha_heatmap.translate([-1.2, 0, 0])
    
    geometries_to_draw = [pcd_original, pcd_alpha_heatmap]
    
    if reconstructed_mesh_o3d is not None:
        # 重建网格在右边
        reconstructed_mesh_o3d.translate([1.2, 0, 0])
        geometries_to_draw.append(reconstructed_mesh_o3d)
        
    # d) 启动可视化窗口
    o3d.visualization.draw_geometries(
        geometries_to_draw,
        window_name="Input (Center) | Alpha Heatmap (Left) | Reconstruction (Right)",
        width=1200,
        height=800,
        mesh_show_wireframe=True
    )
    print("Visualization window closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the input, alpha heatmap, and reconstruction result of a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--shapenet_path', type=str, required=True, help="Path to the root of the ShapeNetCore.v2 dataset.")
    parser.add_argument('--num_points', type=int, default=2048, help="Number of points to sample from the mesh (should match training).")
    parser.add_argument('--sample_index', type=int, default=None, help="Index of the sample to visualize. If not provided, a random one is chosen.")
    args = parser.parse_args()
    
    if args.shapenet_path is None or args.model_path is None:
        parser.print_help()
    else:
        visualize_inference(args)