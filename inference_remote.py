# remote_inference.py
import torch
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from tqdm import tqdm
import os

# [重要] 确保这里的模型定义和数据加载器与您的训练脚本完全一致
# 我们直接从您的训练脚本中导入它们
from final_train_v14_SDF_RL_correct import PointNetSDF, PyGShapeNetDataset

# ==============================================================================
# --- 配置参数 (您只需要修改这里) ---
# ==============================================================================
# 训练版本，用于构建正确的路径
training_version = 'v14_SDF_RL_correct'

# 您想要评估的模型是在第几个epoch保存的？
CHECKPOINT_EPOCH = 30 

# 您想要测试的数据集样本的索引列表
# 您可以放入多个索引，脚本会自动为每个样本生成一组图片
SAMPLE_INDICES = [100, 250, 500, 1024, 2048] 

# 重建的分辨率 (64或128是速度和质量的良好平衡点)
RESOLUTION = 128

# 图片输出目录
OUTPUT_DIR = f"inference_results_{training_version}"

# 预设的相机视角
# 您可以根据需要添加、删除或修改这些视角
VIEWPOINTS = {
    "front": {
        "lookat": [0, 0, 0],  # 相机看向的点 (世界中心)
        "front": [0, 1.5, 0.2], # 相机的位置向量 (从哪个方向看)
        "up": [0, 0, 1],     # 相机的“上”方向 (Z轴朝上)
        "zoom": 0.8
    },
    "top_down": {
        "lookat": [0, 0, 0],
        "front": [0, 0.1, 2],   # 从正上方往下看
        "up": [0, 1, 0],     # Y轴是“上”
        "zoom": 0.8
    },
    "side_iso": {
        "lookat": [0, 0, 0],
        "front": [1, 1, 1], # 从斜对角45度角看
        "up": [0, 0, 1],
        "zoom": 0.8
    },
}

# ==============================================================================
# --- 主程序 ---
# ==============================================================================
def main():
    # --- 1. 设置路径和加载模型 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = os.path.join(project_root, f'checkpoints_{training_version}')
    checkpoint_path = os.path.join(checkpoint_dir, f"pointnet_sdf_{training_version}_epoch_{CHECKPOINT_EPOCH}.pth")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = PointNetSDF().to(DEVICE)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"致命错误: 在路径 {checkpoint_path} 未找到模型文件。请检查 CHECKPOINT_EPOCH 是否正确。")
        return
        
    model.eval()
    print(f"模型已从 {checkpoint_path} 加载。")

    # --- 2. 加载数据集 ---
    dataset = PyGShapeNetDataset(root_dir="/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2", split='test')

    # --- 3. 循环处理每个要测试的样本 ---
    for sample_idx in SAMPLE_INDICES:
        print(f"\n--- 正在处理样本 {sample_idx} ---")
        
        # --- 3.1. 获取数据和重建网格 (逻辑与之前类似) ---
        data_sample = dataset[sample_idx].to(DEVICE)
        original_points = data_sample.pos.cpu().numpy()

        grid_min, grid_max = -1.1, 1.1
        coords = np.linspace(grid_min, grid_max, RESOLUTION)
        x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
        query_grid = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        query_grid_torch = torch.tensor(query_grid, dtype=torch.float32, device=DEVICE)

        sdf_values = []
        with torch.no_grad():
            scene_feature = model.encode_scene(data_sample)
            print(f"正在查询 {RESOLUTION**3} 个点的SDF值...")
            for points_batch in tqdm(torch.split(query_grid_torch, 1024 * 64, dim=0)):
                sdf_batch = model.query_sdf(scene_feature.unsqueeze(0), points_batch.unsqueeze(0))
                sdf_values.append(sdf_batch.squeeze(0).cpu())
        sdf_values = torch.cat(sdf_values, dim=0).numpy().reshape(RESOLUTION, RESOLUTION, RESOLUTION)

        print("正在使用 Marching Cubes 提取网格...")
        try:
            verts, faces, _, _ = marching_cubes(sdf_values, level=0.0)
            verts = verts * (grid_max - grid_min) / (RESOLUTION - 1) + grid_min
        except (ValueError, RuntimeError):
            print(f"警告: 样本 {sample_idx} 未能成功提取网格，跳过。")
            continue

        if len(verts) == 0:
            print(f"警告: 样本 {sample_idx} 提取的网格为空，跳过。")
            continue

        # --- 3.2. 准备Open3D几何体 ---
        reconstructed_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces)
        )
        reconstructed_mesh.compute_vertex_normals()
        reconstructed_mesh.paint_uniform_color([0.7, 0.7, 0.7]) # 灰色网格

        original_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(original_points))
        original_pcd.paint_uniform_color([1, 0, 0]) # 红色点云

        # --- 3.3. 核心：无头渲染并保存图片 ---
        for view_name, view_params in VIEWPOINTS.items():
            print(f"正在从 '{view_name}' 视角渲染...")
            
            # 创建一个不可见的渲染器窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False) 
            
            # 添加几何体
            vis.add_geometry(reconstructed_mesh)
            vis.add_geometry(original_pcd)
            
            # 获取视图控制器并设置相机参数
            ctr = vis.get_view_control()
            ctr.set_lookat(view_params["lookat"])
            ctr.set_front(view_params["front"])
            ctr.set_up(view_params["up"])
            ctr.set_zoom(view_params["zoom"])
            
            # 定义输出图片路径
            output_path = os.path.join(OUTPUT_DIR, f"recon_sample_{sample_idx}_view_{view_name}.png")
            
            # 捕获屏幕截图并保存
            vis.capture_screen_image(output_path, do_render=True)
            
            # 销毁窗口，释放资源
            vis.destroy_window()
            
            print(f"  -> 已保存至 {output_path}")

    print("\n所有推理任务完成！")

if __name__ == '__main__':
    # 您需要先安装 open3d 和 scikit-image:
    # python -m pip install open3d scikit-image
    main()