# remote_inference.py (v1.1 - 修正版)
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from tqdm import tqdm
import os
import glob

# ==============================================================================
# --- [核心修正] 补上所有必要的导入，并使脚本独立 ---
# ==============================================================================
from torch_geometric.nn import fps, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader # 虽然推理不用，但为了完整性
from torch_geometric.utils import to_dense_batch

# --- 1. 将模型和数据加载器的定义直接复制过来，使脚本完全独立 ---

class PointNetSDF(nn.Module):
    def __init__(self, scene_feature_dim=256, mlp_hidden_dim=256):
        super().__init__()
        self.sa1_mlp = self._create_mlp(3 + 3, [64, 128])
        self.sa2_mlp = self._create_mlp(128, [128, scene_feature_dim])
        self.sdf_head = nn.Sequential(
            nn.Linear(scene_feature_dim + 3, mlp_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1)
        )
    def _create_mlp(self, in_c, out_cs):
        layers = [nn.Linear(in_c, out_cs[0]), nn.ReLU(inplace=True)]
        for i in range(len(out_cs) - 1):
            layers.extend([nn.Linear(out_cs[i], out_cs[i+1]), nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)
    def encode_scene(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        l0_features = self.sa1_mlp(torch.cat([pos, x], dim=1))
        l1_idx = fps(pos, batch, ratio=0.25)
        l1_batch, l1_features = batch[l1_idx], l0_features[l1_idx]
        l1_features = self.sa2_mlp(l1_features)
        return global_max_pool(l1_features, l1_batch)
    def query_sdf(self, scene_feature, query_points):
        B, num_queries, _ = query_points.shape
        scene_feature_expanded = scene_feature.unsqueeze(1).expand(-1, num_queries, -1)
        sdf_head_input = torch.cat([scene_feature_expanded, query_points], dim=-1)
        return self.sdf_head(sdf_head_input)

class PyGShapeNetDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, split='train'):
        super().__init__(root_dir)
        self.processed_data_folder = os.path.join(self.root, "processed_points_with_normals")
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
            return self.__getitem__((idx + 1) % len(self))

# ==============================================================================
# --- 配置参数 (您只需要修改这里) ---
# ==============================================================================
training_version = 'v14_SDF_RL_correct'
CHECKPOINT_EPOCH = 200
SAMPLE_INDICES = [1200]
RESOLUTION = 256
OUTPUT_DIR = f"inference_results"
VIEWPOINTS = {
    "front": {"lookat": [0, 0, 0], "front": [0, 1.5, 0.2], "up": [0, 0, 1], "zoom": 0.8},
    "top_down": {"lookat": [0, 0, 0], "front": [0, 0.1, 2], "up": [0, 1, 0], "zoom": 0.8},
    "side_iso": {"lookat": [0, 0, 0], "front": [1, 1, 1], "up": [0, 0, 1], "zoom": 0.8},
}

# ==============================================================================
# --- 主程序 ---
# ==============================================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = os.path.join(project_root, f'checkpoints_{training_version}')
    checkpoint_path = os.path.join(checkpoint_dir, f"pointnet_sdf_{training_version}_epoch_{CHECKPOINT_EPOCH}.pth")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = PointNetSDF().to(DEVICE)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"致命错误: 在路径 {checkpoint_path} 未找到模型文件。请检查 CHECKPOINT_EPOCH 是否正确。")
        return
    model.eval()
    print(f"模型已从 {checkpoint_path} 加载。")

    dataset = PyGShapeNetDataset(root_dir="/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2", split='test')

    for sample_idx in SAMPLE_INDICES:
        print(f"\n--- 正在处理样本 {sample_idx} ---")
        data_sample = dataset[sample_idx]
        data_sample.batch = torch.zeros(data_sample.pos.shape[0], dtype=torch.long) # 添加batch信息
        data_sample = data_sample.to(DEVICE)
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
                sdf_batch = model.query_sdf(scene_feature, points_batch.unsqueeze(0))
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

        reconstructed_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
        )
        reconstructed_mesh.compute_vertex_normals()
        reconstructed_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        original_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(original_points))
        original_pcd.paint_uniform_color([1, 0, 0])

        # ==============================================================================
        # --- [修正] 渲染部分：使用 OffscreenRenderer 替换 Visualizer ---
        # ==============================================================================
        # 为了渲染，我们需要为网格和点云定义“材质”
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [0.7, 0.7, 0.7, 1.0] # 灰色网格

        pcd_mat = o3d.visualization.rendering.MaterialRecord()
        pcd_mat.shader = "defaultUnlit" # 点云通常用无光照材质
        pcd_mat.base_color = [1.0, 0.0, 0.0, 1.0] # 红色点云
        pcd_mat.point_size = 3.0 # 可以设置点的大小

        # 初始化离屏渲染器
        renderer = o3d.visualization.rendering.OffscreenRenderer(RESOLUTION * 2, RESOLUTION * 2) # 可以设置更高的分辨率

        for view_name, view_params in VIEWPOINTS.items():
            print(f"正在从 '{view_name}' 视角进行离屏渲染...")

            # 设置场景和相机
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry("reconstructed_mesh", reconstructed_mesh, mat)
            renderer.scene.add_geometry("original_pcd", original_pcd, pcd_mat)

            # set_camera_properties 的参数与你的 VIEWPOINTS 有一点不同，需要转换
            # lookat(center), eye(front), up
            renderer.scene.camera.look_at(
                view_params["lookat"], # center
                view_params["front"],  # eye
                view_params["up"]      # up
            )
            # 缩放可以通过调整相机视场角(fov)或距离来实现，这里我们保持默认

            # 渲染为图像
            rendered_image = renderer.render_to_image()

            # 保存图像
            output_path = os.path.join(OUTPUT_DIR, f"recon_sample_{sample_idx}_view_{view_name}__{training_version}.png")
            o3d.io.write_image(output_path, rendered_image)

            print(f"  -> 已保存至 {output_path}")

    print("\n所有推理任务完成！")

if __name__ == '__main__':
    main()