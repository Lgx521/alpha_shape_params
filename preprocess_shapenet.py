# preprocess_shapenet.py
import torch
import trimesh
import os
import glob
from tqdm import tqdm
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

# --- 配置您的路徑 ---
SHAPENET_ROOT = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2" # 您的ShapeNet數據集根目錄
OUTPUT_DIR = os.path.join(SHAPENET_ROOT, "processed_points_with_normals") # 預處理數據的存放位置
NUM_POINTS = 2048 # 採樣點數，與訓練時保持一致

def process_and_save(path, output_dir):
    try:
        relative_path = os.path.relpath(os.path.dirname(path), SHAPENET_ROOT)
        save_folder = os.path.join(output_dir, relative_path)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "points_and_normals.pt")

        if os.path.exists(save_path):
            return

        mesh = trimesh.load(path, process=False)
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces') or len(mesh.vertices) == 0:
            return

        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])

        points, normals = sample_points_from_meshes(
            pytorch3d_mesh,
            num_samples=NUM_POINTS,
            return_normals=True
        )

        # 保存為包含點和法線的字典
        data_dict = {'pos': points.squeeze(0), 'x': normals.squeeze(0)}
        torch.save(data_dict, save_path)
    except Exception as e:
        print(f"處理 {path} 時出錯: {e}")

if __name__ == '__main__':
    if not os.path.isdir(SHAPENET_ROOT) or "path/to" in SHAPENET_ROOT:
        print("致命錯誤: 請先在 preprocess_shapenet.py 中更新 SHAPENET_ROOT 路徑！")
        exit()
        
    print(f"開始預處理數據，源目錄: {SHAPENET_ROOT}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    paths = glob.glob(os.path.join(SHAPENET_ROOT, "**/model_normalized.ply"), recursive=True)
    if not paths:
        raise ValueError(f"在 {SHAPENET_ROOT} 中未找到 'model_normalized.ply' 文件。")
        
    print(f"找到 {len(paths)} 個模型，開始處理...")
    
    for path in tqdm(paths, desc="Preprocessing models"):
        process_and_save(path, OUTPUT_DIR)
        
    print("預處理完成！")