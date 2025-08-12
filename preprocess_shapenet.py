"""
preprocess_shapenet.py
将 train_data_dir / val_data_dir 中的网格文件（.ply/.obj）采样为固定点数（默认 1024），
并保存为 preprocessed_train/*.pt, preprocessed_val/*.pt 供训练脚本高速读取。

依赖: trimesh, pytorch3d
"""

import os
import glob
import torch
import argparse
from tqdm import tqdm

try:
    import trimesh
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes
except Exception as e:
    raise RuntimeError("preprocessing requires trimesh and pytorch3d installed") from e

def process_folder(src_folder, dst_folder, num_points=1024, overwrite=False):
    os.makedirs(dst_folder, exist_ok=True)
    mesh_files = sorted(glob.glob(os.path.join(src_folder, "*.ply")) + glob.glob(os.path.join(src_folder, "*.obj")))
    if len(mesh_files) == 0:
        print(f"No meshes in {src_folder}")
        return
    for mesh_path in tqdm(mesh_files, desc=f"Processing {src_folder}"):
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        out_path = os.path.join(dst_folder, name + ".pt")
        if os.path.exists(out_path) and not overwrite:
            continue
        try:
            tm = trimesh.load(mesh_path, force='mesh')
            if tm.is_empty:
                print(f"Empty mesh: {mesh_path}, skipping")
                continue
            verts = torch.tensor(tm.vertices, dtype=torch.float32).unsqueeze(0)
            faces = torch.tensor(tm.faces, dtype=torch.int64).unsqueeze(0)
            mesh = Meshes(verts=verts, faces=faces)
            points = sample_points_from_meshes(mesh, num_points)[0]  # (num_points,3)
            # optionally center & normalize scale so all clouds are comparable
            centroid = points.mean(dim=0, keepdim=True)
            points = points - centroid
            scale = torch.sqrt((points ** 2).sum(dim=1).max())
            points = points / (scale + 1e-9)
            torch.save(points, out_path)
        except Exception as e:
            print(f"Failed to process {mesh_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, default="train_data_dir")
    parser.add_argument("--val_folder", type=str, default="val_data_dir")
    parser.add_argument("--out_train", type=str, default="preprocessed_train")
    parser.add_argument("--out_val", type=str, default="preprocessed_val")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    process_folder(args.train_folder, args.out_train, num_points=args.num_points, overwrite=args.overwrite)
    process_folder(args.val_folder, args.out_val, num_points=args.num_points, overwrite=args.overwrite)
    print("Preprocessing finished.")
