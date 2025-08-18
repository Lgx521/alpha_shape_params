# final_train_v12_SelfSupervised_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import glob
from torch.utils.tensorboard import SummaryWriter

# ==============================================================================
# --- 版本与实验配置 ---
# ==============================================================================
training_version = 'v12_1'
'''
V12: 最终方案。在经历了RL框架的种种“策略崩溃”问题后，我们回归到
     最直接、最强大、最适合此任务的自监督学习范式。
    - 损失函数: 采用逐点的、密集的SDF损失，彻底杜绝“作弊”可能。
    - 目标: 训练一个真正能学习复杂几何的、健壮的最终模型。
'''

# --- (核心依赖、模型架构、数据加载器与之前V16版本完全相同，此处省略以保持简洁) ---
from torch_geometric.nn import fps, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_directory = os.path.join(project_root, f'checkpoints_{training_version}')
os.makedirs(save_directory, exist_ok=True)
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

# --- 4. 损失函数 (V12 - 自监督SDF损失) ---
def calculate_sdf_loss_v12_final(model, scene_feature, query_points, weights):
    query_points.requires_grad = True # Eikonal loss needs gradients
    predicted_sdf = model.query_sdf(scene_feature, query_points)
    
    B, num_total_queries, _ = query_points.shape
    num_surface = num_total_queries // 2
    
    surface_pred_sdf = predicted_sdf[:, :num_surface]
    near_surface_queries = query_points[:, num_surface:]
    
    # Loss A: 表面损失
    loss_fidelity = F.l1_loss(surface_pred_sdf, torch.zeros_like(surface_pred_sdf))

    # Loss B: Eikonal 正则化损失 (作用于所有点以保证全局平滑)
    grad_outputs = torch.ones_like(predicted_sdf)
    gradients = torch.autograd.grad(
        outputs=predicted_sdf, inputs=query_points,
        grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    loss_eikonal = F.mse_loss(gradients.norm(dim=-1), torch.ones_like(gradients.norm(dim=-1)))

    total_loss = weights['w_fidelity'] * loss_fidelity + weights['w_eikonal'] * loss_eikonal
    return total_loss

# --- 5. 训练主函数 (V12 - 自监督) ---
def main():
    # --- 超参数配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
    BATCH_SIZE = 256 # 使用更小的Batch Size，因为梯度计算更复杂
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    
    LOSS_WEIGHTS = { 'w_fidelity': 1.0, 'w_eikonal': 0.1 }
    NUM_QUERY_POINTS_PER_SAMPLE = 4096

    writer = SummaryWriter(f'runs/pointnet_alpha_{training_version}_experiment')
    model = PointNetSDF().to(DEVICE)
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    global_step = 0
    print(f"在 {DEVICE} 上开始训练 (版本: {training_version})")
    
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)
            points_dense, mask = to_dense_batch(batch_data.pos, batch_data.batch)
            B = points_dense.shape[0]

            scene_feature = model.encode_scene(batch_data)
            
            # --- 采样查询点 (V15/V16 鲁棒采样策略) ---
            num_surface = NUM_QUERY_POINTS_PER_SAMPLE // 2
            num_near_surface = NUM_QUERY_POINTS_PER_SAMPLE - num_surface
            surface_queries_list = []
            for i in range(B):
                sample_points = points_dense[i, mask[i]]
                indices = torch.randperm(sample_points.shape[0], device=DEVICE)[:num_surface]
                sampled_surface_points = sample_points[indices]
                if sampled_surface_points.shape[0] < num_surface:
                    padding = torch.zeros(num_surface - sampled_surface_points.shape[0], 3, device=DEVICE)
                    sampled_surface_points = torch.cat([sampled_surface_points, padding], dim=0)
                surface_queries_list.append(sampled_surface_points)
            surface_queries = torch.stack(surface_queries_list)
            
            perturbations = torch.randn(B, num_near_surface, 3, device=DEVICE) * 0.05
            base_points_for_perturb = surface_queries[:, torch.randint(0, num_surface, (num_near_surface,)), :]
            near_surface_queries = base_points_for_perturb + perturbations
            query_points = torch.cat([surface_queries, near_surface_queries], dim=1)
            
            # --- 自监督学习框架 ---
            loss = calculate_sdf_loss_v12_final(model, scene_feature.detach(), query_points, LOSS_WEIGHTS)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            save_file_name = f"pointnet_sdf_{training_version}_epoch_{epoch+1}.pth"
            save_path = os.path.join(save_directory, save_file_name)
            torch.save(model.state_dict(), save_path)
            print(f"\n✅ 模型已保存至 {save_path}")

    writer.close()
    print(f"训练完成 (版本: {training_version})。")

if __name__ == '__main__':
    main()