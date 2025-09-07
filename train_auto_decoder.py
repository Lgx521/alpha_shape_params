# train_auto_decoder_v18.py
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
training_version = 'v19'
'''
V18 - 第一阶段: 自解码器预训练
目标: 训练一个强大的SDF解码器，并为数据集中每个形状优化一个专属的隐编码。

V19 - 增加L2 Norm
'''

# --- 核心依赖导入 ---
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_directory = os.path.join(project_root, f'checkpoints_{training_version}')
os.makedirs(save_directory, exist_ok=True)

# --- 1. 全新模型: SDF自解码器 (SDF Auto-Decoder) ---
class SDFAutoDecoder(nn.Module):
    def __init__(self, num_shapes, latent_dim=256, mlp_hidden_dim=256):
        super().__init__()
        
        # [核心] 可学习的隐编码嵌入层，作为“密码本”
        self.latent_codes = nn.Embedding(num_shapes, latent_dim)
        # 初始化隐编码，使其均值为0，标准差较小，有助于稳定训练初期
        torch.nn.init.normal_(self.latent_codes.weight.data, 0.0, 0.01)

        # SDF解码器 (与之前相同)
        self.sdf_head = nn.Sequential(
            nn.Linear(latent_dim + 3, mlp_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, shape_indices, query_points):
        """
        Args:
            shape_indices (Tensor): 形状的索引, shape (B,).
            query_points (Tensor): 查询点, shape (B, num_queries, 3).
        """
        B, num_queries, _ = query_points.shape
        
        # 从“密码本”中查找对应的隐编码
        latent_z = self.latent_codes(shape_indices) # (B, latent_dim)
        
        latent_z_expanded = latent_z.unsqueeze(1).expand(-1, num_queries, -1)
        sdf_head_input = torch.cat([latent_z_expanded, query_points], dim=-1)
        
        predicted_sdf = self.sdf_head(sdf_head_input)
        return predicted_sdf

# --- 2. 修改后的数据加载器 (返回样本索引) ---
class PyGShapeNetDatasetWithIdx(Dataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.processed_data_folder = os.path.join(self.root, "processed_points_with_normals")
        self.paths = sorted(glob.glob(os.path.join(self.processed_data_folder, "**/*.pt"), recursive=True))
        if not self.paths: raise ValueError(f"在 '{self.processed_data_folder}' 中未找到预处理的 '.pt' 文件。")
        print(f"为自解码器找到了 {len(self.paths)} 个模型。")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            data_dict = torch.load(self.paths[idx], weights_only=False)
            points, normals = data_dict['pos'], data_dict['x']
            center = points.mean(dim=0)
            points_centered = points - center
            scale = (points_centered.norm(p=2, dim=1)).max()
            points_normalized = points_centered / scale
            
            # [核心修改] 返回数据和该数据的索引
            return Data(pos=points_normalized, x=normals), idx
        except Exception:
            # 简单处理，实际项目中可以记录失败的索引
            return self.__getitem__((idx + 1) % len(self))

# --- 3. 损失函数 (自监督SDF损失 - 保持不变) ---
# def calculate_sdf_loss_v12_final(model, shape_indices, query_points, weights):
#     query_points.requires_grad = True
#     predicted_sdf = model(shape_indices, query_points)
    
#     B, num_total_queries, _ = query_points.shape
#     num_surface = num_total_queries // 2
    
#     surface_pred_sdf = predicted_sdf[:, :num_surface]
    
#     loss_fidelity = F.l1_loss(surface_pred_sdf, torch.zeros_like(surface_pred_sdf))

#     grad_outputs = torch.ones_like(predicted_sdf)
#     gradients = torch.autograd.grad(
#         outputs=predicted_sdf, inputs=query_points,
#         grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True
#     )[0]
#     loss_eikonal = F.mse_loss(gradients.norm(dim=-1), torch.ones_like(gradients.norm(dim=-1)))

#     total_loss = weights['w_fidelity'] * loss_fidelity + weights['w_eikonal'] * loss_eikonal
#     return total_loss

def calculate_sdf_loss_v12_final(model, shape_indices, query_points, weights):
    query_points.requires_grad = True
    predicted_sdf = model(shape_indices, query_points)
    
    B, num_total_queries, _ = query_points.shape
    num_surface = num_total_queries // 2
    
    # 1. Fidelity Loss (L1 norm on surface points)
    surface_pred_sdf = predicted_sdf[:, :num_surface]
    loss_fidelity = F.l1_loss(surface_pred_sdf, torch.zeros_like(surface_pred_sdf))

    # 2. Eikonal Loss (L2 norm of gradients)
    grad_outputs = torch.ones_like(predicted_sdf)
    gradients = torch.autograd.grad(
        outputs=predicted_sdf, inputs=query_points,
        grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    loss_eikonal = F.mse_loss(gradients.norm(dim=-1), torch.ones_like(gradients.norm(dim=-1)))

    # --- 3. L2 Regularization Loss (New Addition) ---
    loss_l2 = 0.0
    for param in model.parameters():
        loss_l2 += torch.norm(param, p=2)**2
    
    # 4. Total Loss
    total_loss = (weights['w_fidelity'] * loss_fidelity + 
                  weights['w_eikonal'] * loss_eikonal +
                  weights['w_l2'] * loss_l2)
    
    return total_loss

# --- 4. 训练主函数 (第一阶段) ---
def main():
    # --- 超参数配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "./ShapeNetCore.v2/ShapeNetCore.v2"
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    EPOCHS = 100 # 自解码器预训练需要更多轮次来优化每个隐编码
    
    LOSS_WEIGHTS = { 'w_fidelity': 1.0, 'w_eikonal': 0.1, 'w_l2' : 0.8 }
    NUM_QUERY_POINTS_PER_SAMPLE = 4096

    writer = SummaryWriter(f'./runs/pointnet_alpha_{training_version}_experiment')
    
    dataset = PyGShapeNetDatasetWithIdx(root_dir=SHAPENET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    
    num_shapes = len(dataset)
    model = SDFAutoDecoder(num_shapes=num_shapes).to(DEVICE)
    
    # [核心] 优化器现在需要同时优化SDF解码器的权重和隐编码嵌入层的权重
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    global_step = 0
    print(f"在 {DEVICE} 上开始第一阶段预训练 (版本: {training_version})")
    
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for data, shape_indices in progress_bar:
            data = data.to(DEVICE)
            shape_indices = shape_indices.to(DEVICE)
            points_dense, mask = to_dense_batch(data.pos, data.batch)
            B = points_dense.shape[0]

            # --- 采样查询点 (鲁棒采样策略) ---
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
            loss = calculate_sdf_loss_v12_final(model, shape_indices, query_points, LOSS_WEIGHTS)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            # 保存整个模型，它同时包含了SDF解码器权重和训练好的隐编码“密码本”
            save_file_name = f"autodecoder_{training_version}_epoch_{epoch+1}.pth"
            save_path = os.path.join(save_directory, save_file_name)
            torch.save(model.state_dict(), save_path)
            print(f"\n✅ 自解码器模型已保存至 {save_path}")

    writer.close()
    print(f"第一阶段训练完成 (版本: {training_version})。")

if __name__ == '__main__':
    main()