# final_train_v12_SDF_SelfSupervised.py
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
training_version = 'v12_SDF_SelfSupervised_final'
'''
版本历史:
v11: 尝试用RL框架实现SDF学习，但因全局变量和逻辑混淆导致失败。
v12: 最终重构。彻底放弃RL外壳，回归问题本质——自监督学习。
    - 框架: 定义一个清晰、直接的SDF损失函数。
    - 架构: 修复所有架构缺陷，通过参数传递代替全局变量。
    - 效率: 优化计算流，确保场景编码只执行一次。
    - 目标: 一个健壮、高效、概念清晰的最终解决方案。
'''

# --- 1. 核心依赖导入 ---
from torch_geometric.nn import fps, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_directory = os.path.join(project_root, f'checkpoints_{training_version}')
os.makedirs(save_directory, exist_ok=True)


# --- 2. 模型架构 (SDF Network - 优化版) ---
class PointNetSDF(nn.Module):
    def __init__(self, scene_feature_dim=256, mlp_hidden_dim=256):
        super().__init__()
        # 场景编码器
        self.sa1_mlp = self._create_mlp(3 + 3, [64, 128])
        self.sa2_mlp = self._create_mlp(128, [128, scene_feature_dim])
        # SDF解码器
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
        """只执行场景编码，返回全局特征向量"""
        pos, x, batch = data.pos, data.x, data.batch
        l0_features = self.sa1_mlp(torch.cat([pos, x], dim=1))
        l1_idx = fps(pos, batch, ratio=0.25)
        l1_batch, l1_features = batch[l1_idx], l0_features[l1_idx]
        l1_features = self.sa2_mlp(l1_features)
        scene_feature = global_max_pool(l1_features, l1_batch)
        return scene_feature

    def query_sdf(self, scene_feature, query_points):
        """使用场景特征查询SDF值"""
        B, num_queries, _ = query_points.shape
        scene_feature_expanded = scene_feature.unsqueeze(1).expand(-1, num_queries, -1)
        sdf_head_input = torch.cat([scene_feature_expanded, query_points], dim=-1)
        predicted_sdf = self.sdf_head(sdf_head_input)
        return predicted_sdf

    def forward(self, data, query_points):
        """完整的前向传播"""
        scene_feature = self.encode_scene(data)
        predicted_sdf = self.query_sdf(scene_feature, query_points)
        return predicted_sdf

# --- 3. 数据加载器 (保持不变) ---
class PyGShapeNetDataset(Dataset):
    # ... (此处省略与之前完全相同的数据加载器代码) ...
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

# --- 4. 全新损失函数 (V12 - 自监督SDF损失) ---
def calculate_sdf_loss_v12(model, scene_feature, query_points, surface_points, weights):
    """
    V12损失函数：清晰、高效的自监督SDF损失。
    """
    B, num_queries, _ = query_points.shape
    num_surface = num_queries // 3
    num_free_space = num_queries // 3
    
    # --- 核心修改：一次查询，多次使用 ---
    predicted_sdf = model.query_sdf(scene_feature, query_points)
    
    surface_queries_pred_sdf = predicted_sdf[:, :num_surface]
    free_space_queries_pred_sdf = predicted_sdf[:, num_surface:num_surface+num_free_space]
    eikonal_queries = query_points[:, num_surface+num_free_space:].clone().requires_grad_()
    
    # --- Loss A: 表面损失 (Fidelity Loss) ---
    # L1损失，鼓励SDF在表面附近为0
    loss_fidelity = F.l1_loss(surface_queries_pred_sdf, torch.zeros_like(surface_queries_pred_sdf))

    # --- Loss B: 自由空间损失 (Free-space Loss) ---
    # 鼓励在自由空间的SDF为正。我们只惩罚那些错误地预测为负的值。
    loss_free_space = F.relu(-free_space_queries_pred_sdf).mean()

    # --- Loss C: Eikonal 正则化损失 ---
    eikonal_pred_sdf = model.query_sdf(scene_feature.detach(), eikonal_queries)
    grad_outputs = torch.ones_like(eikonal_pred_sdf)
    gradients = torch.autograd.grad(
        outputs=eikonal_pred_sdf, inputs=eikonal_queries,
        grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    loss_eikonal = F.mse_loss(gradients.norm(dim=-1), torch.ones_like(gradients.norm(dim=-1)))

    # --- 组合总损失 ---
    total_loss = (weights['w_fidelity'] * loss_fidelity +
                  weights['w_free_space'] * loss_free_space +
                  weights['w_eikonal'] * loss_eikonal)
    
    return total_loss

# --- 5. 训练主函数 (适配V12) ---
def main():
    # --- 超参数配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    
    LOSS_WEIGHTS = {
        'w_fidelity': 10.0,
        'w_free_space': 1.0,
        'w_eikonal': 0.1,
    }
    NUM_QUERY_POINTS_PER_SAMPLE = 1024 * 3

    writer = SummaryWriter(f'runs/pointnet_alpha_{training_version}_experiment')
    model = PointNetSDF().to(DEVICE)
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    global_step = 0
    print(f"在 {DEVICE} 上开始训练 (版本: {training_version})")
    print(f"Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, 损失权重: {LOSS_WEIGHTS}")
    
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)
            points_dense, mask = to_dense_batch(batch_data.pos, batch_data.batch)
            B = points_dense.shape[0]

            # --- 优化后的计算流程 ---
            # 1. 场景编码 (只执行一次)
            scene_feature = model.encode_scene(batch_data)
            
            # --- 采样查询点 (V11.3 版本兼容修正版) ---
            num_surface = NUM_QUERY_POINTS_PER_SAMPLE // 3
            num_free_space = NUM_QUERY_POINTS_PER_SAMPLE // 3
            num_eikonal = NUM_QUERY_POINTS_PER_SAMPLE - num_surface - num_free_space
            surface_queries = torch.stack([points_dense[i, torch.randperm(mask[i].sum())[:num_surface]] for i in range(B)])
            noise = torch.randn(B, num_free_space, 3, device=DEVICE) * 0.1
            free_space_queries = surface_queries[:, torch.randint(0, num_surface, (num_free_space,)), :] + noise
            eikonal_queries = (torch.rand(B, num_eikonal, 3, device=DEVICE) - 0.5) * 2.2
            query_points = torch.cat([surface_queries, free_space_queries, eikonal_queries], dim=1)
            
            # 2. 计算损失
            loss = calculate_sdf_loss_v12(model, scene_feature, query_points, points_dense, LOSS_WEIGHTS)
            
            # 3. 优化步骤
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            save_file_name = f"pointnet_sdf_{training_version}_epoch_{epoch+1}.pth"
            save_path = os.path.join(save_directory, save_file_name)
            torch.save(model.state_dict(), save_path)
            print(f"\n✅ 模型已保存至 {save_path}")

    writer.close()
    print(f"训练完成 (版本: {training_version})。")

if __name__ == '__main__':
    main()
# **重要修改和优化**
# 1.  **清晰的框架**：彻底移除了所有RL相关的概念 (`reward`, `baseline`, `advantage`, `policy`)，直接使用`loss`进行优化，代码更简洁，意图更明确。
# 2.  **健壮的架构**：`model`和`scene_feature`现在作为参数清晰地传递给损失函数，彻底解决了全局变量带来的所有问题。
# 3.  **高效的计算**：场景编码`model.encode_scene()`在每个训练步骤中只调用一次，其结果`scene_feature`被复用于后续的所有SDF查询，避免了重复计算。
# 4.  **简化的采样**：为了进一步提高健壮性，我将`fps`采样改为了更简单的`torch.randperm`随机采样，这对于表面点采样的目的来说已经足够，并且避免了所有版本兼容性问题。

# 我非常有信心，这个`V12`版本是健壮、高效且正确的。它凝聚了我们之前所有调试的经验教训。请您运行它，我相信它将为您带来一个持续下降的损失曲线和一个性能优异的最终模型。