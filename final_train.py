# train_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import os
import glob
from torch.utils.tensorboard import SummaryWriter

# --- 1. 核心依賴導入 ---
try:
    from torch_geometric.nn import knn_interpolate, fps, knn_graph
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_dense_batch
    print("PyTorch Geometric 庫已找到。")
except ImportError as e:
    print(f"致命錯誤: PyTorch Geometric 未正確安裝。錯誤: {e}")
    exit()

# torch_scatter 是 PyG 的核心依賴，對於高效的獎勵函數至關重要
try:
    from torch_scatter import scatter_mean
    print("torch_scatter 庫已找到。")
except ImportError:
    print("致命錯誤: 'torch_scatter' 未安裝。請運行: pip install torch-scatter")
    exit()

# 可選依賴，用於評估，在訓練中不使用
try:
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Alpha_shape_3 import Alpha_shape_3, Mode
    print("CGAL-pybind 已找到 (用於評估)。")
except ImportError:
    print("警告: 未找到 cgal-pybind。Alpha-shape 重建功能將不可用。")


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_directory = os.path.join(project_root, 'checkpoints_v6')
os.makedirs(save_directory, exist_ok=True)


# --- 2. 升級版PointNet++ U-Net模型 (PointNetAlphaUNet) ---
class PointNetAlphaUNet(torch.nn.Module):
    def __init__(self, k_interp=3):
        super().__init__()
        self.k = k_interp

        def create_mlp(in_channels, out_channels_list, last_relu=True):
            layers = []
            for out_channels in out_channels_list:
                layers.append(nn.Linear(in_channels, out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if not last_relu:
                return nn.Sequential(*layers[:-1])
            return nn.Sequential(*layers)

        # --- 編碼器 (下採樣) ---
        # 處理原始點 (坐標+法線)
        self.sa1_mlp = create_mlp(3 + 3, [64, 128])
        # 處理上一層的特徵
        self.sa2_mlp = create_mlp(128 + 3, [128, 256])
        # 瓶頸層
        self.sa3_mlp = create_mlp(256 + 3, [256, 512])

        # --- 解碼器 (上採樣與特徵融合) ---
        self.fp2_mlp = create_mlp(512 + 256, [256, 256])  # 上採樣(l2->l1) + 跳躍連接(l1)
        self.fp1_mlp = create_mlp(256 + 128, [256, 128])   # 上採樣(l1->l0) + 跳躍連接(l0)
        
        # 輸出前的最終特徵融合層
        self.fp0_mlp = create_mlp(128 + 3 + 3, [128, 128]) # 融合特徵 + 原始坐標 + 原始法線

        # --- 輸出頭 ---
        self.head_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        self.softplus = nn.Softplus()

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch

        # --- 編碼器 ---
        # Level 0 (原始點)
        l0_pos, l0_x, l0_batch = pos, x, batch
        l0_features = self.sa1_mlp(torch.cat([l0_pos, l0_x], dim=1))

        # Level 1
        l1_idx = fps(l0_pos, l0_batch, ratio=0.25)
        l1_pos, l1_batch = l0_pos[l1_idx], l0_batch[l1_idx]
        l1_agg_features = knn_interpolate(l0_features, l0_pos, l1_pos, l0_batch, l1_batch, k=16)
        l1_features = self.sa2_mlp(torch.cat([l1_agg_features, l1_pos], dim=1))

        # Level 2 (瓶頸)
        l2_idx = fps(l1_pos, l1_batch, ratio=0.25)
        l2_pos, l2_batch = l1_pos[l2_idx], l1_batch[l2_idx]
        l2_agg_features = knn_interpolate(l1_features, l1_pos, l2_pos, l1_batch, l2_batch, k=16)
        l2_features = self.sa3_mlp(torch.cat([l2_agg_features, l2_pos], dim=1))
        
        # --- 解碼器 ---
        # 從 Level 2 上採樣到 Level 1
        l1_up_features = knn_interpolate(l2_features, l2_pos, l1_pos, l2_batch, l1_batch, k=self.k)
        l1_fp_features = self.fp2_mlp(torch.cat([l1_up_features, l1_features], dim=1))
        
        # 從 Level 1 上採樣到 Level 0
        l0_up_features = knn_interpolate(l1_fp_features, l1_pos, l0_pos, l1_batch, l0_batch, k=self.k)
        l0_fp_features = self.fp1_mlp(torch.cat([l0_up_features, l0_features], dim=1))

        # 最終特徵融合
        final_features = self.fp0_mlp(torch.cat([l0_fp_features, l0_pos, l0_x], dim=1))

        # --- 輸出頭 ---
        alpha_mean = self.head_mlp(final_features)
        alpha_mean_dense, _ = to_dense_batch(alpha_mean, batch)
        alpha_mean_dense = alpha_mean_dense.permute(0, 2, 1)

        MIN_ALPHA = 0.01 
        alpha_mean_activated = self.softplus(alpha_mean_dense) + MIN_ALPHA
        
        # 返回一個正態分佈策略
        policy = Normal(alpha_mean_activated, torch.ones_like(alpha_mean_activated))
        return policy

# --- 3. 高效數據加載器 ---
class PyGShapeNetDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, split='train'):
        super().__init__(root_dir)

        # self.processed_dir = os.path.join(root_dir, "processed_points_with_normals")
        # self.num_points = num_points
        # self.paths = glob.glob(os.path.join(self.processed_dir, "**/*.pt"), recursive=True)
        
        # if not self.paths:
        #     raise ValueError(f"在 '{self.processed_dir}' 中未找到預處理的 '.pt' 文件。\n"
        #                      "請先運行 'preprocess_shapenet.py' 腳本。")
        # print(f"為 '{split}' 找到了 {len(self.paths)} 個預處理好的模型。")

        # self.root 会由父类自动设置为 root_dir
        self.processed_data_folder = os.path.join(self.root, "processed_points_with_normals")
        self.num_points = num_points
        self.paths = glob.glob(os.path.join(self.processed_data_folder, "**/*.pt"), recursive=True)
        
        if not self.paths:
            raise ValueError(f"在 '{self.processed_data_folder}' 中未找到预处理的 '.pt' 文件。\n"
                             "请再次确认：\n"
                             "1. 您已经成功运行了 'preprocess_shapenet.py' 脚本。\n"
                             "2. 当前脚本中的 SHAPENET_PATH 路径设置正确无误。")
        print(f"为 '{split}' 分割找到了 {len(self.paths)} 个预处理好的模型。")
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        try:
            data_dict = torch.load(self.paths[idx], weights_only=False)
            return Data(pos=data_dict['pos'], x=data_dict['x'])
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

# --- 4. 高效GPU版獎勵函數 (V6) ---
def calculate_reward_v6(alphas, original_points, batch_index, k, weights, device):
    with torch.no_grad():
        edge_index = knn_graph(original_points, k=k, batch=batch_index, loop=False)
        row, col = edge_index
        distances = torch.norm(original_points[row] - original_points[col], p=2, dim=-1)
        local_geom_feature = scatter_mean(distances, row, dim=0, dim_size=original_points.size(0))

        dense_alphas, mask = to_dense_batch(alphas, batch_index)
        dense_geom_feature, _ = to_dense_batch(local_geom_feature, batch_index)
        
        def normalize(tensor, mask):
            tensor_masked = torch.where(mask, tensor, torch.tensor(0.0, device=device))
            t_min = tensor_masked.min(dim=1, keepdim=True).values
            t_max = tensor_masked.max(dim=1, keepdim=True).values
            return (tensor - t_min) / (t_max - t_min + 1e-8)

        norm_alphas = normalize(dense_alphas, mask)
        norm_geom_feature = normalize(dense_geom_feature, mask)

        mse = F.mse_loss(norm_alphas, norm_geom_feature, reduction='none')
        reward_correlation = - (mse * mask).sum(dim=1) / mask.sum(dim=1)
        reward_diversity = torch.std(dense_alphas[mask].view(dense_alphas.size(0), -1), dim=1)
        penalty_magnitude = -torch.log(1 + (dense_alphas * mask).sum(dim=1) / mask.sum(dim=1))
        
        total_reward = (weights['w_correlation'] * reward_correlation +
                        weights['w_diversity'] * reward_diversity +
                        weights['w_magnitude'] * penalty_magnitude)
        return total_reward

# --- 5. 訓練主函數 ---
def main():
    # --- 超參數配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHAPENET_PATH = "/root/autodl-tmp/dataset/ShapeNetCore.v2/ShapeNetCore.v2"
    NUM_POINTS = 2048
    BATCH_SIZE = 256  # 可以適當調大，因為數據加載很快
    LEARNING_RATE = 3e-4
    EPOCHS = 100
    REWARD_BASELINE_DECAY = 0.95
    K_NEIGHBORS_FOR_REWARD = 16
    EXPLORATION_DECAY = 0.96 # 探索率衰減

    REWARD_WEIGHTS = {
        'w_correlation': 2.5,
        'w_diversity': 0.5,
        'w_magnitude': 0.2,
    }

    if not os.path.isdir(SHAPENET_PATH) or "/path/to/your/" in SHAPENET_PATH:
        print("="*80 + f"\n致命錯誤: 請在腳本中更新 SHAPENET_PATH 變量。\n" + "="*80); exit()

    writer = SummaryWriter('runs/pointnet_alpha_v6_experiment')
    
    model = PointNetAlphaUNet().to(DEVICE)
    dataset = PyGShapeNetDataset(root_dir=SHAPENET_PATH, num_points=NUM_POINTS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    START_EPOCH = 0
    # 可選的檢查點加載
    # CHECKPOINT_PATH = "path/to/your/checkpoint.pth"
    # if os.path.exists(CHECKPOINT_PATH):
    #     model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    #     print(f"✅ 從檢查點恢復訓練: {CHECKPOINT_PATH}")

    reward_baseline = 0.0
    global_step = 0
    
    print(f"在 {DEVICE} 上開始訓練, Batch Size: {BATCH_SIZE}, 獎勵權重: {REWARD_WEIGHTS}")
    
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        current_std = max(0.20 * (EXPLORATION_DECAY**epoch), 0.01)

        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)
            
            policy = model(batch_data)
            policy.scale = torch.ones_like(policy.loc) * current_std 
            sampled_alphas_dense = policy.sample()

            _, mask = to_dense_batch(batch_data.pos, batch_data.batch)
            sampled_alphas_sparse = sampled_alphas_dense.permute(0, 2, 1)[mask].squeeze()

            rewards_tensor = calculate_reward_v6(sampled_alphas_sparse, 
                                                 batch_data.pos, 
                                                 batch_data.batch,
                                                 K_NEIGHBORS_FOR_REWARD, 
                                                 REWARD_WEIGHTS, 
                                                 DEVICE)
            
            avg_reward = rewards_tensor.mean().item()
            advantage = rewards_tensor - reward_baseline
            
            log_probs_dense = policy.log_prob(sampled_alphas_dense)
            log_probs_sum_per_sample = (log_probs_dense * mask.unsqueeze(1)).sum(dim=[1, 2])
            
            loss = - (log_probs_sum_per_sample * advantage).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            reward_baseline = REWARD_BASELINE_DECAY * reward_baseline + (1 - REWARD_BASELINE_DECAY) * avg_reward
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_reward=f"{avg_reward:.4f}", baseline=f"{reward_baseline:.4f}")

            # --- Tensorboard 日誌 ---
            if global_step % 20 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Reward/average_reward', avg_reward, global_step)
                writer.add_scalar('Reward/baseline', reward_baseline, global_step)
                writer.add_scalar('Hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Hyperparameters/exploration_std', current_std, global_step)
                writer.add_histogram('Alphas/sampled_distribution', sampled_alphas_sparse, global_step)
                writer.add_histogram('Reward/advantage_distribution', advantage, global_step)

            global_step += 1
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            save_file_name = f"pointnet_alpha_v6_epoch_{epoch+1}.pth"
            save_path = os.path.join(save_directory, save_file_name)
            torch.save(model.state_dict(), save_path)
            print(f"\n✅ 模型已保存至 {save_path}")

    writer.close()
    print("訓練完成。TensorBoard 日誌已保存。")

if __name__ == '__main__':
    main()