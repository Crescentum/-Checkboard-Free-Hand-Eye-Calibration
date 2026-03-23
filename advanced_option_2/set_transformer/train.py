import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.gelu(self.fc_o(O)) # 改为 GELU
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        
    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


def matrix_to_6d(matrix):
    return matrix[..., :3, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]
    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return x + self.net(x)

class AXBSolver(nn.Module):
    def __init__(self, dim_in=18, hidden_dim=256):
        super().__init__()
        
        self.enc1 = ISAB(dim_in, hidden_dim, num_heads=8, num_inds=32, ln=True)
        self.enc2 = ISAB(hidden_dim, hidden_dim, num_heads=8, num_inds=32, ln=True)
        self.enc3 = ISAB(hidden_dim, hidden_dim, num_heads=8, num_inds=32, ln=True)
        
        self.pma = PMA(hidden_dim, num_heads=8, num_seeds=1, ln=True)
        
        self.rot_head = nn.Sequential(
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 6)
        )
        
        self.trans_head = nn.Sequential(
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        e = self.enc1(x)
        e = self.enc2(e) + e  
        e = self.enc3(e) + e  
        
        z = self.pma(e).squeeze(1)
        
        rot_6d = self.rot_head(z)
        trans = self.trans_head(z)
        
        return torch.cat([rot_6d, trans], dim=-1)


class AXBDataset(Dataset):
    def __init__(self, num_samples=5000, set_size=30, 
                 base_noise_A_rot=0.0, base_noise_A_trans=0.0,
                 base_noise_B_rot=0.0, base_noise_B_trans=0.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.base_noise_A_rot_rad = base_noise_A_rot * math.pi / 180.0
        self.base_noise_A_trans = base_noise_A_trans / 100.0 
        self.base_noise_B_rot_rad = base_noise_B_rot * math.pi / 180.0
        self.base_noise_B_trans = base_noise_B_trans / 100.0 
        self.curr_noise_B_rot = self.base_noise_B_rot_rad
        self.curr_noise_B_trans = self.base_noise_B_trans

    def update_noise_scale(self, scale_factor):
        self.curr_noise_B_rot = self.base_noise_B_rot_rad * scale_factor
        self.curr_noise_B_trans = self.base_noise_B_trans * scale_factor

    def _get_random_transform(self):
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.linalg.det(q) < 0: q *= -1
        t = torch.randn(3, 1) 
        T = torch.eye(4)
        T[:3, :3] = q
        T[:3, 3:] = t
        return T

    def _add_noise(self, T, rot_std_rad, trans_std):
        if rot_std_rad <= 1e-9 and trans_std <= 1e-9: return T
        axis = F.normalize(torch.randn(3), dim=0)
        angle = torch.randn(1) * rot_std_rad
        K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R_delta = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        T_delta = torch.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3:] = torch.randn(3, 1) * trans_std
        return T @ T_delta

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        X_gt = self._get_random_transform()
        inv_X = torch.inverse(X_gt)
        A_list, B_list = [], []
        A_gt_list, B_gt_list = [], []
        for _ in range(self.set_size):
            A_gt = self._get_random_transform()
            B_gt = inv_X @ A_gt @ X_gt
            A_gt_list.append(A_gt)
            B_gt_list.append(B_gt)
            A_list.append(self._add_noise(A_gt, self.base_noise_A_rot_rad, self.base_noise_A_trans))
            B_list.append(self._add_noise(B_gt, self.curr_noise_B_rot, self.curr_noise_B_trans))
        A_mats, B_mats = torch.stack(A_list), torch.stack(B_list)
        A_gt_mats = torch.stack(A_gt_list)
        B_gt_mats = torch.stack(B_gt_list)
        feat_A = torch.cat([matrix_to_6d(A_mats), A_mats[:, :3, 3]], dim=-1)
        feat_B = torch.cat([matrix_to_6d(B_mats), B_mats[:, :3, 3]], dim=-1)
        return torch.cat([feat_A, feat_B], dim=-1), \
               torch.cat([matrix_to_6d(X_gt.unsqueeze(0)).squeeze(), X_gt[:3, 3]], dim=-1), \
               A_mats, B_mats, A_gt_mats, B_gt_mats

def log_cosh_loss(pred, target):
    diff = pred - target
    return (diff + F.softplus(-2. * diff) - math.log(2.)).mean()

def compute_pose_errors(pred_vec, target_vec):
    with torch.no_grad():
        t_err = torch.norm(pred_vec[:, 6:] - target_vec[:, 6:], dim=1).mean() * 100.0
        R_pred = compute_rotation_matrix_from_ortho6d(pred_vec[:, :6])
        R_gt = compute_rotation_matrix_from_ortho6d(target_vec[:, :6])
        R_diff = torch.bmm(R_pred, R_gt.transpose(1, 2))
        trace = R_diff[:,0,0]+R_diff[:,1,1]+R_diff[:,2,2]
        r_err = torch.rad2deg(torch.acos(torch.clamp((trace - 1.0)/2.0, -1.0+1e-6, 1.0-1e-6))).mean()
    return t_err.item(), r_err.item()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir = "checkpoints_2"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = AXBDataset(50000, 20, 0.1, 0.05, 1.5, 1.5)
    val_dataset = AXBDataset(1000, 20, 0.1, 0.05, 1.5, 1.5)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    model = AXBSolver(hidden_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    
    def lr_lambda(epoch):
        if epoch < 10: return (epoch + 1) / 10
        return max(0.01, 0.5 * (1 + math.cos(math.pi * min(1, (epoch - 10) / 190))))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')

    for epoch in range(200):
        noise_scale = 1.0 if epoch < 40 else max(0.01, 1.0 - (epoch - 40) / 140)
        train_dataset.update_noise_scale(noise_scale)
        alg_lambda = max(0.5, 5.0 * (0.97 ** epoch)) 

        model.train()
        train_stats = {"loss": 0, "loss_rot":0, "loss_trans":0, "t_err": 0, "r_err": 0}
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for x_in, y_gt, A, B, A_gt, B_gt in pbar:
            x_in, y_gt, A, B, A_gt, B_gt = x_in.to(device), y_gt.to(device), A.to(device), B.to(device), A_gt.to(device), B_gt.to(device)   
            
            optimizer.zero_grad()
            pred = model(x_in)
            
            loss_rot = log_cosh_loss(pred[:, :6], y_gt[:, :6])
            loss_trans = log_cosh_loss(pred[:, 6:], y_gt[:, 6:])
            
            R_x = compute_rotation_matrix_from_ortho6d(pred[:, :6])
            T_x = torch.eye(4, device=device).unsqueeze(0).repeat(pred.size(0),1,1)
            T_x[:,:3,:3], T_x[:,:3,3:] = R_x, pred[:,6:].unsqueeze(-1)
            loss_alg = log_cosh_loss(A_gt @ T_x.unsqueeze(1), T_x.unsqueeze(1) @ B_gt)
            
            total_loss = loss_rot * 5.0 + loss_trans * 2.0 + alg_lambda * loss_alg

            train_stats["loss_rot"] += loss_rot.item()
            train_stats["loss_trans"] += loss_trans.item()
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            
            t_e, r_e = compute_pose_errors(pred, y_gt)
            train_stats["loss"] += total_loss.item()
            train_stats["t_err"] += t_e; train_stats["r_err"] += r_e

        scheduler.step()

        model.eval()
        val_stats = {"loss": 0, "t_err": 0, "r_err": 0}
        with torch.no_grad():
            for x_in, y_gt, A, B, A_gt, B_gt in val_loader:
                x_in, y_gt = x_in.to(device), y_gt.to(device)
                pred = model(x_in)
                t_e, r_e = compute_pose_errors(pred, y_gt)
                val_stats["loss"] += log_cosh_loss(pred, y_gt).item()
                val_stats["t_err"] += t_e; val_stats["r_err"] += r_e

        n_t, n_v = len(train_loader), len(val_loader)
        print(f"Epoch {epoch+1:03d} | Noise: {noise_scale:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Train -> Loss: {train_stats['loss']/n_t:.4f} | Loss_rot: {train_stats['loss_rot']/n_t:.4f} | Loss_trans: {train_stats['loss_trans']/n_t:.4f} | T: {train_stats['t_err']/n_t:.2f}cm | R: {train_stats['r_err']/n_t:.2f}°")
        print(f"  Val   -> Loss: {val_stats['loss']/n_v:.4f} | T: {val_stats['t_err']/n_v:.2f}cm | R: {val_stats['r_err']/n_v:.2f}°")

        if (val_stats['loss']/n_v) < best_val_loss:
            best_val_loss = val_stats['loss']/n_v
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

if __name__ == "__main__":
    main()