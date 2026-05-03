import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def _permute_prediction_1d(prediction: torch.Tensor):
    """Generate all permutations along the last dimension for a 1D prediction.

    prediction: (M,)
    return: (M!, M)
    """
    from itertools import permutations

    device = prediction.device
    torch_perm_list = []
    for p in permutations(range(prediction.shape[0]), prediction.shape[0]):
        idx = torch.tensor(list(p), dtype=torch.int64, device=device)
        torch_perm_list.append(prediction.index_select(0, idx))
    return torch.stack(torch_perm_list, dim=0)


class PeriodicPermutationDoALoss(nn.Module):
    """Permutation-invariant periodic DoA loss (torch) consistent with src/criterions.py.

    - Inputs are expected in **radians**.
    - Error is wrapped with period pi (i.e., theta ≡ theta + pi), then measured by RMSE or MSE.
    - For each sample, all permutations of predictions are evaluated and the minimal loss is used.

    Shapes:
        doa_pred: (B, M)
        doa_gt:   (B, M)
    """

    def __init__(self, mode: str = "rmse"):
        super().__init__()
        mode = str(mode).lower().strip()
        if mode not in {"rmse", "mse"}:
            raise ValueError("mode must be 'rmse' or 'mse'")
        self.mode = mode

    def forward(self, doa_pred: torch.Tensor, doa_gt: torch.Tensor) -> torch.Tensor:
        if doa_pred.dim() != 2 or doa_gt.dim() != 2:
            raise ValueError(f"Expect doa_pred/doa_gt of shape (B,M); got {tuple(doa_pred.shape)} / {tuple(doa_gt.shape)}")
        if doa_pred.shape != doa_gt.shape:
            raise ValueError(f"Shape mismatch: {tuple(doa_pred.shape)} vs {tuple(doa_gt.shape)}")

        B, M = doa_pred.shape
        losses = []
        for b in range(B):
            pred_b = doa_pred[b]
            gt_b = doa_gt[b]
            pred_perm = _permute_prediction_1d(pred_b)  # (M!, M)
            # periodic wrap with period pi -> [-pi/2, pi/2]
            err = (((pred_perm - gt_b) + (math.pi / 2.0)) % math.pi) - (math.pi / 2.0)  # (M!, M)
            if self.mode == "rmse":
                val = torch.linalg.norm(err, dim=-1) / math.sqrt(M)
            else:
                val = (torch.linalg.norm(err, dim=-1) ** 2) / float(M)
            losses.append(torch.min(val))
        # Return per-sample mean over the batch (NOT sum).
        # This matches the scale of other losses (which generally use mean) and avoids
        # confusing logging that depends on batch_size.
        return torch.mean(torch.stack(losses, dim=0))


class floss(nn.Module):
    """
        lambda1: U_dir约束系数
        lambda2: Vh_dir约束系数
        lambda3: U_mse约束系数
        lambda4: V_mse约束系数
        lambda5: S_mse误差系数
        lambda6: recon_loss误差系数
    """
    def __init__(
        self,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=0.1,
        lambda4=0.1,
        lambda5=0.1,
        lambda6=0.1,
        R=1,
        lambda_doa: float = 0.0,
        doa_mode: str = "rmse",
        dir_mode: str = "vector",
        lambda_fft_recon: float = 0.0,
        fft_recon_mode: str = "mag_l1",
        lambda_phase_recon: float = 0.0,
        # --- NEW: coherent/rank-deficient friendly dir-loss options ---
        # When sources are coherent, GT covariance can be rank-deficient (effective rank=1),
        # but training may still use r>1. In that case a full subspace alignment loss will
        # over-constrain the arbitrary 2nd basis vector and hurt convergence.
        dir_effective_rank: int | str = "full",
        dir_rank_ratio_th: float = 0.2,
        dir_repulsion: float = 0.05,
    ):
        super(floss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.lambda6 = lambda6
        self.lambda_doa = float(lambda_doa)
        self.lambda_phase = 0.0
        self.doa_criterion = PeriodicPermutationDoALoss(mode=doa_mode) if self.lambda_doa > 0 else None
        self.lambda_phase_recon = float(lambda_phase_recon)
        self.dir_mode = str(dir_mode).lower().strip()
        if self.dir_mode not in {"vector", "subspace"}:
            raise ValueError("dir_mode must be 'vector' or 'subspace'")
        self.rank = R  # 假设奇异值的秩为32

        # dir-loss effective-rank control
        # - "full": keep original behavior
        # - 1: always use rank-1 containment (recommended for coherent sources)
        # - "auto": decide per-batch using S_targets ratio (s2/s1 < dir_rank_ratio_th)
        if isinstance(dir_effective_rank, str):
            der = dir_effective_rank.lower().strip()
            if der not in {"full", "auto"}:
                raise ValueError("dir_effective_rank must be 'full', 'auto', or an int like 1")
            self.dir_effective_rank = der
        elif isinstance(dir_effective_rank, int):
            if dir_effective_rank < 1:
                raise ValueError("dir_effective_rank int must be >= 1")
            self.dir_effective_rank = int(dir_effective_rank)
        else:
            raise ValueError("dir_effective_rank must be 'full', 'auto', or an int like 1")
        self.dir_rank_ratio_th = float(dir_rank_ratio_th)
        self.dir_repulsion = float(dir_repulsion)

        # FFT reconstruction supervision (optional)
        self.lambda_fft_recon = float(lambda_fft_recon)
        self.fft_recon_mode = str(fft_recon_mode).lower().strip()
        if self.lambda_fft_recon > 0 and self.fft_recon_mode not in {"complex_l1", "mag_l1", "mag_mse"}:
            raise ValueError("fft_recon_mode must be one of: complex_l1, mag_l1, mag_mse")

    def forward(
        self,
        U_origional,
        S_origional,
        Vh_origional,
        U_tilde,
        S_tilde,
        V_tilde,
        epoch_id=100,
        doa_pred_rad: torch.Tensor = None,
        doa_gt_rad: torch.Tensor = None,
        fft_pred: torch.Tensor = None,
        fft_target: torch.Tensor = None,
        hd_pred: torch.Tensor = None,
        hd_target: torch.Tensor = None,
    ):

        # 处理复数表示
        U_tilde = self._to_complex(U_tilde)  # (batch_size, 64, 32)
        U_origional = self._to_complex(U_origional)
        Vh_dir = 0.0
        V_mse = 0.0
        S_mse = 0.0
        recon_loss = 0.0
        fft_recon_loss = 0.0
        phase_recon_loss = 0.0

        # [NEW] Explicit constraint on Dual-Domain Refinement Module outputs
        # Mean Squared Error on the refined covariance matrix (Hd_final) vs Clean Covariance (clean_c)
        if hd_pred is not None and hd_target is not None:
            hd_pred_c = self._to_complex(hd_pred)
            hd_target_c = self._to_complex(hd_target)
            recon_loss = torch.mean(torch.abs(hd_pred_c - hd_target_c)**2)

        # 若rank ！= U_original的最后一维，则截取前rank列进行计算
        if U_origional.dim() == 3 and U_origional.shape[-1] > self.rank:
            U_origional = U_origional[:, :, :self.rank]

        U_mse = self.phase_aligned_nmse(U_tilde, U_origional)
        if self.dir_mode == "subspace":
            U_dir = self._subspace_dir_loss_with_effective_rank(
                U_tilde,
                U_origional,
                S_origional=S_origional,
            )
        else:
            U_dir = self._phase_inv_dir_loss(U_tilde, U_origional, epoch_id)

        # 合并损失
        total_loss = (
            self.lambda1 * U_dir
            + self.lambda2 * Vh_dir
            + self.lambda3 * U_mse
            + self.lambda4 * V_mse
            + self.lambda5 * S_mse
            + self.lambda6 * recon_loss
        )

        # FFT spectrum reconstruction loss (optional)
        if self.lambda_fft_recon > 0:
            if fft_pred is None or fft_target is None:
                raise ValueError("lambda_fft_recon>0 but fft_pred/fft_target not provided")
            fft_pred_c = self._to_complex(fft_pred)
            fft_target_c = self._to_complex(fft_target)
            if self.fft_recon_mode == "complex_l1":
                fft_recon_loss = torch.mean(torch.abs(fft_pred_c - fft_target_c))
            elif self.fft_recon_mode == "mag_l1":
                fft_recon_loss = torch.mean(torch.abs(torch.abs(fft_pred_c) - torch.abs(fft_target_c)))
            elif self.fft_recon_mode == "mag_mse":
                fft_recon_loss = torch.mean((torch.abs(fft_pred_c) - torch.abs(fft_target_c)) ** 2)

            # NEW: phase-only loss
            eps = 1e-8
            pred_u = fft_pred_c / (torch.abs(fft_pred_c) + eps)
            tgt_u = fft_target_c / (torch.abs(fft_target_c) + eps)
            l_phase = torch.mean(torch.abs(pred_u - tgt_u))
            fft_recon_loss = fft_recon_loss + self.lambda_phase * l_phase

        total_loss = total_loss + self.lambda_fft_recon * fft_recon_loss

        doa_loss = 0.0
        if self.doa_criterion is not None:
            if doa_pred_rad is None or doa_gt_rad is None:
                raise ValueError("lambda_doa>0 but doa_pred_rad/doa_gt_rad not provided")
            doa_loss = self.doa_criterion(doa_pred_rad, doa_gt_rad)
            total_loss = total_loss + self.lambda_doa * doa_loss

        # Phase reconstruction loss (relative phase on Hd)
        if self.lambda_phase_recon > 0:
            if hd_pred is None or hd_target is None:
                raise ValueError("lambda_phase_recon > 0 but hd_pred/hd_target not provided")
            hd_pred_c = self._to_complex(hd_pred)
            hd_target_c = self._to_complex(hd_target)
            
            # Differential Phase Loss (Phase Gradient)
            # Compute phase difference between adjacent elements along the last dimension (columns)
            # D_ij = H_i,j+1 * conj(H_i,j). This captures relative phase shift (DoA info).
            pred_diff = hd_pred_c[..., 1:] * torch.conj(hd_pred_c[..., :-1])
            target_diff = hd_target_c[..., 1:] * torch.conj(hd_target_c[..., :-1])

            eps = 1e-8
            
            # Normalize to unit vectors (extract phase difference only)
            u_pred_diff = pred_diff / (torch.abs(pred_diff) + eps)
            u_target_diff = target_diff / (torch.abs(target_diff) + eps)
            
            # Weight by target magnitude (trust strong signal regions more)
            weight = torch.abs(target_diff)
            avg_weight = weight.mean().detach() + eps
            
            # Weighted mean distance on unit circle
            phase_recon_loss = torch.mean(torch.abs(u_pred_diff - u_target_diff) * weight) / avg_weight
            
            total_loss = total_loss + self.lambda_phase_recon * phase_recon_loss

        # Note: keep legacy return structure, but extend with fft_recon_loss at the end
        return total_loss, U_dir, Vh_dir, U_mse, V_mse, S_mse, recon_loss, doa_loss, fft_recon_loss, phase_recon_loss

    def _subspace_dir_loss_with_effective_rank(
        self,
        U_pred: torch.Tensor,
        U_gt: torch.Tensor,
        S_origional: torch.Tensor = None,
    ) -> torch.Tensor:
        """Handle coherent/rank-deficient GT when using subspace dir-loss.

        - full: original subspace_proj_loss (align full r-dim subspace)
        - 1: containment loss (require GT top-1 direction lies in span(U_pred))
        - auto: if s2/s1 < threshold -> use rank-1 containment else full
        """
        def _with_repulsion(base_loss: torch.Tensor) -> torch.Tensor:
            """Always attach orthogonality regularization in subspace mode when enabled."""
            if self.dir_repulsion > 0 and U_pred.dim() == 3 and U_pred.shape[-1] > 1:
                return base_loss + self.dir_repulsion * self.subspace_repulsion_loss(U_pred)
            return base_loss

        der = self.dir_effective_rank
        if der == "full":
            return _with_repulsion(self.subspace_proj_loss(U_pred, U_gt))

        if der == "auto":
            k = self._estimate_effective_rank_from_s(S_origional)
            if k <= 1:
                return _with_repulsion(self.subspace_containment_loss(U_pred, U_gt, k=1))
            return _with_repulsion(self.subspace_proj_loss(U_pred, U_gt))

        # int path
        k = int(der)
        r_pred = U_pred.shape[-1] if U_pred.dim() == 3 else 1
        if k == r_pred:
            # same as full rank; keep existing formula
            return _with_repulsion(self.subspace_proj_loss(U_pred, U_gt))
        loss = self.subspace_containment_loss(U_pred, U_gt, k=k)
        return _with_repulsion(loss)

    def _estimate_effective_rank_from_s(self, S: torch.Tensor) -> int:
        """Estimate effective signal-subspace rank from singular values.

        Heuristic for coherent sources: if s2/s1 is small, treat as rank-1.
        S can be real (B,r) or complex; we use abs.
        """
        if S is None:
            return 1
        if S.dim() == 1:
            # (r,) -> (1,r)
            S = S.unsqueeze(0)
        if S.dim() != 2:
            return 1
        if S.size(1) < 2:
            return 1
        s = torch.abs(S.to(torch.float32))
        s1 = s[:, 0]
        s2 = s[:, 1]
        ratio = s2 / (s1 + 1e-8)
        # batch average decision (stable)
        if torch.mean(ratio).item() < self.dir_rank_ratio_th:
            return 1
        return 2

    def subspace_containment_loss(self, U_pred: torch.Tensor, U_gt: torch.Tensor, k: int = 1, eps: float = 1e-8) -> torch.Tensor:
        """Containment loss: require top-k GT directions to lie in span(U_pred).

        Loss = mean_b mean_i (1 - ||U_pred^H u_gt_i||^2)
        where u_gt_i are columns of U_gt.

        This avoids over-constraining arbitrary extra GT basis vectors when GT is rank-deficient.
        """
        if U_pred.dim() == 2:
            U_pred = U_pred.unsqueeze(-1)
            U_gt = U_gt.unsqueeze(-1)
        if U_pred.dim() != 3 or U_gt.dim() != 3:
            raise ValueError(f"Expect U_pred/U_gt of shape (B,N,r) or (B,N); got {tuple(U_pred.shape)} / {tuple(U_gt.shape)}")
        if U_pred.shape[0] != U_gt.shape[0] or U_pred.shape[1] != U_gt.shape[1]:
            raise ValueError(f"Shape mismatch: {tuple(U_pred.shape)} vs {tuple(U_gt.shape)}")

        # normalize columns
        U_pred = U_pred / (torch.linalg.norm(U_pred, dim=1, keepdim=True) + eps)
        U_gt = U_gt / (torch.linalg.norm(U_gt, dim=1, keepdim=True) + eps)

        r_pred = U_pred.shape[-1]
        r_gt = U_gt.shape[-1]
        k = int(max(1, min(k, r_gt)))

        # (B, r_pred, k) = U_pred^H U_gt[:, :, :k]
        C = U_pred.conj().transpose(1, 2) @ U_gt[:, :, :k]
        # energy of projection for each gt vector: sum over r_pred
        proj_energy = (torch.abs(C) ** 2).sum(dim=1)  # (B, k)
        proj_energy = torch.clamp(proj_energy, 0.0, 1.0)
        return torch.mean(1.0 - proj_energy)

    def subspace_repulsion_loss(self, U_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Encourage different columns of U_pred to be orthogonal (avoid collapse).

        For each batch: G = U^H U (r x r). Penalize off-diagonal magnitude.
        """
        if U_pred.dim() != 3:
            return torch.tensor(0.0, device=U_pred.device)
        U_pred = U_pred / (torch.linalg.norm(U_pred, dim=1, keepdim=True) + eps)
        G = U_pred.conj().transpose(1, 2) @ U_pred  # (B,r,r)
        # zero diagonal
        diag = torch.diagonal(G, dim1=-2, dim2=-1)
        off = G - torch.diag_embed(diag)
        return torch.mean(torch.abs(off) ** 2)

    def _to_complex(self, tensor):
        """将实部虚部分离的张量转换为复数"""
        if torch.is_complex(tensor):
            return tensor
        if tensor.shape[-1] == 2:
            return torch.complex(tensor[..., 0], tensor[..., 1])
        return tensor

    def _compute_mse(self, U, U_origional):
        """计算MSE"""
        # 计算复数MSE，U与U_original本来就为复数形式，直接计算即可，但是需要考虑反相位的影响
        # 将U与U_origional
        U_mse = torch.sum(torch.abs((U - U_origional)) ** 2) / torch.sum(torch.abs(U_origional) ** 2)
        return U_mse

    def phase_aligned_nmse_origional(self, u_pred, u_gt):
        # supports (B,N) or (B,N,r)
        sum_dim = 1 if u_pred.dim() == 3 else -1
        inner = torch.sum(torch.conj(u_pred) * u_gt, dim=sum_dim)  # complex
        phi = torch.angle(inner)  # real
        # align prediction
        u_aligned = u_pred * torch.exp(-1j * phi).unsqueeze(sum_dim)
        num = torch.sum(torch.abs(u_aligned - u_gt) ** 2, dim=sum_dim)
        den = torch.sum(torch.abs(u_gt) ** 2, dim=sum_dim)
        return (num / den).mean()

    def phase_aligned_nmse(self, u_pred, u_gt):
        # supports (B,N) or (B,N,r)
        sum_dim = 1 if u_pred.dim() == 3 else -1
        inner = torch.sum(torch.conj(u_pred) * u_gt, dim=sum_dim)  # complex
        phase = inner / (torch.abs(inner) + 1e-8)  # e^{jφ}
        # align prediction
        u_aligned = u_pred * phase.unsqueeze(sum_dim)
        num = torch.sum(torch.abs(u_aligned - u_gt) ** 2, dim=sum_dim)
        den = torch.sum(torch.abs(u_gt) ** 2, dim=sum_dim)
        return (num / den).mean()

    def phase_aligned_nmse2(self, u_pred, u_gt):
        # inner product <u_pred, u_gt>
        sum_dim = 1 if u_pred.dim() == 3 else -1
        inner = torch.sum(torch.conj(u_pred) * u_gt, dim=sum_dim)
        phase_factor = inner / (torch.abs(inner) + 1e-8)  # e^{jφ}

        # 对齐方式1: e^{-jφ}
        u_aligned1 = u_pred * torch.conj(phase_factor).unsqueeze(sum_dim)
        mse1 = torch.sum(torch.abs(u_aligned1 - u_gt) ** 2, dim=sum_dim)

        # 对齐方式2: e^{-j(φ+π)} = -e^{-jφ}
        u_aligned2 = -u_aligned1
        mse2 = torch.sum(torch.abs(u_aligned2 - u_gt) ** 2, dim=sum_dim)

        # 取最小的MSE
        nmse = torch.minimum(mse1, mse2).mean()
        return nmse

    def _phase_inv_dir_loss(self, pred, target, epoch_id = 0):
        # pred/target: (B,N) or (B,N,r) complex
        if pred.dim() == 3:
            ip = torch.sum(torch.conj(target) * pred, dim=1)  # (B,r)
            num = torch.abs(ip)
            den = (torch.linalg.norm(pred, dim=1) * torch.linalg.norm(target, dim=1) + 1e-8)
            return torch.mean(1.0 - num / den)
        ip = torch.sum(torch.conj(target) * pred, dim=-1)  # <pred, target>
        num = torch.abs(ip)
        den = (torch.norm(pred, dim=-1) * torch.norm(target, dim=-1) + 1e-8)
        return torch.mean(1.0 - num / den)  # 1 - |cos(θ)|

    def subspace_proj_loss(self, U_pred: torch.Tensor, U_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Projection-based subspace loss (rotation/permutation invariant).

        Supports:
          - (B, N) or (B, N, r) complex
        For r=1 it reduces to the same family of direction losses.

        Loss: 1 - (||U_gt^H U_pred||_F^2 / r)
        When U columns are unit-norm, this is in [0,1].
        """
        if U_pred.dim() == 2:
            # (B,N) -> (B,N,1)
            U_pred = U_pred.unsqueeze(-1)
            U_gt = U_gt.unsqueeze(-1)
        if U_pred.dim() != 3 or U_gt.dim() != 3:
            raise ValueError(f"Expect U_pred/U_gt of shape (B,N,r) or (B,N); got {tuple(U_pred.shape)} / {tuple(U_gt.shape)}")
        if U_pred.shape != U_gt.shape:
            raise ValueError(f"Shape mismatch: {tuple(U_pred.shape)} vs {tuple(U_gt.shape)}")

        # normalize columns for stability (does not change subspace)
        U_pred = U_pred / (torch.linalg.norm(U_pred, dim=1, keepdim=True) + eps)
        U_gt = U_gt / (torch.linalg.norm(U_gt, dim=1, keepdim=True) + eps)

        r = U_pred.shape[-1]
        # (B,r,r) = U_gt^H U_pred
        C = U_gt.conj().transpose(1, 2) @ U_pred
        sim = (torch.abs(C) ** 2).sum(dim=(-2, -1)) / float(r)  # (B,)
        # numerical guard
        sim = torch.clamp(sim, 0.0, 1.0)
        return torch.mean(1.0 - sim)

    def rayleigh_loss(self, A, u):
        # A: (m,n), u: (n,) or (batch,n)
        Au = A.matmul(u.unsqueeze(-1)).squeeze(-1)  # shape (m,) or (batch,m)
        num = (Au.abs() ** 2).sum(-1)  # ||A u||^2
        return -num.mean()

    def compute_usvh(self,U, V, S):
        batch_size = U.size(0)

        # 提取实部和虚部
        U_real = U.real  # [B, 64]
        U_imag = U.imag  # [B, 64]
        V_real = V.real  # [B, 64]
        V_imag = V.imag  # [B, 64]

        # 计算 U * diag(S)
        # 将 S 扩展到与 U 相同的维度
        S_expanded = S.unsqueeze(-1)  # [B, 1, 1]
        U_real_S = U_real.unsqueeze(-1) * S_expanded  # [B, 64, 1]
        U_imag_S = U_imag.unsqueeze(-1) * S_expanded  # [B, 64, 1]

        # 计算 (U * diag(S)) * V^H
        # 使用广播机制计算外积
        real_part = (U_real_S * V_real.unsqueeze(1)) - (U_imag_S * V_imag.unsqueeze(1))
        imag_part = (U_real_S * V_imag.unsqueeze(1)) + (U_imag_S * V_real.unsqueeze(1))

        # 组合结果
        result = torch.stack([real_part, imag_part], dim=-1)  # [B, 64, 64, 2]

        return result

def get_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, max_lr, min_lr):
    """
    warmup_epochs: warmup 轮数
    total_epochs: 总训练轮数
    base_lr: warmup 开始学习率
    max_lr: warmup 结束学习率（也是余弦退火的起始学习率）
    min_lr: 余弦退火的最低学习率
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性 warmup: base_lr -> max_lr
            return (base_lr + (max_lr - base_lr) * (epoch + 1) / warmup_epochs) / base_lr
        else:
            # 余弦退火: max_lr -> min_lr
            progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
            return cosine_lr / base_lr
    return LambdaLR(optimizer, lr_lambda)

def get_scheduler_LR(optimizer, warmup_epochs, total_epochs, base_lr, max_lr, min_lr):
    """
    warmup_epochs: warmup 轮数（线性 0.01 -> 0.1）
    total_epochs: 总训练轮数（200）
    base_lr: warmup 开始学习率（0.01）
    max_lr: warmup 结束学习率（0.1）
    min_lr: 线性衰减的最低学习率（0.01）
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性 warmup: base_lr -> max_lr（按你的格式返回相对因子）
            return (base_lr + (max_lr - base_lr) * (epoch + 1) / warmup_epochs) / base_lr
        else:
            # 线性衰减: max_lr -> min_lr
            progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            lin_lr = max_lr + (min_lr - max_lr) * progress
            return lin_lr / base_lr
    return LambdaLR(optimizer, lr_lambda)