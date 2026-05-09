import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Backbone Blocks --------------------
class GatedConv1D(nn.Module):
    """Depthwise-separable gated conv (低MACs)."""

    def __init__(self, dim: int, k: int = 5):
        super().__init__()
        self.dim = dim
        self.dw = nn.Conv1d(dim, dim * 2, k, groups=dim, padding=k // 2, bias=True)
        self.pw = nn.Conv1d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        y = self.dw(x)
        a, g = y[:, :self.dim, :], y[:, self.dim:, :]
        y = a * torch.sigmoid(g)
        y = self.pw(y)
        return y.permute(0, 2, 1)


class GroupedProjectedAttention(nn.Module):
    """
    Drop-in 替代原 GroupedProjectedAttention：
    - 在 dim(通道) 维分组，组内计算注意力（等同多头）
    - 用 1D Conv 生成每个组的 k_len 个“动态锚点”的 over-T 聚合权重（对 K/V 各自或共享）
    - 复杂度 O(B · groups · T · k_len · head_dim)

    Args:
        dim:        输入通道维 (必须能被 groups 整除)
        att_dim:    每组 Q/K 的维度 (head_dim_QK)
        groups:     分组数 (= heads)
        k_len:      动态锚点个数 (T 被压到 k_len)
        attn_dropout: 注意力矩阵的 dropout
        proj_dropout: 输出的 dropout
        temperature: softmax 温度 (qk/√d 之外的缩放；一般 1.0 即可)
        share_kv:   聚合权重是否 K/V 共享 (True 推荐；False 为K/V各自单独一套)
    """
    def __init__(self,
                 dim: int,
                 att_dim: int,
                 groups: int,
                 k_len: int,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 temperature: float = 1.0,
                 share_kv: bool = True):
        super().__init__()
        assert dim % groups == 0, "dim 必须能被 groups 整除"
        self.dim = dim
        self.groups = groups
        self.gdim = dim // groups
        self.att_dim = att_dim
        self.k_len = k_len
        self.temperature = float(temperature)
        self.share_kv = share_kv

        # 每组独立的 Q/K/V/输出线性层
        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])  # V也映射为att_dim
        self.os = nn.ModuleList([nn.Linear(att_dim, self.gdim, bias=False) for _ in range(groups)])  # O将att_dim降维回gdim

        # 用轻量 1D Conv 生成“动态锚点”对 T 的聚合权重（logits），
        # 形状：(B, k_len, T)，再对 T softmax，得到 beta[b, k, t]
        # 对 K 和 V 都是 att_dim 作为 in_channels
        self.proj_gen_k = nn.Conv1d(in_channels=att_dim, out_channels=k_len, kernel_size=3, padding=1, groups=1)
        if share_kv:
            self.proj_gen_v = None
        else:
            self.proj_gen_v = nn.Conv1d(in_channels=att_dim, out_channels=k_len, kernel_size=3, padding=1, groups=1)

        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        # 轻量初始化，更稳
        for lin in list(self.qs) + list(self.ks) + list(self.vs) + list(self.os):
            nn.init.xavier_uniform_(lin.weight)
            if getattr(lin, "bias", None) is not None:
                nn.init.zeros_(lin.bias)
        nn.init.kaiming_uniform_(self.proj_gen_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.proj_gen_k.bias)
        if self.proj_gen_v is not None:
            nn.init.kaiming_uniform_(self.proj_gen_v.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj_gen_v.bias)

    @torch.no_grad()
    def _apply_key_padding_mask_over_T(self, logits: torch.Tensor, key_padding_mask: torch.Tensor):
        """
        logits: (B, k_len, T) —— 对 T 维做 softmax 前的logits
        key_padding_mask: (B, T) —— True 表示该 t 位置是 padding，应被屏蔽
        """
        # 把被mask的位置logits置为一个大负数，softmax 后≈0
        very_neg = torch.finfo(logits.dtype).min / 2
        logits.masked_fill_(key_padding_mask.unsqueeze(1), very_neg)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, T, dim]
        key_padding_mask: [B, T]，True 表示该位置是 padding（可选）
        return: [B, T, dim]
        """
        B, T, D = x.shape
        assert D == self.dim

        # [B, T, groups, gdim] -> [B, groups, T, gdim]
        xg = x.view(B, T, self.groups, self.gdim).permute(0, 2, 1, 3)

        outs = []
        scale = 1.0 / math.sqrt(self.att_dim)

        for g in range(self.groups):
            # 1) 每组计算 Q/K/V
            q = self.qs[g](xg[:, g])  # [B, T, att_dim]
            k = self.ks[g](xg[:, g])  # [B, T, att_dim]
            v = self.vs[g](xg[:, g])  # [B, T, gdim]

            # 2) 生成“锚点对T的聚合权重” beta_k, beta_v  (softmax dim = T)
            #    logits_k: [B, k_len, T]  ← Conv1d(att_dim -> k_len) over T
            logits_k = self.proj_gen_k(k.transpose(1, 2))  # [B, k_len, T]
            if key_padding_mask is not None:
                self._apply_key_padding_mask_over_T(logits_k, key_padding_mask)
            beta_k = F.softmax(logits_k, dim=-1)  # [B, k_len, T]

            if self.share_kv:
                beta_v = beta_k
            else:
                logits_v = self.proj_gen_v(v.transpose(1, 2))  # [B, k_len, T]
                if key_padding_mask is not None:
                    self._apply_key_padding_mask_over_T(logits_v, key_padding_mask)
                beta_v = F.softmax(logits_v, dim=-1)

            # 3) 按“锚点”聚合 K/V： [B, k_len, T] @ [B, T, att_dim] -> [B, k_len, att_dim]
            k_red = torch.einsum('bkt,bta->bka', beta_k, k)  # 压 T 得到 k_len 个动态 anchor 的 K
            v_red = torch.einsum('bkt,btg->bkg', beta_v, v)  # 同理得到 V

            # 4) 常规注意力：Q[T,att] × K_red[k_len,att]^T -> [T,k_len]
            att_logits = torch.matmul(q, k_red.transpose(1, 2)) * (scale * self.temperature)
            att = F.softmax(att_logits, dim=-1)  # [B, T, k_len]
            att = self.attn_drop(att)

            # 5) 加权汇总： [B, T, k_len] × [B, k_len, att_dim] -> [B, T, att_dim]
            out_g = torch.matmul(att, v_red)
            out_g = self.os[g](out_g)           # 组内输出线性: [B, T, att_dim] -> [B, T, gdim]
            outs.append(out_g)

        out = torch.cat(outs, dim=-1)           # 拼回 dim
        out = self.proj_drop(out)
        return out


# -------------------- Axial Low-Rank Frequency Gate --------------------
class AxialLowRankFreqGate(nn.Module):
    def __init__(self, M: int, N: int, hidden: int = 32, temperature: float = 0.2, k: int = 1):
        super().__init__()
        self.M, self.N, self.T = M, N, temperature
        self.hidden = hidden
        # Robust statistic for low-SNR gating: Top-K mean is much less sensitive to
        # random noise spikes than max/amax.
        self.topk = 4
        self.row_mlp = nn.Sequential(
            nn.Linear(M, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, M, bias=False)
        )
        self.col_mlp = nn.Sequential(
            nn.Linear(N, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, N, bias=False)
        )

    def _topk_mean(self, x: torch.Tensor, dim: int, k: int) -> torch.Tensor:
        """Return mean of top-k values along a dimension."""
        k = int(k)
        k = max(1, min(k, x.size(dim)))
        v, _ = torch.topk(x, k=k, dim=dim)
        return v.mean(dim=dim)

    def forward(self, Hf: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(Hf)
        # [Improvement 1] Use MEAN + MAX statistics (CBAM-style).
        # Mean captures background level; Max captures dominant signal peaks.
        # Simple summation implies an "OR" logic for importance.

        # Robust stats: mean + top-k mean (replaces max).
        r_stat = mag.mean(dim=2) + self._topk_mean(mag, dim=2, k=self.topk)  # (B,M)
        c_stat = mag.mean(dim=1) + self._topk_mean(mag, dim=1, k=self.topk)  # (B,N)
        
        r_w = self.row_mlp(r_stat)
        c_w = self.col_mlp(c_stat)
        
        # [Improvement 2] Gate Sharpening (Power scaling).
        # Sigmoid gives 0.0~1.0. Squaring it pushes weak gates (e.g. 0.2) -> 0.04,
        # while keeping strong gates (e.g. 0.9) -> 0.81. This cleans the noise floor aggressively.
        gate = torch.sigmoid((r_w.unsqueeze(2) + c_w.unsqueeze(1)) / self.T)

        # new ，对门限加平方，进一步压低低权重
        gate = gate.pow(1)
        
        return gate


# -------------------- NEW: Neural Ortho Refiner  --------------------
class NeuralOrthoRefiner(nn.Module):
    """One-step learned orthogonality refinement (no QR/SVD; rule-compliant).
    U' = U - a * U * sym(U^H U - I);  V' = V - b * V * sym(V^H V - I)
    a,b in (0, 0.5] learned via sigmoid.
    """

    def __init__(self, init_scale: float = 0.1, max_scale: float = 0.5):
        super().__init__()
        self.max_scale = float(max_scale)
        self.a_raw = nn.Parameter(torch.tensor(math.log(init_scale / (max_scale - init_scale))))
        self.b_raw = nn.Parameter(torch.tensor(math.log(init_scale / (max_scale - init_scale))))

    def _scale(self, p):
        return torch.sigmoid(p) * self.max_scale

    def _sym(self, G):
        return 0.5 * (G + G.conj().transpose(-2, -1))

    def forward(self, U: torch.Tensor, V: torch.Tensor):
        a = self._scale(self.a_raw)
        b = self._scale(self.b_raw)
        Br = U.shape[-1]
        I = torch.eye(Br, device=U.device, dtype=U.dtype)
        Gu = U.conj().transpose(-2, -1) @ U
        Gv = V.conj().transpose(-2, -1) @ V
        Du = self._sym(Gu - I)
        Dv = self._sym(Gv - I)
        U2 = U - a * (U @ Du)
        V2 = V - b * (V @ Dv)
        # column normalization (allowed by rules)
        U2 = U2 / (torch.linalg.norm(U2, dim=1, keepdim=True) + 1e-8)
        V2 = V2 / (torch.linalg.norm(V2, dim=1, keepdim=True) + 1e-8)
        return U2, V2


class ChannelAttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, M, C)
        w = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # (B, C)
        w = self.fc(w).unsqueeze(1)  # (B, 1, C)
        return x * w  # (B, M, C)


# -------------------- Spatial Refinement Block (2D Conv Denoising) --------------------
class SpatialRefinementBlock(nn.Module):
    """
    针对协方差矩阵优化的空间去噪模块
    """

    def __init__(self, in_channels=2, hidden_channels=32, enforce_hermitian=True):
        super().__init__()
        self.enforce_hermitian = enforce_hermitian
        # 是否启用 Forward-Backward Averaging (FBA)
        # Controlled by DDCRNet.use_fba (default: False), so clean targets can
        # receive the same projection during training/validation if enabled.
        self.use_fba = False
        # 保持你原有的网络结构不变，这部分设计很好
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            # 最后一个卷积通常初始化为接近 0，这样初始状态下 output ≈ input
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )

        # [Tip] 初始化最后一个卷积层的权重更小，让初始阶段更接近 Identity Mapping
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        # x: (B, 2, M, N) representing (Real, Imag) parts

        # 1. 计算残差去噪
        residual = self.net(x)
        out = x + residual
        
        if not self.enforce_hermitian:
            return out

        # 2. [关键优化] 强制恢复厄米特对称性 (Hermitian Symmetry)
        # R_out = 0.5 * (R + R^H)
        # Real part: (A + A.T) / 2
        # Imag part: (B - B.T) / 2  <-- 注意这里是减，因为 R_ji = conj(R_ij) => Im(R_ji) = -Im(R_ij)

        out_real = out[:, 0, :, :]
        out_imag = out[:, 1, :, :]

        # 恢复实部对称： R_ij_real == R_ji_real
        sym_real = 0.5 * (out_real + out_real.transpose(-1, -2))

        # 恢复虚部反对称： R_ij_imag == -R_ji_imag
        # 对角线虚部必须为 0

        sym_imag = 0.5 * (out_imag - out_imag.transpose(-1, -2))

        # 强制虚部对角线为 0（数值上更稳）
        diag = torch.diagonal(sym_imag, dim1=-2, dim2=-1)
        sym_imag = sym_imag - torch.diag_embed(diag)

        # [Optional] Forward-Backward Averaging (Centro-Hermitian enforcement)
        # 仅在 self.use_fba=True 时启用，避免与使用 clean 协方差监督时产生不可达的 loss floor。
        out_ri = torch.stack([sym_real, sym_imag], dim=1)
        if self.use_fba:
            out_ri = apply_fba_ri(out_ri)
        return out_ri


def apply_fba_ri(x_ri: torch.Tensor) -> torch.Tensor:
    """Apply Forward-Backward Averaging (FBA) to RI-form covariance.

    Args:
        x_ri: (B,2,N,N) with x_ri[:,0]=Real, x_ri[:,1]=Imag

    Returns:
        (B,2,N,N) after centro-Hermitian projection:
        R_fb = 0.5 * (R + J * conj(R) * J)
    """
    if x_ri.dim() != 4 or x_ri.size(1) != 2 or x_ri.size(-1) != x_ri.size(-2):
        raise ValueError(f"Expect x_ri with shape (B,2,N,N), got {tuple(x_ri.shape)}")
    real = x_ri[:, 0]
    imag = x_ri[:, 1]

    # J * X * J 等价于翻转最后两个维度 (flip up-down & left-right)
    real_flip = torch.flip(real, dims=[-2, -1])
    imag_flip = torch.flip(imag, dims=[-2, -1])

    out_real = 0.5 * (real + real_flip)
    out_imag = 0.5 * (imag - imag_flip)
    return torch.stack([out_real, out_imag], dim=1)


def diag_whiten_hermitian(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Diagonal whitening / normalization for Hermitian matrices.

    Given R (B,N,N) complex Hermitian-like, return:
        Rw = D^{-1/2} R D^{-1/2}, where D=diag(Re(R)).

    This suppresses diagonal dominance at low SNR and emphasizes off-diagonal structure.
    """
    if R.dim() != 3 or R.size(-1) != R.size(-2):
        raise ValueError(f"Expect R with shape (B,N,N), got {tuple(R.shape)}")
    d = torch.diagonal(R.real, dim1=-2, dim2=-1)  # (B,N)
    inv_sqrt = torch.rsqrt(torch.clamp(d, min=eps))
    # (B,N,1) * (B,N,N) * (B,1,N)
    return (inv_sqrt.unsqueeze(-1) * R) * inv_sqrt.unsqueeze(-2)


# -------------------- DDCR Predictor  --------------------
class DDCRNet(nn.Module):
    def __init__(self, M: int, N: int, r: int,
                 dim: int = 64, depth: int = 2, groups: int = 4, attn_dim = 32, doa_num_baselines: int = 4,
                 kernel_size: int = 3, temperature: float = 0.2,
                 k_len: int = 8, tau_s: float = 0.9, gate_hidden: int = 32,
                 task_mode: str = "doa", num_cls_max_sources: int = 4,
                 refine_mode: str = "both"):
        super().__init__()
        self.M, self.N, self.r = M, N, r
        self.temperature = temperature
        self.tau_s = tau_s
        self.k_len = k_len
        self.gate_hidden = gate_hidden
        self.groups = groups
        self.depth = depth
        self.dim = dim
        self.attn_dim = attn_dim
        self.doa_num_baselines = int(doa_num_baselines)
        self.task_mode = str(task_mode).lower().strip()
        if self.task_mode not in {"doa", "num_cls"}:
            raise ValueError(f"task_mode must be 'doa' or 'num_cls', got: {task_mode}")
        self.num_cls_max_sources = int(num_cls_max_sources)
        if self.num_cls_max_sources < 2:
            raise ValueError("num_cls_max_sources must be >= 2 for classification")

        # DoA output clipping: tanh-compression near +/-90deg can saturate gradients.
        # Keep it OFF by default; enable only for inference safety if needed.
        self.clip_theta = False

        # (Step-1) Spectral Refinement (FFT域)
        # enforce_hermitian=False because FFT spectrum is NOT Hermitian symmetric.
        self.spectral_refine = SpatialRefinementBlock(in_channels=2, hidden_channels=32, enforce_hermitian=False)

        # (Step-2) Spatial Refinement (空域)
        self.spatial_refine = SpatialRefinementBlock(in_channels=2, hidden_channels=32, enforce_hermitian=True)

        # Forward-Backward Averaging (FBA) switch: keep OFF by default.
        # If enabled, we must apply the same transform to clean targets in train/validate.
        self.use_fba = False
        self.spatial_refine.use_fba = self.use_fba

        # Refinement switch:
        #   both / spectral_only / spatial_only / none
        self.set_refine_mode(refine_mode)

        # Diagonal whitening / normalization (recommended ON for low SNR)
        self.use_diag_whiten = False
        # Optional: also whiten after spatial refinement (ablation). Usually keep OFF.
        self.whiten_after_refine = False

        # self.channel_fusion = ChannelAttentionFusion(in_channels=2 * N, reduction= 4)

        # backbone
        self.input_proj = nn.Linear(2 * N, dim)
        
        # [Optimization] Use fixed Sinusoidal Positional Encoding for ULA geometry inductive bias.
        # Learnable POS can overfit to specific sensor indices.
        self.pos_fixed = self._get_sinusoidal_pos_enc(self.N, dim)
        self.register_buffer('pos', self.pos_fixed) # Registered as buffer (not a parameter)

        # 修改
        att_dim = self.attn_dim
        blocks = []
        for i in range(depth):
            if i % 2 == 0:
                blocks.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    GroupedProjectedAttention(dim, att_dim, groups, k_len=k_len)
                ))
            else:
                blocks.append(nn.Sequential(nn.LayerNorm(dim), GatedConv1D(dim, kernel_size)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)

        # heads
        # output r complex vectors -> 2*r real/imag channels per sensor token
        self.u_head = nn.Linear(dim, 2 * self.r)
        # -------------------- DoA head (NEW): operate on token features BEFORE u_head --------------------
        # Motivation: at low SNR, fitted u may be inaccurate -> doa from u degrades.
        # We try to estimate DoA directly from the backbone token features x_n: (B,M,dim).
        #
        # Design goals:
        # - exploit ULA geometry: DoA mainly lives in *relative phase progression* along array
        # - keep global-phase invariance: use sin/cos of phase differences of learned complex tokens
        # - be light and stable: local gated conv + attention pooling over sensor index m
        self.doa_feat_proj = nn.Linear(dim, dim)
        self.doa_token_to_complex = nn.Linear(dim, 2)  # produce per-sensor complex surrogate
        self.doa_x_proj = nn.Linear(3 * self.doa_num_baselines, dim)
        self.doa_x_conv = GatedConv1D(dim, k=5)
        self.doa_x_pool = nn.Linear(dim, 1)
        self.doa_x_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim, 2 * self.M),  # (sin, cos) for each of r sources
        )

        # -------------------- Source-count head (independent task) --------------------
        # Used when task_mode == "num_cls".
        # Class index 0..C-1 corresponds to source count 1..C.
        # self.count_pool = nn.Linear(dim, 1)
        # self.count_head = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(dim, self.num_cls_max_sources),
        # )

        self.apply(self._init)

    def _get_sinusoidal_pos_enc(self, M, dim):
        """Generate fixed sinusoidal positional encoding for ULA."""
        pe = torch.zeros(M, dim)
        position = torch.arange(0, M, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # (1, M, dim)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None: nn.init.zeros_(m.bias)

    @torch.no_grad()
    def set_tau_s(self, tau_begin = 1.0, tau_end = 1.0,epoch_id: int = 0, epoch_end: int = 100):
        self.tau_s = float(np.interp(epoch_id, [0, epoch_end], [tau_begin, tau_end]))
        if epoch_id > epoch_end:
            self.tau_s = tau_end

    def set_refine_mode(self, refine_mode: str = "both"):
        """Set refinement path.

        Args:
            refine_mode: one of {"both", "spectral_only", "spatial_only", "none"}
        """
        mode = str(refine_mode).lower().strip()
        valid_modes = {"both", "spectral_only", "spatial_only", "none"}
        if mode not in valid_modes:
            raise ValueError(f"refine_mode must be one of {valid_modes}, got: {refine_mode}")
        self.refine_mode = mode
        self.use_spectral_refine = mode in {"both", "spectral_only"}
        self.use_spatial_refine = mode in {"both", "spatial_only"}

    def forward(self, H, return_aux: bool = False):
        """Forward.

        Args:
                        H: (B,N,N,2) real/imag (or complex (B,N,N))
            return_aux: if True, also return a dict of intermediate tensors

        Returns:
                        task_mode='doa':
                            (U, S, V, theta_deg) or (U, S, V, theta_deg, aux)
                        task_mode='num_cls':
                            (count_logits,) or (count_logits, aux)
        """
        B = H.size(0)
        H, scale = self.frobenius_norm_normalize(H)
        if H.dim() == 4 and H.shape[-1] == 2:
            H = torch.view_as_complex(H.to(torch.float32))

        aux = {} if bool(return_aux) else None
        if aux is not None:
            aux["H_norm"] = H

        # Diagonal whitening BEFORE FFT/gating to mitigate diagonal dominance.
        if bool(getattr(self, "use_diag_whiten", False)):
            H = diag_whiten_hermitian(H)
        if aux is not None:
            aux["H_whiten"] = H

        Hd = H
        if self.use_spectral_refine:
            # frequency-domain processing
            Hf_pre = torch.fft.fft2(Hd, norm='ortho')
            if aux is not None:
                aux["Hf_pre"] = Hf_pre

            # Hf is (B, M, N) complex. Convert to (B, 2, M, N) for CNN.
            Hf_ri = torch.view_as_real(Hf_pre).permute(0, 3, 1, 2).contiguous()
            Hf_ri = self.spectral_refine(Hf_ri)
            Hf_post = torch.view_as_complex(Hf_ri.permute(0, 2, 3, 1).contiguous())
            if aux is not None:
                aux["Hf_post"] = Hf_post

            Hd = torch.fft.ifft2(Hf_post, norm='ortho')
            if aux is not None:
                aux["Hd_ifft"] = Hd

        if self.use_spatial_refine:
            # spatial-domain refinement
            Hd_ri = torch.view_as_real(Hd).permute(0, 3, 1, 2).contiguous()
            Hd_ri = self.spatial_refine(Hd_ri)
            Hd = torch.view_as_complex(Hd_ri.permute(0, 2, 3, 1).contiguous())
            if aux is not None:
                aux["Hd_refined"] = Hd

        # Optional whitening after refine (usually OFF; helpful for ablations)
        if bool(getattr(self, "whiten_after_refine", False)):
            Hd = diag_whiten_hermitian(Hd)
        if aux is not None:
            aux["Hd_final"] = Hd

        # dual representation
        f1 = torch.view_as_real(Hd).reshape(B, self.N, -1)
        x = self.input_proj(torch.cat([f1], dim=-1)) + self.pos

        for blk in self.blocks:
            x = x + blk(x)
        x_n = self.norm(x)
        if aux is not None:
            aux["x_n"] = x_n

        if self.task_mode == "num_cls":
            att_logits = self.count_pool(x_n)               # (B,N,1)
            att = torch.softmax(att_logits, dim=1)          # (B,N,1)
            z_global = torch.sum(att * x_n, dim=1)          # (B,dim)
            count_logits = self.count_head(z_global).to(torch.float32)
            if aux is not None:
                aux["count_logits"] = count_logits
                return (count_logits, aux)
            return (count_logits,)

        feat = self.u_head(x_n)

        # ---- DoA prediction: from token features x_n (before u) ----
        theta_deg = self.predict_theta_deg_from_x(x_n)

        # heads
        feat = feat.view(B, self.N, self.r, 2)
        U = torch.view_as_complex(feat)

        # column normalization + sorting
        U = U / (torch.linalg.norm(U, dim=1, keepdim=True) + 1e-8)

        # spectral self-calibration (multi-source): U^H R U -> (B, r, r)
        Uh = U.conj().transpose(1, 2)  # (B, r, M)
        M_ = Uh @ Hd @ U  # (B, r, r)
        S_best = torch.abs(torch.diagonal(M_, dim1=-2, dim2=-1))
        V = U.clone()
        out = (U, S_best * scale.squeeze(-1, -2), V, theta_deg)
        if aux is not None:
            return out + (aux,)
        return out

    def _doa_features_from_tokens(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Build ULA-inspired multi-baseline features from token features.

        We first map each sensor token to a *complex surrogate* z_m = a_m + j b_m.
        Then for baselines k=1..K compute:
          c_m^(k) = conj(z_m) * z_{m+k}
        and use [sin(angle), cos(angle), |c|] as stable, global-phase-invariant features.

        Args:
            x_tokens: (B, M, dim)
        Returns:
            feat: (B, M-1, 3K)  (padding zeros for unavailable edges when k>1)
        """
        if x_tokens.dim() != 3:
            raise ValueError(f"Expect x_tokens with shape (B,M,dim), got {tuple(x_tokens.shape)}")
        B, M, _ = x_tokens.shape
        K = max(1, min(self.doa_num_baselines, M - 1))
        L = M - 1

        # produce complex surrogate per sensor
        xt = self.doa_feat_proj(x_tokens)
        ri = self.doa_token_to_complex(xt)  # (B,M,2)
        z = torch.view_as_complex(ri.to(torch.float32).contiguous())  # (B,M) complex64
        z = z / (torch.linalg.norm(z, dim=1, keepdim=True) + 1e-8)

        device = x_tokens.device
        dtype = x_tokens.dtype
        feat = torch.zeros((B, L, 3 * K), device=device, dtype=dtype)

        eps = 1e-8
        for i, k in enumerate(range(1, K + 1)):
            c = torch.conj(z[:, :-k]) * z[:, k:]  # (B, M-k)
            phi = torch.atan2(c.imag, c.real)
            sin_phi = torch.sin(phi).to(dtype)
            cos_phi = torch.cos(phi).to(dtype)
            mag = torch.abs(c).to(dtype)
            mag = mag / (mag.mean(dim=1, keepdim=True) + eps)

            ch0 = 3 * i
            valid_len = M - k
            feat[:, :valid_len, ch0] = sin_phi
            feat[:, :valid_len, ch0 + 1] = cos_phi
            feat[:, :valid_len, ch0 + 2] = mag

        return feat

    def predict_theta_deg_from_x(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Predict DoA (deg) from backbone token features (B,M,dim)."""
        edge_feat = self._doa_features_from_tokens(x_tokens)  # (B, M-1, 3K)
        z = self.doa_x_proj(edge_feat)
        z = z + self.doa_x_conv(z)
        att_logits = self.doa_x_pool(z)
        att = torch.softmax(att_logits, dim=1)
        z_global = torch.sum(att * z, dim=1)
        doa_sc = self.doa_x_head(z_global)
        doa_sc = doa_sc.view(-1, self.M, 2)
        doa_sc = doa_sc / (torch.linalg.norm(doa_sc, dim=-1, keepdim=True) + 1e-8)
        theta_rad = torch.atan2(doa_sc[:, :, 0], doa_sc[:, :, 1])
        theta_deg = theta_rad * (180.0 / math.pi)
        if bool(getattr(self, "clip_theta", False)):
            theta_deg = 90.0 * torch.tanh(theta_deg / 90.0)
        return theta_deg.to(torch.float32)

    def _doa_features_from_u2(self, U: torch.Tensor) -> torch.Tensor:
        """
        多基线 DoA 特征：
          对 k=1..K 计算 c_m^(k) = conj(u_m) * u_{m+k}
          并输出每个 k 的 [sin(angle), cos(angle), |c|]，最后在 feature 维拼接成 3K 维。

        返回:
          feat: (B, M-1, 3K)  —— 每个 m 位置都有 K 个基线特征（对 k>可用范围的部分用 0 padding）
        """
        # U: (B, M) complex
        B, M = U.shape
        K = max(1, min(self.doa_num_baselines, M - 1))

        # 统一的输出长度（以 k=1 的边数为基准）：L = M-1
        L = M - 1
        device = U.device
        dtype = U.real.dtype

        # 初始化输出 (B, L, 3K)
        feat = torch.zeros((B, L, 3 * K), device=device, dtype=dtype)

        # 逐基线填充：k 基线只有 (M-k) 条边，放到前 (M-k) 个位置，其余 padding 为 0
        eps = 1e-8
        for i, k in enumerate(range(1, K + 1)):
            # c^(k): (B, M-k)
            c = torch.conj(U[:, :-k]) * U[:, k:]

            # angle -> sin/cos，避免相位跳变不连续
            phi = torch.atan2(c.imag, c.real)  # (B, M-k)
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)

            # 幅度可作为置信度特征（也可做归一化）
            mag = torch.abs(c)  # (B, M-k)
            # 可选：幅度归一化（让不同 batch 更一致）
            mag = mag / (mag.mean(dim=1, keepdim=True) + eps)

            # 写入 feat 的对应通道
            # 通道布局： [sin_k, cos_k, mag_k] 在连续的 3 个通道
            ch0 = 3 * i
            ch1 = ch0 + 1
            ch2 = ch0 + 2

            valid_len = M - k  # = c.shape[1]
            feat[:, :valid_len, ch0] = sin_phi
            feat[:, :valid_len, ch1] = cos_phi
            feat[:, :valid_len, ch2] = mag

        return feat

    def frobenius_norm_normalize(self, x: torch.Tensor, eps: float = 1e-8):
        assert x.dim() == 4 and x.size(-1) == 2, "Expect x of shape (B,M,N,2)"
        # 计算 |H|^2 = Re^2 + Im^2
        mag2 = x[..., 0] ** 2 + x[..., 1] ** 2

        # Frobenius 范数：sqrt( sum_{m,n} |H_{mn}|^2 )
        scales = torch.sqrt(mag2.sum(dim=(-2, -1), keepdim=True) + eps)
        scales = scales.unsqueeze(-1)
        x_norm = x / (scales + eps)
        return x_norm, scales


