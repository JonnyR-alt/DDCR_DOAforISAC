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

        # 重新拼接
        return torch.stack([sym_real, sym_imag], dim=1)


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


# -------------------- SVD Predictor  --------------------
class SVDNet(nn.Module):
    def __init__(self, M: int, N: int, r: int,
                 dim: int = 64, depth: int = 2, groups: int = 4, attn_dim = 32, doa_num_baselines: int = 4,
                 kernel_size: int = 3, temperature: float = 0.2,
                 k_len: int = 8, tau_s: float = 0.9, gate_hidden: int = 32,
                 input_mode: str = "cov", snap_T: int | None = None):
        super().__init__()
        self.M, self.N, self.r = M, N, r
        self.input_mode = str(input_mode or "cov").lower()
        if self.input_mode not in {"cov", "snap"}:
            raise ValueError(f"Unknown input_mode={input_mode}, expected 'cov' or 'snap'.")
        self.snap_T = int(snap_T) if snap_T is not None else None
        self.temperature = temperature
        self.tau_s = tau_s
        self.k_len = k_len
        self.gate_hidden = gate_hidden
        self.groups = groups
        self.depth = depth
        # embedding dimension (token hidden size).
        # NOTE: do NOT confuse with raw snapshot feature dim (2*T).
        self.dim = int(dim)
        self.attn_dim = attn_dim
        self.doa_num_baselines = int(doa_num_baselines)

        # DoA output clipping: tanh-compression near +/-90deg can saturate gradients.
        # Keep it OFF by default; enable only for inference safety if needed.
        self.clip_theta = False

        # (Step-1) Spectral Refinement (Replaces simple FreqGate)
        # Move the Conv-based denoising to Frequency domain.
        # enforce_hermitian=False because FFT spectrum is NOT Hermitian symmetric.
        self.spectral_refine = SpatialRefinementBlock(in_channels=2, hidden_channels=32, enforce_hermitian=False)

        # (Step-2) Spatial Refinement (Optional Post-IFFT cleanup)
        # Keep this for strict Hermitian enforcement and additional spatial denoising.
        self.spatial_refine = SpatialRefinementBlock(in_channels=2, hidden_channels=32, enforce_hermitian=True)

        # Diagonal whitening / normalization (recommended ON for low SNR)
        self.use_diag_whiten = False
        # Optional: also whiten after spatial refinement (ablation). Usually keep OFF.
        self.whiten_after_refine = False

        # self.channel_fusion = ChannelAttentionFusion(in_channels=2 * N, reduction= 4)

        # backbone
        # cov-mode token dim: each row has N complex entries -> 2N real features
        self.input_proj = nn.Linear(2 * N, dim)

        # snap-mode token dim: each sensor token uses T complex samples -> 2T real features
        # Separate proj keeps cov/snap ablation clean.
        self.input_proj_snap = None
        if self.input_mode == "snap":
            if self.snap_T is None:
                raise ValueError("input_mode='snap' requires snap_T (e.g. 200).")
            self.input_proj_snap = nn.Linear(2 * self.snap_T, dim)

        # -------------------- Signal-as-Token (snap-mode): reg token + reconstruction head --------------------
        # reg token follows the paper: concatenate with real/imag parts along antenna dimension.
        # Here we implement it as TWO learnable tokens (Real-token, Imag-token).
        self.use_reg_token = (self.input_mode == "snap")
        self.reg_tokens_ri = None
        self.reg_doa_head = None
        self.snap_recon_head = None
        if self.use_reg_token:
            # (1,2,dim): two tokens
            self.reg_tokens_ri = nn.Parameter(torch.zeros(1, 2, dim))
            nn.init.trunc_normal_(self.reg_tokens_ri, std=0.02)
            # DoA regression head reads the two reg tokens
            self.reg_doa_head = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(dim, 2 * self.r),  # per-source (sin,cos)
            )
            # Reconstruction head maps each sensor token back to 2T real values
            self.snap_recon_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, 2 * int(self.snap_T)),
            )
        
        # [Optimization] Use fixed Sinusoidal Positional Encoding for ULA geometry inductive bias.
        # Learnable POS can overfit to specific sensor indices.
        self.pos_fixed = self._get_sinusoidal_pos_enc(M, dim)
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
            nn.Linear(dim, 2 * self.r),  # (sin, cos) for each of r sources
        )

        # -------------------- DoA head (legacy): derived from predicted U --------------------
        # Kept for backward compatibility / ablations.
        # self.doa_u_proj = nn.Linear(3 * self.doa_num_baselines, dim)
        # self.doa_u_conv = GatedConv1D(dim, k=5)
        # self.doa_u_pool = nn.Linear(dim, 1)
        # self.doa_u_head = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(dim, 2),
        # )
        # self.doa_u_proj = nn.Sequential(
        #     nn.Linear(3, doa_h),
        #     nn.ReLU(inplace=True),
        # )
        # self.doa_u_conv = GatedConv1D(doa_h, k=5)
        # self.doa_u_pool = nn.Sequential(
        #     nn.Linear(doa_h, doa_h // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(doa_h // 2, 1),
        # )
        # self.doa_u_head = nn.Sequential(
        #     nn.Linear(doa_h, doa_h),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(doa_h, 2),
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

    def forward(self, H, return_aux: bool = False):
        """Forward.

        Args:
            H: (B,M,N,2) real/imag (or complex (B,M,N))
            return_aux: if True, also return a dict of intermediate tensors

        Returns:
            (U, S, V, theta_deg) or (U, S, V, theta_deg, aux)
        """
        B = H.size(0)

        # -------------------- SNAPSHOT TOKEN MODE --------------------
        # H: complex snapshots (B,T,N) or real/imag (B,T,N,2)
        # tokens: (B,N,2T) -> backbone -> heads
        if self.input_mode == "snap":
            aux = {} if bool(return_aux) else None

            if H.dim() == 4 and H.size(-1) == 2:
                X = torch.view_as_complex(H.to(torch.float32).contiguous())
            elif torch.is_complex(H):
                X = H
            else:
                raise ValueError(
                    f"input_mode='snap' expects (B,T,N) complex or (B,T,N,2), got {tuple(H.shape)}"
                )
            if X.dim() != 3:
                raise ValueError(f"snapshots must be 3D (B,T,N), got {tuple(X.shape)}")
            if X.shape[-1] != self.N:
                raise ValueError(f"snapshots last dim must be N={self.N}, got {X.shape[-1]}")
            if self.snap_T is not None and X.shape[1] != self.snap_T:
                raise ValueError(f"snapshots T mismatch: expected {self.snap_T}, got {X.shape[1]}")
            if self.input_proj_snap is None:
                raise RuntimeError("input_proj_snap is None but input_mode='snap'.")

            # energy normalize per example for stability
            eps = 1e-8
            scales = torch.sqrt((X.real ** 2 + X.imag ** 2).sum(dim=(-2, -1)) + eps)  # (B,)
            Xn = X / (scales.view(B, 1, 1) + eps)

            # sensor tokens: (B,T,N)->(B,N,T)->(B,N,2T)
            Xn_nt = Xn.transpose(1, 2).contiguous()  # (B,N,T)
            f = torch.view_as_real(Xn_nt)  # (B,N,T,2)
            # NOTE: numpy complex128 -> torch complex128 -> view_as_real -> float64.
            # Linear layers are float32 by default, so cast to float32 here.
            f = f.reshape(B, self.N, -1).to(torch.float32)  # (B,N,2T)
            if aux is not None:
                aux["X_norm"] = Xn_nt

            # sensor tokens embedding
            x_tokens = self.input_proj_snap(f) + self.pos  # (B,N,dim)

            # prepend 2 reg tokens (Real/Imag) like Signal-as-Token
            if self.use_reg_token and self.reg_tokens_ri is not None:
                reg = self.reg_tokens_ri.expand(B, -1, -1)  # (B,2,dim)
                x = torch.cat([reg, x_tokens], dim=1)  # (B,2+N,dim)
            else:
                x = x_tokens
            for blk in self.blocks:
                x = x + blk(x)
            x_n = self.norm(x)

            # DoA prediction: from reg tokens if enabled
            if self.use_reg_token and self.reg_doa_head is not None:
                reg_out = x_n[:, :2, :].reshape(B, -1)  # (B,2*dim)
                sincos = self.reg_doa_head(reg_out).view(B, self.r, 2)  # (B,r,2)
                theta_rad = torch.atan2(sincos[..., 0], sincos[..., 1])
                theta_deg = theta_rad * (180.0 / math.pi)
            else:
                theta_deg = self.predict_theta_deg_from_x(x_n)

            # snapshot reconstruction: only over sensor tokens (exclude reg tokens)
            if self.use_reg_token and self.snap_recon_head is not None:
                sensor_feat = x_n[:, 2:, :]  # (B,N,dim)
                recon_flat = self.snap_recon_head(sensor_feat)  # (B,N,2T)
                recon_ri = recon_flat.view(B, self.N, int(self.snap_T), 2)
                recon_ri = recon_ri.permute(0, 2, 1, 3).contiguous()  # (B,T,N,2)
                snap_recon = torch.view_as_complex(recon_ri.to(torch.float32))  # (B,T,N)
                if aux is not None:
                    aux["snap_recon"] = snap_recon

            # placeholders for U/S/V to keep training script compatible
            U = torch.zeros((B, self.N, self.r), device=x_n.device, dtype=torch.complex64)
            S_best = scales.view(B, 1).repeat(1, self.r).to(torch.float32)
            V = U.clone()

            out = (U, S_best, V, theta_deg)
            if aux is not None:
                return out + (aux,)
            return out

        # -------------------- COVARIANCE MODE (original) --------------------
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

        # frequency domain processing
        Hf_pre = torch.fft.fft2(H, norm='ortho')
        if aux is not None:
            aux["Hf_pre"] = Hf_pre

        # [MODIFIED] Use Spectral Refinement Block (CNN) instead of FreqGate
        # Hf is (B, M, N) complex. Convert to (B, 2, M, N) for CNN.
        Hf_ri = torch.view_as_real(Hf_pre).permute(0, 3, 1, 2).contiguous()
        Hf_ri = self.spectral_refine(Hf_ri)
        Hf_post = torch.view_as_complex(Hf_ri.permute(0, 2, 3, 1).contiguous())
        if aux is not None:
            aux["Hf_post"] = Hf_post

        Hd = torch.fft.ifft2(Hf_post, norm='ortho')
        if aux is not None:
            aux["Hd_ifft"] = Hd

        # NEW: Spatial Refinement (Denoising)
        # Hd is (B, M, N) complex64. Convert to (B, 2, M, N) float32
        Hd_ri = torch.view_as_real(Hd).permute(0, 3, 1, 2).contiguous()
        Hd_ri = self.spatial_refine(Hd_ri)
        # Convert back to (B, M, N) complex64 just for naming consistency (though next line splits it anyway)
        Hd = torch.view_as_complex(Hd_ri.permute(0, 2, 3, 1).contiguous())
        if aux is not None:
            aux["Hd_refined"] = Hd

        # Optional whitening after refine (usually OFF; helpful for ablations)
        if bool(getattr(self, "whiten_after_refine", False)):
            Hd = diag_whiten_hermitian(Hd)
        if aux is not None:
            aux["Hd_final"] = Hd

        # dual representation
        f1 = torch.view_as_real(Hd).reshape(B, self.M, -1)  # (B,M,2N)
        # f2 = torch.view_as_real(Hc).reshape(B, self.M, -1)  # (B,M,2N)
        x = self.input_proj(torch.cat([f1], dim=-1)) + self.pos

        for blk in self.blocks:
            x = x + blk(x)
        x_n = self.norm(x)
        feat = self.u_head(x_n)

        # ---- DoA prediction: from token features x_n (before u) ----
        theta_deg = self.predict_theta_deg_from_x(x_n)

        # heads
        feat = feat.view(B, self.M, self.r, 2)
        U = torch.view_as_complex(feat)
        # V = torch.view_as_complex(self.v_head(feat).view(B, self.N, 2))
        # S_pred = self.s_head(feat) + 1e-6

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
        doa_sc = doa_sc.view(-1, self.r, 2)
        doa_sc = doa_sc / (torch.linalg.norm(doa_sc, dim=-1, keepdim=True) + 1e-8)
        theta_rad = torch.atan2(doa_sc[:, :, 0], doa_sc[:, :, 1])
        theta_deg = theta_rad * (180.0 / math.pi)
        if bool(getattr(self, "clip_theta", False)):
            theta_deg = 90.0 * torch.tanh(theta_deg / 90.0)
        return theta_deg.to(torch.float32)

    def _doa_features_from_u(self, u_complex: torch.Tensor) -> torch.Tensor:
        """Build per-edge features from complex u.

        Args:
            u_complex: (B, M) complex
        Returns:
            edge_feat: (B, M-1, 3) float32 with [Re(c), Im(c), |c|],
                      where c_m = conj(u_m) * u_{m+1}
        """
        if u_complex.dim() != 2:
            raise ValueError(f"Expect u_complex with shape (B,M), got {tuple(u_complex.shape)}")
        u = u_complex.to(torch.complex64)
        u = u / (torch.linalg.norm(u, dim=1, keepdim=True) + 1e-8)
        if u.shape[1] < 2:
            # no adjacent edge exists; return a dummy edge to avoid crash
            B, M = u.shape
            return torch.zeros((B, 1, 3), device=u.device, dtype=torch.float32)
        c = torch.conj(u[:, :-1]) * u[:, 1:]
        return torch.stack([c.real, c.imag, torch.abs(c)], dim=-1).to(torch.float32)

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

    def predict_theta_deg_from_u(self, u_complex: torch.Tensor) -> torch.Tensor:
        """Predict DoA (deg) from a provided complex u using the model's DoA head.

        This is used by ablation scripts to feed GT-u into the same learnable DoA head.

        Args:
            u_complex: (B, M) complex
        Returns:
            theta_deg: (B,) float32
        """
        edge_feat = self._doa_features_from_u2(u_complex)  # (B, M-1, 3)
        z = self.doa_u_proj(edge_feat)
        z = z + self.doa_u_conv(z)
        att_logits = self.doa_u_pool(z)
        att = torch.softmax(att_logits, dim=1)
        z_global = torch.sum(att * z, dim=1)
        doa_sc = self.doa_u_head(z_global)
        doa_sc = doa_sc / (torch.linalg.norm(doa_sc, dim=-1, keepdim=True) + 1e-8)
        theta_rad = torch.atan2(doa_sc[:, 0], doa_sc[:, 1])
        theta_deg = theta_rad * (180.0 / math.pi)
        if bool(getattr(self, "clip_theta", False)):
            theta_deg = 90.0 * torch.tanh(theta_deg / 90.0)
        return theta_deg.to(torch.float32)

    def frobenius_norm_normalize(self, x: torch.Tensor, eps: float = 1e-8):
        assert x.dim() == 4 and x.size(-1) == 2, "Expect x of shape (B,M,N,2)"
        # 计算 |H|^2 = Re^2 + Im^2
        mag2 = x[..., 0] ** 2 + x[..., 1] ** 2

        # Frobenius 范数：sqrt( sum_{m,n} |H_{mn}|^2 )
        scales = torch.sqrt(mag2.sum(dim=(-2, -1), keepdim=True) + eps)
        scales = scales.unsqueeze(-1)
        x_norm = x / (scales + eps)
        return x_norm, scales


class IterativeSVDNetWrapper(nn.Module):
    """Iterative (deflation-style) wrapper for multi-source estimation.

    This wrapper keeps the base network unchanged (it predicts ONE complex vector per call),
    and unrolls K steps:

        u_i = base_net(R_i)
        P_i = u_i u_i^H
        R_{i+1} = (I - P_i) R_i (I - P_i)

    Notes:
    - Input is expected to be complex matrix in real/imag format: (B, N, N, 2).
    - The returned U_seq is (B, K, N, 2) in the same real/imag format.
    - For numerical stability, u_i is re-normalized to unit norm and P_i is regularized.
    """

    def __init__(self, base_net: nn.Module, eps: float = 1e-8, detach_deflation: bool = False):
        super().__init__()
        self.base_net = base_net
        self.eps = float(eps)
        self.detach_deflation = bool(detach_deflation)

    def _to_complex_mat(self, R_ri: torch.Tensor) -> torch.Tensor:
        # R_ri: (B,N,N,2) -> complex (B,N,N)
        if R_ri.dim() != 4 or R_ri.size(-1) != 2:
            raise ValueError(f"Expect R with shape (B,N,N,2), got {tuple(R_ri.shape)}")
        return torch.view_as_complex(R_ri.to(torch.float32))

    def _to_ri(self, X: torch.Tensor) -> torch.Tensor:
        # complex -> (..,2) float32
        return torch.view_as_real(X)

    def forward(self, R_ri: torch.Tensor, K: int, return_residual: bool = False):
        """Run K-step iterative estimation.

        Args:
            R_ri: (B,N,N,2) real/imag covariance-like matrix.
            K: number of sources/iterations.
            return_residual: if True, also return final residual matrix (B,N,N,2).

        Returns:
            U_seq_ri: (B,K,N,2)
            (optional) R_res_ri: (B,N,N,2)
        """

        K = int(K)
        if K <= 0:
            raise ValueError("K must be positive")

        R = self._to_complex_mat(R_ri)  # (B,N,N)
        B, N, N2 = R.shape
        if N != N2:
            raise ValueError(f"Expect square matrix, got {tuple(R.shape)}")

        I = torch.eye(N, device=R.device, dtype=R.dtype).unsqueeze(0)  # (1,N,N)

        u_list = []
        doa_list = []
        for _ in range(K):
            # base net expects (B,M,N,2); here we use N as both M and N
            out = self.base_net(self._to_ri(R))
            # backward compatible: allow base_net to return either 3-tuple or 4-tuple
            if isinstance(out, (list, tuple)) and len(out) == 4:
                u_c, _, _, theta_deg = out
            else:
                u_c, _, _ = out
                theta_deg = None
            u_c = u_c.to(R.dtype)
            # unit-norm
            u_c = u_c / (torch.linalg.norm(u_c, dim=1, keepdim=True) + self.eps)
            u_list.append(self._to_ri(u_c))  # (B,N,2)
            if theta_deg is not None:
                doa_list.append(theta_deg)

            # PSD-friendly deflation: R <- P_perp R P_perp
            u_col = u_c.unsqueeze(-1)  # (B,N,1)
            P = u_col @ u_col.conj().transpose(-2, -1)  # (B,N,N)
            P_perp = I - P
            if self.detach_deflation:
                P_perp = P_perp.detach()
            R = P_perp @ R @ P_perp
            # keep Hermitian numerically
            R = 0.5 * (R + R.conj().transpose(-2, -1))

        U_seq_ri = torch.stack(u_list, dim=1)  # (B,K,N,2)
        doa_seq = torch.stack(doa_list, dim=1) if len(doa_list) > 0 else None  # (B,K)
        if return_residual:
            return U_seq_ri, doa_seq, self._to_ri(R)
        return U_seq_ri, doa_seq


    def frobenius_norm_normalize(self, x: torch.Tensor, eps: float = 1e-8):
        assert x.dim() == 4 and x.size(-1) == 2, "Expect x of shape (B,M,N,2)"
        # 计算 |H|^2 = Re^2 + Im^2
        mag2 = x[..., 0] ** 2 + x[..., 1] ** 2

        # Frobenius 范数：sqrt( sum_{m,n} |H_{mn}|^2 )
        scales = torch.sqrt(mag2.sum(dim=(-2, -1), keepdim=True) + eps)
        scales = scales.unsqueeze(-1)
        x_norm = x / (scales + eps)
        return x_norm, scales


    # ------------ Step-3: Export a structurally pruned model ------------
    @torch.no_grad()
    def export_pruned_model(self, keep_k_len: int, keep_gate_hidden: int):
        keep_k_len = int(max(8, min(self.k_len, keep_k_len)))
        keep_gate_hidden = int(max(8, min(self.gate_hidden, keep_gate_hidden)))

        # 1) new model with smaller k_len & gate hidden
        new_model = SVDNet(self.M, self.N, self.r,
                                   dim=self.dim, depth=self.depth, groups=self.groups,
                                   kernel_size=3, temperature=self.temperature,
                                   k_len=keep_k_len, tau_s=self.tau_s, gate_hidden=keep_gate_hidden)

        def copy_like(a, b):
            if a.shape == b.shape: b.data.copy_(a.data)

        # 2) shared weights_u_based
        copy_like(self.input_proj.weight, new_model.input_proj.weight)
        copy_like(self.input_proj.bias, new_model.input_proj.bias)
        new_model.pos.data.copy_(self.pos.data)
        new_model.scene_emb.weight.data.copy_(self.scene_emb.weight.data)

        # 3) blocks (LN + GPA / GatedConv)
        for blk_old, blk_new in zip(self.blocks, new_model.blocks):
            mods_old = list(blk_old.children())
            mods_new = list(blk_new.children())
            # LayerNorm
            copy_like(mods_old[0].weight, mods_new[0].weight)
            copy_like(mods_old[0].bias, mods_new[0].bias)
            if isinstance(mods_old[1], GroupedProjectedAttention):
                gpa_old: GroupedProjectedAttention = mods_old[1]
                gpa_new: GroupedProjectedAttention = mods_new[1]
                # Q/K/V/O identical
                for (qo, qn) in zip(gpa_old.qs, gpa_new.qs): copy_like(qo.weight, qn.weight)
                for (ko, kn) in zip(gpa_old.ks, gpa_new.ks): copy_like(ko.weight, kn.weight)
                for (vo, vn) in zip(gpa_old.vs, gpa_new.vs): copy_like(vo.weight, vn.weight)
                for (oo, on) in zip(gpa_old.os, gpa_new.os): copy_like(oo.weight, on.weight)
                # column selection for Pk/Pv by energy
                alpha = 1.0
                for g in range(gpa_old.groups):
                    Pk = gpa_old.Pk[g].data
                    Pv = gpa_old.Pv[g].data
                    score = (Pk.pow(2).sum(dim=0) + alpha * Pv.pow(2).sum(dim=0)).cpu().numpy()
                    keep_idx = np.argsort(-score)[:keep_k_len]
                    keep_idx = np.sort(keep_idx)
                    gpa_new.Pk[g].data.copy_(Pk[:, keep_idx])
                    gpa_new.Pv[g].data.copy_(Pv[:, keep_idx])
            else:
                gc_old: GatedConv1D = mods_old[1]
                gc_new: GatedConv1D = mods_new[1]
                copy_like(gc_old.dw.weight, gc_new.dw.weight);
                copy_like(gc_old.dw.bias, gc_new.dw.bias)
                copy_like(gc_old.pw.weight, gc_new.pw.weight);
                copy_like(gc_old.pw.bias, gc_new.pw.bias)

        # 4) heads
        copy_like(self.u_head.weight, new_model.u_head.weight);
        copy_like(self.u_head.bias, new_model.u_head.bias)
        copy_like(self.v_head.weight, new_model.v_head.weight);
        copy_like(self.v_head.bias, new_model.v_head.bias)
        copy_like(self.s_head[0].weight, new_model.s_head[0].weight);
        copy_like(self.s_head[0].bias, new_model.s_head[0].bias)

        # 5) freq gate hidden pruning
        if self.freq_gate.hidden == keep_gate_hidden:
            for i in [0, 2]:
                copy_like(self.freq_gate.row_mlp[i].weight, new_model.freq_gate.row_mlp[i].weight)
                copy_like(self.freq_gate.col_mlp[i].weight, new_model.freq_gate.col_mlp[i].weight)
        else:
            W1r = self.freq_gate.row_mlp[0].weight.data  # (hidden, M)
            W2r = self.freq_gate.row_mlp[2].weight.data  # (M, hidden)
            W1c = self.freq_gate.col_mlp[0].weight.data  # (hidden, N)
            W2c = self.freq_gate.col_mlp[2].weight.data  # (N, hidden)
            s_row = W1r.pow(2).sum(dim=1).sqrt() * W2r.pow(2).sum(dim=0).sqrt()
            s_col = W1c.pow(2).sum(dim=1).sqrt() * W2c.pow(2).sum(dim=0).sqrt()
            score = (s_row + s_col).cpu().numpy()
            keep = np.argsort(-score)[:keep_gate_hidden]
            keep = np.sort(keep)
            new_model.freq_gate.row_mlp[0].weight.data.copy_(W1r[keep, :])
            new_model.freq_gate.col_mlp[0].weight.data.copy_(W1c[keep, :])
            new_model.freq_gate.row_mlp[2].weight.data.copy_(W2r[:, keep])
            new_model.freq_gate.col_mlp[2].weight.data.copy_(W2c[:, keep])

        # 6) copy ortho_refine params
        new_model.ortho_refine.a_raw.data.copy_(self.ortho_refine.a_raw.data)
        new_model.ortho_refine.b_raw.data.copy_(self.ortho_refine.b_raw.data)

        return new_model