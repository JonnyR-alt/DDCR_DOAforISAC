import importlib
import torch
import torch.nn as nn


_base_module = importlib.import_module("models.model_副本")
_BaseSVDNet = _base_module.SVDNet

diag_whiten_hermitian = _base_module.diag_whiten_hermitian
apply_fba_ri = _base_module.apply_fba_ri


class SVDNetMLPHead(_BaseSVDNet):
    """Control variant that replaces the geometry-aware DoA head with plain MLP regressors."""

    def __init__(
        self,
        *args,
        mlp_dropout: float = 0.1,
        mlp_hidden: int = None,
        mlp_hidden2: int = None,
        mlp_u_hidden: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        pooled_dim = 2 * self.dim
        # Defaults are chosen to keep the total parameter count close to the
        # original model, so the readout-head ablation is not confounded by a
        # large capacity gap. Under the common setting (N=8, dim=128), this
        # yields an MLP head of roughly 5.5e4 parameters, which is close to the
        # replaced geometry-aware head.
        hidden = int(mlp_hidden or max(160, int(1.25 * self.dim)))
        hidden2 = int(mlp_hidden2 or max(48, int(0.3 * hidden)))
        u_hidden = int(mlp_u_hidden or max(64, 8 * self.N))

        # Keep the original attribute names so the existing training utilities
        # (freeze_mode / ablation hooks) still recognize these modules as DoA heads.
        self.doa_feat_proj = nn.Identity()
        self.doa_token_to_complex = nn.Identity()
        self.doa_x_proj = nn.Identity()
        self.doa_x_conv = nn.Identity()
        self.doa_x_pool = nn.Identity()
        self.doa_x_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden2, self.M),
        )

        # Optional U-based debug path used by the training code when doa_input is
        # switched to gt_u / steering_u. This is also kept as a plain MLP baseline.
        self.doa_u_proj = nn.Identity()
        self.doa_u_conv = nn.Identity()
        self.doa_u_pool = nn.Identity()
        self.doa_u_head = nn.Sequential(
            nn.Linear(2 * self.N, u_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(u_hidden, u_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(u_hidden, self.M),
        )

        self.doa_x_head.apply(self._init)
        self.doa_u_head.apply(self._init)

    @staticmethod
    def _bound_theta_deg(theta_raw: torch.Tensor) -> torch.Tensor:
        # Direct angle regression is unconstrained; keep the output in the
        # broadside range used by the dataset.
        return 90.0 * torch.tanh(theta_raw / 90.0)

    def predict_theta_deg_from_x(self, x_tokens: torch.Tensor) -> torch.Tensor:
        if x_tokens.dim() != 3:
            raise ValueError(f"Expect x_tokens with shape (B,N,dim), got {tuple(x_tokens.shape)}")
        x_tokens = x_tokens.to(torch.float32)
        # Lightweight pure-MLP baseline: summarize token sequence first, then regress.
        # This avoids the large parameter cost of flattening all N x dim features.
        x_mean = x_tokens.mean(dim=1)
        x_std = x_tokens.std(dim=1, unbiased=False)
        pooled = torch.cat([x_mean, x_std], dim=-1)
        theta_raw = self.doa_x_head(pooled)
        return self._bound_theta_deg(theta_raw).to(torch.float32)

    def predict_theta_deg_from_u(self, u_complex: torch.Tensor) -> torch.Tensor:
        if u_complex.dim() != 2:
            raise ValueError(f"Expect u_complex with shape (B,N), got {tuple(u_complex.shape)}")
        u_complex = u_complex.to(torch.complex64)
        u_complex = u_complex / (torch.linalg.norm(u_complex, dim=1, keepdim=True) + 1e-8)
        u_ri = torch.view_as_real(u_complex).reshape(u_complex.shape[0], -1).to(torch.float32)
        theta_raw = self.doa_u_head(u_ri)
        return self._bound_theta_deg(theta_raw).to(torch.float32)


SVDNet = SVDNetMLPHead
