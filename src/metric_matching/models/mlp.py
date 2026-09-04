import torch
import torch.nn as nn
from .unet_utils.nn import timestep_embedding


class ResBlockFiLM(nn.Module):
    """
    Residual MLP block with FiLM conditioning (scale/shift from time embedding).

    x -> LN -> FiLM -> Linear -> SiLU -> (Dropout) -> LN -> Linear -> +skip
    """

    def __init__(
        self,
        dim: int,
        temb_dim: int,
        dropout: float = 0.0,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

        # FiLM parameters for feature-wise conditioning
        self.film = nn.Linear(temb_dim, 2 * dim)

        # Helpful: start each block close to identity (safe for diffusion),
        # but does NOT force the final head output to be zero.
        if zero_init_residual:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # x: [B, dim], temb: [B, temb_dim]
        h = self.norm1(x)
        scale, shift = self.film(temb).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift

        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.norm2(h)
        h = self.fc2(h)
        return x + h


class MLPEncoder(nn.Module):
    """
    MLP encoder for vector inputs, preserving the SAME I/O structure:
      forward(h, x) -> [B, rank, out_dim]

    Changes vs your original:
    - Fix B bug when h is None
    - Use residual blocks with LayerNorm + SiLU for stability
    - Inject time via FiLM in every block (better conditioning than concat-once)
    - IMPORTANT: do NOT zero-init the final head (U would be ~0 => dead gradients for U^T U losses)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rank: int = 32,
        out_dim: int = 28,
        time_embedding: bool = False,
        temb_dim: int | None = None,
        dropout: float = 0.0,
        head_init_std: float = 1e-3,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        if input_dim is None:
            raise ValueError(
                "MLPEncoder requires a concrete input_dim (flattened features, do NOT include h)"
            )
        assert num_layers >= 1, "num_layers must be >= 1"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rank = rank
        self.out_dim = out_dim
        self.time_embedding = time_embedding

        # ----- input projection -----
        if self.time_embedding:
            self.in_proj = nn.Linear(input_dim, hidden_dim)
            self.temb_dim = temb_dim if temb_dim is not None else hidden_dim // 4
            # timestep embedding -> temb_dim, then a small MLP
            self.time_mlp = nn.Sequential(
                nn.Linear(self.temb_dim, self.temb_dim),
                nn.SiLU(),
                nn.Linear(self.temb_dim, self.temb_dim),
            )
        else:
            # if no time embedding, still condition blocks on scalar h
            self.in_proj = nn.Linear(input_dim + 1, hidden_dim)
            self.temb_dim = 1
            self.time_mlp = None

        self.act = nn.SiLU()

        # ----- residual FiLM blocks -----
        self.blocks = nn.ModuleList(
            [
                ResBlockFiLM(
                    hidden_dim,
                    self.temb_dim,
                    dropout=dropout,
                    zero_init_residual=zero_init_residual,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

        # ----- final projection: MUST NOT be zero-init for low-rank factor outputs -----
        self.proj = nn.Linear(hidden_dim, rank * out_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, x):
        # x: [B, N, 2] or [B, features]
        if x.dim() > 2:
            x_flat = x.reshape(x.shape[0], -1)
        else:
            x_flat = x

        B = x_flat.shape[0]  # FIX: define B before using it

        # ensure h tensor of shape [B, 1]
        if h is None:
            h_feat = torch.zeros(B, 1, device=x_flat.device, dtype=x_flat.dtype)
        else:
            h_feat = h.reshape(-1, 1).to(device=x_flat.device, dtype=x_flat.dtype)

        # build conditioning embedding
        if self.time_embedding:
            # timestep_embedding returns [B, temb_dim]
            temb = timestep_embedding(h_feat.reshape(-1), self.temb_dim).reshape(B, -1)
            temb = self.time_mlp(temb)
            inp = x_flat
        else:
            # no time embedding: concatenate h and also pass scalar h_feat to blocks as temb
            temb = h_feat  # [B,1]
            inp = torch.cat([x_flat, h_feat], dim=1)

        # trunk
        z = self.act(self.in_proj(inp))
        for blk in self.blocks:
            z = blk(z, temb)

        z = self.out_norm(z)
        out = self.proj(z)
        return out.view(B, self.rank, self.out_dim)
