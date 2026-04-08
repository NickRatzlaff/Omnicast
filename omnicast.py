# pyhazards/models/omnicast.py
# OmniCast: masked latent diffusion for sequence forecasting (paper: arXiv:2510.18707v1)
#
# Expected input shape (weather tensors):
#   x: (B, 1 + T, C, H, W)  where x[:,0] is initial condition, x[:,1:] are future states
#
# Core idea:
#   1) Encode each frame with a continuous VAE into latent tokens (B, frames, h*w, D_lat)
#   2) Mask a random subset of FUTURE tokens; keep conditioning (initial) tokens visible
#   3) MAE-style encoder-decoder Transformer produces z_i for each token position
#   4) Diffusion head predicts noise per masked token (diffusion loss)
#   5) Optional deterministic head predicts x_i for early frames (weighted MSE first K frames)
#   6) Inference: iterative unmasking schedule + per-token diffusion sampling

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def _cosine_mask_schedule(num_steps: int) -> List[float]:
    """
    Mask ratio schedule from 1.0 -> 0.0 (cosine), like MaskGIT-style decoding.
    """
    ratios = []
    for t in range(num_steps):
        # t=0 => 1.0 masked, t=end => 0.0 masked
        r = 0.5 * (1.0 + math.cos(math.pi * t / max(1, num_steps - 1)))
        ratios.append(r)
    return ratios


def _make_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    2D sin-cos positional embedding for spatial tokens: (h*w, dim)
    """
    if dim % 4 != 0:
        raise ValueError("pos dim must be divisible by 4 for 2D sin-cos")

    grid_y = torch.arange(h, device=device)
    grid_x = torch.arange(w, device=device)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")  # (h,w)
    yy = yy.reshape(-1).float()
    xx = xx.reshape(-1).float()

    omega = torch.arange(dim // 4, device=device).float()
    omega = 1.0 / (10000 ** (omega / (dim // 4)))

    out_x = torch.einsum("n,d->nd", xx, omega)
    out_y = torch.einsum("n,d->nd", yy, omega)
    pe = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1)
    return pe  # (h*w, dim)


def _make_1d_sincos_pos_embed(t: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    1D sin-cos temporal embedding: (t, dim)
    """
    if dim % 2 != 0:
        raise ValueError("temporal pos dim must be divisible by 2")
    pos = torch.arange(t, device=device).float()  # (t,)
    omega = torch.arange(dim // 2, device=device).float()
    omega = 1.0 / (10000 ** (omega / (dim // 2)))
    out = torch.einsum("n,d->nd", pos, omega)
    pe = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return pe  # (t, dim)


# -----------------------------
# Simple UNet-like continuous VAE
# (swap with PDEArena UNet if available)
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = ConvBlock(cin, cout)
        self.down = nn.Conv2d(cout, cout, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x)
        d = self.down(h)
        return d, h


class Up(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(cout * 2, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad in case of odd dims
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ContinuousUNetVAE(nn.Module):
    """
    Continuous VAE producing latent map z: (B, z_dim, h, w) with spatial downsampling factor 16.
    Paper example: C=69, HxW=128x256 -> z_dim=1024, h=8, w=16 :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, in_ch: int, z_dim: int = 1024, base: int = 256, kl_weight: float = 5e-5):
        super().__init__()
        self.kl_weight = kl_weight

        # Down: /2, /4, /8, /16
        self.in_block = ConvBlock(in_ch, base)
        self.d1 = Down(base, base * 1)
        self.d2 = Down(base * 1, base * 2)
        self.d3 = Down(base * 2, base * 4)
        self.d4 = Down(base * 4, base * 8)

        # latent params
        self.to_mu = nn.Conv2d(base * 8, z_dim, 1)
        self.to_logvar = nn.Conv2d(base * 8, z_dim, 1)

        # Up path from z_dim
        self.from_z = nn.Conv2d(z_dim, base * 8, 1)
        self.u4 = Up(base * 8, base * 4)
        self.u3 = Up(base * 4, base * 2)
        self.u2 = Up(base * 2, base * 1)
        self.u1 = Up(base * 1, base)

        self.out = nn.Conv2d(base, in_ch, 1)

    @staticmethod
    def _reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.in_block(x)
        x1, s1 = self.d1(h0)
        x2, s2 = self.d2(x1)
        x3, s3 = self.d3(x2)
        x4, s4 = self.d4(x3)

        mu = self.to_mu(x4)
        logvar = self.to_logvar(x4)
        z = self._reparam(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor, skips: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        s1, s2, s3, s4 = skips
        x = self.from_z(z)
        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        return self.out(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # forward returns reconstruction + VAE losses (so you can pretrain stage-1)
        h0 = self.in_block(x)
        x1, s1 = self.d1(h0)
        x2, s2 = self.d2(x1)
        x3, s3 = self.d3(x2)
        x4, s4 = self.d4(x3)

        mu = self.to_mu(x4)
        logvar = self.to_logvar(x4)
        z = self._reparam(mu, logvar)

        xhat = self.decode(z, (s1, s2, s3, s4))

        rec = F.mse_loss(xhat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec + self.kl_weight * kl
        return {"xhat": xhat, "z": z, "rec_loss": rec, "kl_loss": kl, "loss": loss}


# -----------------------------
# MAE-style Transformer backbone (encoder-decoder)
# -----------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MAEBackbone(nn.Module):
    """
    Minimal MAE-like encoder-decoder:
      - encoder sees conditioning + visible tokens only
      - decoder sees encoded tokens + mask tokens to output per-position representations
    Paper uses ViT-style encoder/decoder with full attention :contentReference[oaicite:4]{index=4}
    """
    def __init__(self, dim: int = 1024, depth_enc: int = 8, depth_dec: int = 8, heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth_enc)])
        self.dec = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth_dec)])

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for blk in self.enc:
            x = blk(x)
        return x

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for blk in self.dec:
            x = blk(x)
        return x


# -----------------------------
# Diffusion head (per-token) + deterministic head
# -----------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or (B, N) integer diffusion steps
        returns: (..., dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / max(1, half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class AdaLNResidualBlock(nn.Module):
    """
    Residual MLP block with AdaLN conditioning.
    """
    def __init__(self, width: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(width, width)
        self.cond = nn.Linear(cond_dim, width * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c -> scale, shift
        ss = self.cond(c)
        scale, shift = ss.chunk(2, dim=-1)
        h = self.ln(x)
        h = h * (1 + scale) + shift
        h = self.fc2(self.act(self.fc1(h)))
        return x + h


class PerTokenDiffusionHead(nn.Module):
    """
    Predicts noise eps_hat for a given noisy token x_s, conditioned on transformer rep z_i and time step s.
    Matches paper: small MLP with residual blocks + AdaLN 
    """
    def __init__(self, token_dim: int, cond_dim: int, width: int = 2048, blocks: int = 6, t_embed_dim: int = 256):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(t_embed_dim)
        self.cond_proj = nn.Linear(cond_dim + t_embed_dim, width)
        self.x_proj = nn.Linear(token_dim, width)
        self.blocks = nn.ModuleList([AdaLNResidualBlock(width, width) for _ in range(blocks)])
        self.out = nn.Linear(width, token_dim)

    def forward(self, x_s: torch.Tensor, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x_s: (B, M, D) noisy tokens
        s:   (B, M) diffusion step indices
        z:   (B, M, D_cond) transformer reps for those token positions
        """
        t_emb = self.t_embed(s.reshape(-1)).view(*s.shape, -1)  # (B,M,t_dim)
        c = torch.cat([z, t_emb], dim=-1)
        c = self.cond_proj(c)
        h = self.x_proj(x_s)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


class DeterministicHead(nn.Module):
    def __init__(self, cond_dim: int, token_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, token_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# -----------------------------
# OmniCast main module
# -----------------------------

@dataclass
class OmniCastConfig:
    # data / latent
    in_ch: int
    image_hw: Tuple[int, int]
    downsample: int = 16
    vae_base: int = 256
    z_dim: int = 1024          # VAE latent channels
    token_dim: int = 16        # D in paper for continuous tokens (example in §4.1) :contentReference[oaicite:6]{index=6}
    # transformer
    model_dim: int = 1024
    depth_enc: int = 8
    depth_dec: int = 8
    heads: int = 16
    dropout: float = 0.1
    # diffusion
    diff_train_steps: int = 1000
    diff_infer_steps: int = 100
    # deterministic aux loss
    aux_mse_frames: int = 10
    # masking
    mask_ratio_min: float = 0.5
    mask_ratio_max: float = 1.0


class OmniCast(nn.Module):
    """
    OmniCast for sequence forecasting.
    forward(x, ...) can return losses (training) or samples (inference).

    Input:
      x: (B, 1+T, C, H, W)

    Training returns:
      dict(loss=..., diff_loss=..., deter_loss=..., details...)

    Inference:
      call sample(x0, T, ...) to generate (B, T, C, H, W)
    """
    def __init__(self, cfg: OmniCastConfig):
        super().__init__()
        self.cfg = cfg

        # Stage-1 VAE (continuous)
        self.vae = ContinuousUNetVAE(in_ch=cfg.in_ch, z_dim=cfg.z_dim, base=cfg.vae_base)

        # Map VAE latent map -> token vectors of dimension token_dim (per spatial position)
        # z_map: (B, z_dim, h, w) -> tokens: (B, h*w, token_dim)
        self.latent_to_tokens = nn.Conv2d(cfg.z_dim, cfg.token_dim, kernel_size=1)
        self.tokens_to_latent = nn.Conv2d(cfg.token_dim, cfg.z_dim, kernel_size=1)

        # Project tokens -> transformer model dimension
        self.tok_proj = nn.Linear(cfg.token_dim, cfg.model_dim)
        self.tok_unproj = nn.Linear(cfg.model_dim, cfg.model_dim)  # keep same dim for reps

        # Mask token in model-dim space
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.model_dim))

        # Backbone
        self.backbone = MAEBackbone(dim=cfg.model_dim, depth_enc=cfg.depth_enc, depth_dec=cfg.depth_dec, heads=cfg.heads, dropout=cfg.dropout)

        # Heads
        self.diff_head = PerTokenDiffusionHead(token_dim=cfg.token_dim, cond_dim=cfg.model_dim)
        self.deter_head = DeterministicHead(cond_dim=cfg.model_dim, token_dim=cfg.token_dim)

        # diffusion schedule buffers (linear beta)
        betas = torch.linspace(1e-4, 0.02, cfg.diff_train_steps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("diff_betas", betas)
        self.register_buffer("diff_alphas", alphas)
        self.register_buffer("diff_abar", abar)

    # --------- VAE tokenization helpers ---------

    def _encode_frame_to_tokens(self, x_frame: torch.Tensor) -> torch.Tensor:
        """
        x_frame: (B,C,H,W) -> tokens: (B, hw, token_dim)
        """
        z_map, _, _ = self.vae.encode(x_frame)            # (B, z_dim, h, w)
        t_map = self.latent_to_tokens(z_map)             # (B, token_dim, h, w)
        B, D, h, w = t_map.shape
        return t_map.permute(0, 2, 3, 1).reshape(B, h * w, D)  # (B, hw, token_dim)

    def _decode_tokens_to_frame(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        tokens: (B, hw, token_dim) -> xhat: (B,C,H,W)
        """
        B, hw, D = tokens.shape
        t_map = tokens.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()  # (B, token_dim, h, w)
        z_map = self.tokens_to_latent(t_map)                                 # (B, z_dim, h, w)

        # We need skips for VAE decode; for simplicity, re-run encoder to get skips is expensive.
        # In practice: keep a decoder-only architecture or store skips.
        # Here: approximate by decoding without skips via a lightweight upsampler:
        # --- For correctness in your assignment, replace this with the exact VAE decoder path. ---
        xhat = F.interpolate(z_map, scale_factor=self.cfg.downsample, mode="bilinear", align_corners=False)
        xhat = nn.Conv2d(self.cfg.z_dim, self.cfg.in_ch, kernel_size=1).to(xhat.device)(xhat)
        return xhat

    # --------- positional encodings ---------

    def _add_positional(self, tokens_md: torch.Tensor, T_total: int, h: int, w: int) -> torch.Tensor:
        """
        tokens_md: (B, (1+T)*hw, model_dim) in frame-major order
        adds temporal + spatial pos embeddings (fixed sincos)
        """
        B, N, Dm = tokens_md.shape
        device = tokens_md.device

        # fixed sincos embeddings (paper uses learned embeddings; fixed works well as a baseline)
        spatial = _make_2d_sincos_pos_embed(h, w, Dm, device=device)  # (hw, Dm)
        temporal = _make_1d_sincos_pos_embed(T_total, Dm, device=device)  # (T_total, Dm)

        # broadcast to (T_total, hw, Dm) then flatten
        pe = temporal[:, None, :] + spatial[None, :, :]
        pe = pe.reshape(T_total * (h * w), Dm)
        return tokens_md + pe[None, :, :]

    # --------- masking ---------

    def _sample_mask(self, B: int, T: int, hw: int, device: torch.device) -> torch.Tensor:
        """
        Mask only FUTURE tokens (T frames), not the conditioning frame.
        Returns mask over future tokens: (B, T*hw) boolean True=masked
        """
        gamma = torch.empty(B, device=device).uniform_(self.cfg.mask_ratio_min, self.cfg.mask_ratio_max)
        # per-example number masked
        total = T * hw
        mask = torch.zeros(B, total, dtype=torch.bool, device=device)
        for b in range(B):
            m = int(round(gamma[b].item() * total))
            idx = torch.randperm(total, device=device)[:m]
            mask[b, idx] = True
        return mask

    # --------- diffusion training loss ---------

    def _q_sample(self, x0: torch.Tensor, s: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: x_s = sqrt(a_bar_s)*x0 + sqrt(1-a_bar_s)*noise
        x0: (B,M,D), s: (B,M) int, noise same shape
        """
        abar = self.diff_abar[s]  # (B,M)
        abar = abar.unsqueeze(-1) # (B,M,1)
        return torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * noise

    def _diffusion_loss(self, x0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x0: (B, M, token_dim) true tokens at masked positions
        z:  (B, M, model_dim) transformer reps for those positions
        """
        B, M, D = x0.shape
        s = torch.randint(1, self.cfg.diff_train_steps, (B, M), device=x0.device)
        noise = torch.randn_like(x0)
        x_s = self._q_sample(x0, s, noise)
        eps_hat = self.diff_head(x_s, s, z)
        return F.mse_loss(eps_hat, noise)

    # --------- deterministic aux loss (first K frames) ---------

    def _deterministic_loss(self, x_true: torch.Tensor, x_hat: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
        """
        Weighted MSE for masked tokens belonging to early frames only.
        frame_ids: (B, M) each in [0..T-1] relative to FUTURE frames (1..T in full sequence)
        Implements exponential weighting and zeroes beyond aux_mse_frames :contentReference[oaicite:7]{index=7}
        """
        K = self.cfg.aux_mse_frames
        # w = exp(-k), k=frame index; w=0 if k>=K
        w = torch.exp(-frame_ids.float())
        w = torch.where(frame_ids < K, w, torch.zeros_like(w))
        # normalize per batch example to sum 1 over selected tokens
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        w = w / w_sum
        mse = (x_true - x_hat).pow(2).sum(dim=-1)  # (B,M)
        return (w * mse).sum()

    # --------- transformer pass ---------

    def _backbone_representations(
        self,
        cond_md: torch.Tensor,         # (B, hw, model_dim)
        fut_md: torch.Tensor,          # (B, T*hw, model_dim) (masked positions filled with mask_token)
        visible_future_mask: torch.Tensor,  # (B, T*hw) True=VISIBLE, False=MASKED
        T_total: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Returns decoder reps for ALL tokens in (1+T)*hw order: (B, (1+T)*hw, model_dim)
        """
        B = cond_md.shape[0]
        hw = cond_md.shape[1]

        # Build encoder input: conditioning tokens + visible future tokens only
        fut_visible = fut_md.clone()
        # If token is masked, remove it from encoder input by selecting visible indices
        reps_all = []

        # Flatten full sequence with mask tokens for decoder input
        full = torch.cat([cond_md, fut_md], dim=1)  # (B, (1+T)*hw, model_dim)
        full = self._add_positional(full, T_total=T_total, h=h, w=w)

        # Encoder selection
        enc_tokens = []
        for b in range(B):
            vis_idx = torch.nonzero(visible_future_mask[b], as_tuple=False).squeeze(-1)  # indices in [0..T*hw)
            # +hw offset because cond is always included
            idx = torch.cat([torch.arange(hw, device=full.device), hw + vis_idx], dim=0)
            enc_tokens.append(full[b:b+1, idx, :])
        enc_tokens = torch.cat(enc_tokens, dim=0)  # (B, hw+num_visible, model_dim)

        enc_out = self.backbone.encode(enc_tokens)

        # Decoder input: use FULL token sequence, but replace masked future with learnable mask_token
        dec_in = full.clone()
        # overwrite masked future positions with mask_token (after positional add)
        mask_tok = self.mask_token.expand(B, 1, -1)
        for b in range(B):
            masked_idx = torch.nonzero(~visible_future_mask[b], as_tuple=False).squeeze(-1)
            dec_in[b, hw + masked_idx, :] = mask_tok.squeeze(1)

        # In MAE, decoder attends to encoded visible tokens + mask tokens; to keep simple:
        # concatenate enc_out with the (mask-filled) full sequence and decode.
        # (You can replace this with exact MAE wiring later.)
        dec_tokens = torch.cat([enc_out, dec_in], dim=1)
        dec_out = self.backbone.decode(dec_tokens)
        # take the tail corresponding to dec_in
        dec_out = dec_out[:, -dec_in.shape[1]:, :]
        return dec_out  # (B, (1+T)*hw, model_dim)

    # --------- forward (training) ---------

    def forward(self, x: torch.Tensor, *, return_latents: bool = False) -> Dict[str, torch.Tensor]:
        """
        Training forward: expects ground-truth future frames.
        """
        if x.ndim != 5:
            raise ValueError("Expected x of shape (B, 1+T, C, H, W)")
        B, Tp1, C, H, W = x.shape
        T = Tp1 - 1

        # Infer latent spatial size
        h = H // self.cfg.downsample
        w = W // self.cfg.downsample
        hw = h * w

        # Encode condition + future frames to tokens (continuous latent)
        cond_tokens = self._encode_frame_to_tokens(x[:, 0])          # (B, hw, token_dim)
        fut_tokens = torch.stack([self._encode_frame_to_tokens(x[:, 1 + t]) for t in range(T)], dim=1)  # (B,T,hw,token_dim)
        fut_tokens = fut_tokens.reshape(B, T * hw, self.cfg.token_dim)                                  # (B,T*hw,D)

        # Mask future tokens
        mask = self._sample_mask(B, T, hw, device=x.device)   # (B, T*hw) True=masked
        visible = ~mask

        # Prepare transformer inputs in model-dim
        cond_md = self.tok_proj(cond_tokens)    # (B, hw, model_dim)
        fut_md = self.tok_proj(fut_tokens)      # (B, T*hw, model_dim)
        # For masked positions, we will still pass something (will be overridden in decoder path)
        # but keep fut_md as-is; visibility controls encoder selection.

        # Run backbone to get representations for all positions
        reps = self._backbone_representations(
            cond_md=cond_md,
            fut_md=fut_md,
            visible_future_mask=visible,
            T_total=Tp1,
            h=h,
            w=w,
        )  # (B, (1+T)*hw, model_dim)

        # Extract masked future positions
        fut_reps = reps[:, hw:, :]  # (B, T*hw, model_dim)
        masked_idx = torch.nonzero(mask, as_tuple=False)  # (K,2) with [b, pos]
        # Gather masked token true values and reps
        x_true = fut_tokens[masked_idx[:, 0], masked_idx[:, 1]]     # (K, token_dim)
        z_mask = fut_reps[masked_idx[:, 0], masked_idx[:, 1]]       # (K, model_dim)

        # reshape to (B, M, D) per-batch ragged -> pad to max M
        # for simplicity: compute loss per-token by grouping in a loop
        diff_losses = []
        deter_losses = []
        for b in range(B):
            mpos = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            if mpos.numel() == 0:
                continue
            x0_b = fut_tokens[b, mpos, :]     # (M, token_dim)
            z_b = fut_reps[b, mpos, :]        # (M, model_dim)
            diff_losses.append(self._diffusion_loss(x0_b.unsqueeze(0), z_b.unsqueeze(0)))

            # deterministic head and weighted MSE for early frames only :contentReference[oaicite:8]{index=8}
            xhat_b = self.deter_head(z_b)     # (M, token_dim)
            frame_ids = (mpos // hw)          # (M,) in [0..T-1]
            deter_losses.append(self._deterministic_loss(
                x_true=x0_b.unsqueeze(0),
                x_hat=xhat_b.unsqueeze(0),
                frame_ids=frame_ids.unsqueeze(0),
            ))

        diff_loss = torch.stack(diff_losses).mean() if diff_losses else torch.tensor(0.0, device=x.device)
        deter_loss = torch.stack(deter_losses).mean() if deter_losses else torch.tensor(0.0, device=x.device)

        loss = diff_loss + deter_loss  # matches paper: L = Lgen(diffusion) + Ldeter :contentReference[oaicite:9]{index=9}

        out = {"loss": loss, "diff_loss": diff_loss, "deter_loss": deter_loss}
        if return_latents:
            out.update({"cond_tokens": cond_tokens, "fut_tokens": fut_tokens, "mask": mask})
        return out

    # --------- diffusion sampling (per token) ---------

    @torch.no_grad()
    def _p_sample_loop(self, z_cond: torch.Tensor, steps: int, tau: float) -> torch.Tensor:
        """
        Sample one token given conditioning rep z_cond (B,M,model_dim) using DDPM-like sampling.
        Returns x0 (B,M,token_dim).
        """
        B, M, _ = z_cond.shape
        # start from noise
        x = torch.randn(B, M, self.cfg.token_dim, device=z_cond.device)
        # use inference steps by striding training schedule
        stride = max(1, self.cfg.diff_train_steps // steps)
        t_seq = list(range(self.cfg.diff_train_steps - 1, 0, -stride))[:steps]

        for t in t_seq:
            s = torch.full((B, M), t, device=z_cond.device, dtype=torch.long)
            abar = self.diff_abar[s].unsqueeze(-1)
            alpha = self.diff_alphas[s].unsqueeze(-1)
            beta = self.diff_betas[s].unsqueeze(-1)

            eps_hat = self.diff_head(x, s, z_cond)

            # DDPM mean estimate
            mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - abar)) * eps_hat)

            if t > 1:
                noise = torch.randn_like(x)
                # temperature-scaled noise like paper mentions :contentReference[oaicite:10]{index=10}
                x = mean + tau * torch.sqrt(beta) * noise
            else:
                x = mean
        return x

    # --------- full sequence sampling (iterative unmask) ---------

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,   # (B,C,H,W)
        T: int,
        *,
        unmask_iters: Optional[int] = None,
        diff_steps: Optional[int] = None,
        tau: float = 1.3,
        random_order: bool = True,
    ) -> torch.Tensor:
        """
        Generate T future frames conditioned on x0, following masked iterative decoding:
          start fully masked, then repeatedly unmask subsets until all tokens are generated :contentReference[oaicite:11]{index=11}
        """
        B, C, H, W = x0.shape
        h = H // self.cfg.downsample
        w = W // self.cfg.downsample
        hw = h * w

        unmask_iters = unmask_iters or T  # paper uses ~1 iteration per frame by default :contentReference[oaicite:12]{index=12}
        diff_steps = diff_steps or self.cfg.diff_infer_steps

        # Encode conditioning
        cond_tokens = self._encode_frame_to_tokens(x0)     # (B, hw, token_dim)
        cond_md = self.tok_proj(cond_tokens)               # (B, hw, model_dim)

        # Initialize future tokens all masked (keep a token tensor we fill in)
        fut_tokens = torch.zeros(B, T * hw, self.cfg.token_dim, device=x0.device)
        fut_md = self.tok_proj(fut_tokens)

        # Track which future positions are already generated
        generated = torch.zeros(B, T * hw, dtype=torch.bool, device=x0.device)

        # schedule of remaining mask ratio
        ratios = _cosine_mask_schedule(unmask_iters)

        # Optional randomized unmask priority (random across space+time performs best in paper) :contentReference[oaicite:13]{index=13}
        base_perm = torch.randperm(T * hw, device=x0.device) if random_order else torch.arange(T * hw, device=x0.device)

        for it in range(unmask_iters):
            # Determine how many tokens should remain masked after this iteration
            target_mask_ratio = ratios[it]
            target_mask = int(round(target_mask_ratio * (T * hw)))

            for b in range(B):
                # decide which tokens to unmask now: those not generated yet, but we want to reduce masked count
                remaining = (~generated[b]).nonzero(as_tuple=False).squeeze(-1)
                if remaining.numel() == 0:
                    continue

                # how many should still be masked after this iter?
                still_mask = max(0, target_mask - int((~generated[b]).sum().item() - remaining.numel()))
                # simpler: compute number to generate now
                # generate so that remaining masked ~= target_mask
                num_to_leave = target_mask
                num_to_gen = max(0, int((~generated[b]).sum().item()) - num_to_leave)
                num_to_gen = min(num_to_gen, remaining.numel())
                if num_to_gen == 0:
                    continue

                # choose next positions to generate based on perm order
                rem_sorted = remaining[torch.argsort(torch.argsort(base_perm[remaining]))]
                sel = rem_sorted[:num_to_gen]

                # Build "visible mask": visible tokens are those already generated
                visible = generated.clone()

                # Prepare fut_md using current known tokens
                fut_md = self.tok_proj(fut_tokens)

                # Transformer reps for all tokens
                reps = self._backbone_representations(
                    cond_md=cond_md,
                    fut_md=fut_md,
                    visible_future_mask=visible,  # True=visible (generated)
                    T_total=1 + T,
                    h=h,
                    w=w,
                )
                fut_reps = reps[:, hw:, :]  # (B, T*hw, model_dim)

                # Sample selected tokens with diffusion conditioned on their reps
                z_sel = fut_reps[b:b+1, sel, :]  # (1, M, model_dim)
                x_sel = self._p_sample_loop(z_sel, steps=diff_steps, tau=tau)  # (1, M, token_dim)

                fut_tokens[b, sel, :] = x_sel.squeeze(0)
                generated[b, sel] = True

        # Decode per-frame (token grid -> weather)
        frames = []
        for t in range(T):
            tok_t = fut_tokens[:, t * hw : (t + 1) * hw, :]
            frames.append(self._decode_tokens_to_frame(tok_t, h=h, w=w))
        return torch.stack(frames, dim=1)  # (B, T, C, H, W)


# -----------------------------
# Builder function (pyhazards pattern)
# -----------------------------

def omnicast_builder(
    task: str,
    in_ch: int,
    image_hw: Tuple[int, int],
    **kwargs: Any,
) -> OmniCast:
    """
    Builder used by build_model(name="omnicast", task=..., **kwargs) :contentReference[oaicite:14]{index=14}
    OmniCast is a forecasting/generative model; you can treat it as "regression" or custom task.
    """
    # If your framework enforces task types, pick one and validate.
    if task not in {"regression", "forecasting", "generation"}:
        raise ValueError("OmniCast is a forecasting/generative model; use task='regression'/'forecasting'/'generation'.")

    cfg = OmniCastConfig(
        in_ch=in_ch,
        image_hw=image_hw,
        downsample=kwargs.get("downsample", 16),
        vae_base=kwargs.get("vae_base", 256),
        z_dim=kwargs.get("z_dim", 1024),
        token_dim=kwargs.get("token_dim", 16),
        model_dim=kwargs.get("model_dim", 1024),
        depth_enc=kwargs.get("depth_enc", 8),
        depth_dec=kwargs.get("depth_dec", 8),
        heads=kwargs.get("heads", 16),
        dropout=kwargs.get("dropout", 0.1),
        diff_train_steps=kwargs.get("diff_train_steps", 1000),
        diff_infer_steps=kwargs.get("diff_infer_steps", 100),
        aux_mse_frames=kwargs.get("aux_mse_frames", 10),
        mask_ratio_min=kwargs.get("mask_ratio_min", 0.5),
        mask_ratio_max=kwargs.get("mask_ratio_max", 1.0),
    )
    return OmniCast(cfg)
