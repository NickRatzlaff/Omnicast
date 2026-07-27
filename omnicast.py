"""
OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales

PyTorch implementation based on:
  Nguyen et al., "OmniCast: A Masked Latent Diffusion Model for Weather Forecasting
  Across Time Scales", NeurIPS 2025.

Architecture overview:
  1. Continuous VAE (UNet-based) encodes weather states X ∈ R^{V×H×W} into spatial
     latent grids z ∈ R^{D×h×w}, with 16× spatial downsampling.
  2. MAE-style encoder-decoder Transformer operates over conditioning + future tokens.
     During training, 50–100% of future tokens are masked. The encoder sees only
     visible tokens; the decoder sees encoded tokens + learnable [MASK] tokens.
  3. Per-token diffusion head (small MLP with AdaLN) predicts noise for masked tokens,
     conditioned on the Transformer output z_i.
  4. Auxiliary deterministic MLP head predicts x̂_i from z_i with exponentially
     decaying MSE loss on the first 10 future frames.
  5. Iterative unmasking at inference: cosine schedule reduces mask ratio from 1→0
     over T iterations with random ordering; diffusion sampling generates each token.

Dependencies: torch, einops (pip install torch einops)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# 1. VAE — UNet-based continuous VAE for weather data embedding
# =============================================================================

class ResBlock(nn.Module):
    """PDEArena/LDM-style residual block used by the paper's VAE."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class DownBlock(nn.Module):
    """One VAE encoder resolution level, optionally followed by 2x downsampling."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int = 2,
        dropout: float = 0.0,
        downsample: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, dropout))
        self.downsample = (
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=0)
            if downsample else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(F.pad(x, (0, 1, 0, 1)))
        return x


class UpBlock(nn.Module):
    """One latent-only VAE decoder level, optionally followed by 2x upsampling."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int = 3,
        dropout: float = 0.0,
        upsample: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, dropout))
        self.upsample = nn.Conv2d(out_ch, out_ch, 3, padding=1) if upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.upsample is not None:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.upsample(x)
        return x


class WeatherVAE(nn.Module):
    """
    Continuous VAE for weather data.
    
    Encodes X ∈ R^{V×H×W} → z ∈ R^{D×h×w} with 16× spatial downsampling.
    Paper: UNet with base_channels=256, channel_mults=[1,2,4,4,8], D=1024,
           KL weight=5e-5.
    
    Args:
        in_channels: Number of weather variables V (default 69).
        latent_dim: Dimension D of each latent token (default 1024).
        base_channels: Base hidden dimension of UNet (default 256).
        channel_mults: Channel multipliers for each UNet level.
        n_blocks: Number of ResBlocks per level (default 2).
        dropout: Dropout rate (default 0.0).
    """
    def __init__(
        self,
        in_channels: int = 69,
        latent_dim: int = 1024,
        base_channels: int = 256,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4, 8),
        n_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        channels = [base_channels * m for m in channel_mults]

        # Encoder: one level per multiplier and four downsampling operations.
        self.enc_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            self.enc_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    n_blocks,
                    dropout,
                    downsample=i < len(channels) - 1,
                )
            )
            in_ch = out_ch
        self.enc_mid = nn.Sequential(
            ResBlock(channels[-1], channels[-1], dropout),
            ResBlock(channels[-1], channels[-1], dropout),
        )
        self.enc_out = nn.Sequential(
            nn.GroupNorm(min(32, channels[-1]), channels[-1], eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(channels[-1], 2 * latent_dim, 3, padding=1),
        )

        # Generated weather states must be decoded from z alone, without skips.
        self.dec_in = nn.Conv2d(latent_dim, channels[-1], 3, padding=1)
        self.dec_mid = nn.Sequential(
            ResBlock(channels[-1], channels[-1], dropout),
            ResBlock(channels[-1], channels[-1], dropout),
        )
        self.dec_blocks = nn.ModuleList()
        in_ch = channels[-1]
        for i in reversed(range(len(channels))):
            out_ch = channels[i]
            self.dec_blocks.append(
                UpBlock(
                    in_ch,
                    out_ch,
                    n_blocks=n_blocks + 1,
                    dropout=dropout,
                    upsample=i > 0,
                )
            )
            in_ch = out_ch
        self.dec_out = nn.Sequential(
            nn.GroupNorm(min(32, channels[0]), channels[0], eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode weather state to latent distribution parameters."""
        h = self.enc_in(x)
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_mid(h)
        mu, logvar = self.enc_out(h).chunk(2, dim=1)
        return mu, logvar.clamp(-30.0, 20.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * epsilon."""
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tokens back to weather space."""
        h = self.dec_in(z)
        h = self.dec_mid(h)
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sample → decode.
        Returns: (reconstruction, z, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar

    def vae_loss(
        self, x: torch.Tensor, recon: torch.Tensor,
        mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 5e-5
    ) -> torch.Tensor:
        """Reconstruction MSE + KL divergence."""
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss


# =============================================================================
# 2. Transformer — MAE-style encoder-decoder backbone
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""
    def __init__(self, dim: int, n_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_dropout(self.proj(x))


class TransformerBlock(nn.Module):
    """Transformer block: LayerNorm → MHSA → LayerNorm → MLP."""
    def __init__(self, dim: int, n_heads: int = 16, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAETransformer(nn.Module):
    """
    MAE-style encoder-decoder Transformer backbone.
    
    The encoder processes only visible (conditioning + unmasked) tokens.
    The decoder receives encoded visible tokens + learnable [MASK] tokens
    at masked positions, then outputs z_i for each position.
    
    Paper: 16 encoder layers, 16 decoder layers, dim=1024, 16 heads, dropout=0.1.
    
    Args:
        latent_dim: Dimension D of VAE latent tokens (input to the Transformer).
        hidden_dim: Transformer hidden dimension (default 1024).
        encoder_depth: Number of encoder layers (default 16).
        decoder_depth: Number of decoder layers (default 16).
        n_heads: Number of attention heads (default 16).
        mlp_ratio: MLP expansion ratio (default 4.0).
        dropout: Dropout rate (default 0.1).
        max_spatial: Maximum number of spatial tokens per frame (h×w).
        max_frames: Maximum number of frames (conditioning + future).
    """
    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 1024,
        encoder_depth: int = 16,
        decoder_depth: int = 16,
        n_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_spatial: int = 128,       # h*w tokens per frame
        max_frames: int = 45,         # 1 IC + 44 future
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project VAE tokens to transformer hidden dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # The paper adds spatial + temporal embeddings before both MAE stages.
        self.encoder_spatial_embed = nn.Parameter(
            torch.randn(1, max_spatial, hidden_dim) * 0.02
        )
        self.encoder_temporal_embed = nn.Parameter(
            torch.randn(1, max_frames, hidden_dim) * 0.02
        )
        self.decoder_spatial_embed = nn.Parameter(
            torch.randn(1, max_spatial, hidden_dim) * 0.02
        )
        self.decoder_temporal_embed = nn.Parameter(
            torch.randn(1, max_frames, hidden_dim) * 0.02
        )

        # Encoder (processes visible tokens only)
        self.encoder = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])

        # Decoder (processes full sequence with mask tokens)
        self.decoder = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])

        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def get_pos_embed(
        self,
        n_spatial: int,
        n_frames: int,
        *,
        decoder: bool = False,
    ) -> torch.Tensor:
        """
        Compute combined positional embedding for a sequence of frames.
        Returns shape (1, n_frames * n_spatial, hidden_dim).
        """
        spatial_table = (
            self.decoder_spatial_embed if decoder else self.encoder_spatial_embed
        )
        temporal_table = (
            self.decoder_temporal_embed if decoder else self.encoder_temporal_embed
        )
        if n_spatial > spatial_table.shape[1] or n_frames > temporal_table.shape[1]:
            raise ValueError(
                f"requested {n_frames} frames x {n_spatial} spatial tokens, "
                "which exceeds the configured positional embeddings"
            )
        spatial = spatial_table[:, :n_spatial, :]
        temporal = temporal_table[:, :n_frames, :]
        # Broadcast add: each frame's tokens get temporal + spatial
        # temporal: (1, T, 1, D), spatial: (1, 1, hw, D) → (1, T, hw, D)
        pos = temporal.unsqueeze(2) + spatial.unsqueeze(1)
        return pos.reshape(1, n_frames * n_spatial, self.hidden_dim)

    def forward(
        self,
        cond_tokens: torch.Tensor,      # (B, hw, D_latent) — conditioning frame
        future_tokens: torch.Tensor,     # (B, T*hw, D_latent) — future tokens (with masks filled)
        mask: torch.Tensor,              # (B, T*hw) — True where masked
        n_spatial: int,                  # h*w
    ) -> torch.Tensor:
        """
        Forward pass through MAE encoder-decoder.
        
        Args:
            cond_tokens: Latent tokens from the initial condition frame.
            future_tokens: Future latent tokens (masked positions will be replaced by [MASK]).
            mask: Boolean mask, True at positions to predict.
            n_spatial: Number of spatial tokens per frame (h*w).
            
        Returns:
            z_i: Transformer output at masked positions only, shape (B, n_masked, D_latent).
        """
        B = cond_tokens.shape[0]
        n_cond = cond_tokens.shape[1]          # h*w
        n_future = future_tokens.shape[1]      # T*h*w
        n_frames = 1 + n_future // n_spatial   # total frames

        # Project to hidden dim
        cond_h = self.input_proj(cond_tokens)     # (B, hw, H)
        future_h = self.input_proj(future_tokens) # (B, T*hw, H)

        # Replace masked positions with [MASK] token
        mask_expanded = mask.unsqueeze(-1).expand_as(future_h)  # (B, T*hw, H)
        future_h = torch.where(mask_expanded, self.mask_token.expand(B, n_future, -1), future_h)

        # Concatenate conditioning + future
        full_seq = torch.cat([cond_h, future_h], dim=1)  # (B, (1+T)*hw, H)

        # Add the encoder's spatial-temporal positional embeddings.
        encoder_pos = self.get_pos_embed(n_spatial, n_frames)
        full_seq = full_seq + encoder_pos[:, :full_seq.shape[1], :]

        # --- Encoder: only visible tokens ---
        # Visible = conditioning tokens + unmasked future tokens
        cond_visible = torch.ones(B, n_cond, dtype=torch.bool, device=mask.device)
        full_visible = torch.cat([cond_visible, ~mask], dim=1)  # (B, (1+T)*hw)

        visible_counts = full_visible.sum(dim=1)
        if not torch.equal(visible_counts, visible_counts[:1].expand_as(visible_counts)):
            raise ValueError("each batch item must contain the same number of masked tokens")
        n_visible = int(visible_counts[0].item())
        visible_indices = full_visible.nonzero(as_tuple=False)[:, 1].reshape(B, n_visible)
        visible_tokens = torch.gather(
            full_seq,
            1,
            visible_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        )

        for block in self.encoder:
            visible_tokens = block(visible_tokens)
        visible_tokens = self.encoder_norm(visible_tokens)

        # --- Decoder: full sequence with encoded visible + mask tokens ---
        # Reconstruct the full sequence, then add decoder positional embeddings.
        decoder_seq = self.mask_token.expand(B, full_seq.shape[1], -1).clone()
        decoder_seq.scatter_(
            1,
            visible_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
            visible_tokens,
        )
        decoder_pos = self.get_pos_embed(n_spatial, n_frames, decoder=True)
        decoder_seq = decoder_seq + decoder_pos[:, :full_seq.shape[1], :]

        for block in self.decoder:
            decoder_seq = block(decoder_seq)
        decoder_seq = self.decoder_norm(decoder_seq)

        # Extract outputs at masked positions (future tokens only)
        # Offset by n_cond since mask is over future tokens
        n_masked = int(mask.sum(dim=1)[0].item())
        masked_indices = mask.nonzero(as_tuple=False)[:, 1].reshape(B, n_masked)
        future_output = decoder_seq[:, n_cond:, :]
        z_masked = torch.gather(
            future_output,
            1,
            masked_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        )

        # Project back to latent dim
        z_masked = self.output_proj(z_masked)  # (B, n_masked, D_latent)
        return z_masked


# =============================================================================
# 3. Per-token Diffusion Head — Small MLP with AdaLN
# =============================================================================

class AdaLNResidualBlock(nn.Module):
    """
    Residual block with Adaptive Layer Normalization (AdaLN).
    
    The conditioning vector z_i modulates the LayerNorm via learned scale/shift,
    combined with the diffusion timestep embedding.
    Paper: 6 blocks, width 2048, with AdaLN conditioning from z_i + timestep.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        # AdaLN: project conditioning to scale and shift
        self.adaLN = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) input features
            cond: (B, cond_dim) conditioning = z_i + time_embed
        """
        scale_shift = self.adaLN(cond)  # (B, 2*D)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1 + scale) + shift
        h = F.silu(self.linear1(h))
        h = self.linear2(h)
        return x + h


class DiffusionHead(nn.Module):
    """
    Per-token diffusion denoising network ε_θ(x^s, s, z_i).
    
    Small MLP with AdaLN conditioning from z_i, predicting noise ε
    from noisy token x^s at diffusion step s.
    
    Paper: 6 residual blocks, width 2048, AdaLN with z_i + timestep embedding.
    
    Args:
        token_dim: Dimension D of latent tokens.
        hidden_dim: Width of the MLP blocks (default 2048).
        n_blocks: Number of residual blocks (default 6).
        max_timesteps: Maximum diffusion timesteps for embedding (default 1000).
    """
    def __init__(
        self,
        token_dim: int = 1024,
        hidden_dim: int = 2048,
        n_blocks: int = 6,
        max_timesteps: int = 1000,
    ):
        super().__init__()
        self.token_dim = token_dim

        # Sinusoidal timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project z_i conditioning to hidden_dim
        self.cond_proj = nn.Linear(token_dim, hidden_dim)

        # Project noisy input x^s to hidden_dim
        self.input_proj = nn.Linear(token_dim, hidden_dim)

        # Residual blocks with AdaLN
        self.blocks = nn.ModuleList([
            AdaLNResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])

        # Output projection: predict noise ε
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,   # (B, D) noisy tokens at step s
        timestep: torch.Tensor,   # (B,) diffusion timestep indices
        z_cond: torch.Tensor,     # (B, D) conditioning from transformer
    ) -> torch.Tensor:
        """Predict noise ε from noisy token, timestep, and transformer conditioning."""
        t_emb = self.time_embed(timestep)           # (B, hidden)
        c_emb = self.cond_proj(z_cond)              # (B, hidden)
        cond = t_emb + c_emb                        # Combined conditioning

        h = self.input_proj(x_noisy)                # (B, hidden)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)                   # (B, D) predicted noise


class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        if half == 0:
            return t.float().unsqueeze(-1)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        return F.pad(embedding, (0, self.dim - embedding.shape[-1]))


# =============================================================================
# 4. Deterministic Prediction Head
# =============================================================================

class DeterministicHead(nn.Module):
    """
    Deterministic MLP head g_θ that predicts x̂_i directly from z_i.
    Used for the auxiliary MSE loss on the first 10 future frames.
    """
    def __init__(self, latent_dim: int = 1024, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =============================================================================
# 5. Diffusion Utilities — linear noise schedule, forward/reverse process
# =============================================================================

class DiffusionSchedule(nn.Module):
    """
    Linear noise schedule for the diffusion process.
    
    Paper: 1000 training steps (linear schedule), resampled to 100 at inference.
    Forward: x_s = sqrt(α_s) * x + sqrt(1 - α_s) * ε
    Reverse: DDPM-style with temperature τ scaling.
    """
    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise to x_0 at timestep t."""
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x_0 + sqrt_1_ab * noise

    def p_sample_step(
        self,
        model: DiffusionHead,
        x_s: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: x_{s-1} from x_s.
        Eq. (3): x_{s-1} = (1/√α_s)(x_s - (1-α_s)/√(1-ᾱ_s) * ε_θ) + τ * σ_s * δ
        """
        s = t[0].item()
        alpha_s = self.alphas[s]
        alpha_bar_s = self.alpha_bar[s]
        alpha_bar_prev_s = self.alpha_bar_prev[s]
        beta_s = self.betas[s]

        # Predict noise
        eps_pred = model(x_s, t, z_cond)

        # Compute mean
        coeff1 = 1.0 / alpha_s.sqrt()
        coeff2 = beta_s / (1 - alpha_bar_s).sqrt()
        mean = coeff1 * (x_s - coeff2 * eps_pred)

        if s > 0:
            posterior_variance = beta_s * (1 - alpha_bar_prev_s) / (1 - alpha_bar_s)
            sigma = posterior_variance.clamp_min(0).sqrt()
            noise = torch.randn_like(x_s)
            return mean + tau * sigma * noise
        return mean

    def sample(
        self,
        model: DiffusionHead,
        z_cond: torch.Tensor,
        n_steps: int = 100,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """
        Full reverse diffusion sampling from Gaussian noise.
        
        Args:
            model: Denoising network ε_θ.
            z_cond: (B, D) conditioning vector from transformer.
            n_steps: Number of inference diffusion steps.
            tau: Temperature for sampling diversity.
        """
        device = z_cond.device
        B, D = z_cond.shape

        if not 1 <= n_steps <= self.n_steps:
            raise ValueError(f"n_steps must be in [1, {self.n_steps}], got {n_steps}")

        # Choose training-time noise levels, then recompute transition betas for
        # those levels. Merely skipping DDPM steps does not produce a valid
        # 100-step reverse process.
        if n_steps == 1:
            step_indices = torch.tensor([self.n_steps - 1], device=device)
        else:
            step_indices = torch.linspace(
                0, self.n_steps - 1, n_steps, device=device
            ).round().long()
        selected_alpha_bar = self.alpha_bar[step_indices]
        selected_alpha_bar_prev = F.pad(
            selected_alpha_bar[:-1], (1, 0), value=1.0
        )
        spaced_alphas = selected_alpha_bar / selected_alpha_bar_prev
        spaced_betas = 1.0 - spaced_alphas

        x = torch.randn(B, D, device=device, dtype=z_cond.dtype)
        for j in reversed(range(n_steps)):
            t = step_indices[j].expand(B)
            eps_pred = model(x, t, z_cond)
            alpha = spaced_alphas[j]
            beta = spaced_betas[j]
            alpha_bar = selected_alpha_bar[j]
            alpha_bar_prev = selected_alpha_bar_prev[j]
            mean = alpha.rsqrt() * (
                x - beta / (1 - alpha_bar).sqrt() * eps_pred
            )
            if j > 0:
                posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                x = mean + tau * posterior_variance.clamp_min(0).sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


# =============================================================================
# 6. OmniCast — Full model combining all components
# =============================================================================

class OmniCast(nn.Module):
    """
    OmniCast: Masked Latent Diffusion Model for Weather Forecasting.
    
    Combines:
      - WeatherVAE: Encodes/decodes weather data to/from continuous latent space.
      - MAETransformer: Encoder-decoder backbone operating on latent tokens.
      - DiffusionHead: Per-token noise prediction conditioned on transformer output.
      - DeterministicHead: Auxiliary MSE prediction for early frames.
    
    Args:
        vae: Pre-trained WeatherVAE model (frozen during stage 2 training).
        latent_dim: Dimension D of latent tokens.
        hidden_dim: Transformer hidden dimension.
        n_spatial: Spatial tokens per frame (h*w).
        n_future_frames: Number of future frames T.
        n_det_frames: Number of frames for deterministic loss (default 10).
        diffusion_train_steps: Training diffusion steps (default 1000).
        diffusion_infer_steps: Inference diffusion steps (default 100).
    """
    def __init__(
        self,
        vae: WeatherVAE,
        latent_dim: int = 1024,
        hidden_dim: int = 1024,
        n_spatial: int = 128,       # h*w = 8*16
        n_future_frames: int = 44,
        n_det_frames: int = 10,
        diffusion_train_steps: int = 1000,
        diffusion_infer_steps: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.vae = vae
        self.latent_dim = latent_dim
        self.n_spatial = n_spatial
        self.n_future_frames = n_future_frames
        self.n_det_frames = n_det_frames
        self.diffusion_train_steps = diffusion_train_steps
        self.diffusion_infer_steps = diffusion_infer_steps

        # Freeze VAE
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

        # Transformer backbone (use smaller depth/heads for demo; paper uses 16/16)
        enc_depth = kwargs.get('encoder_depth', 16)
        dec_depth = kwargs.get('decoder_depth', 16)
        n_heads = kwargs.get('n_heads', 16)
        mlp_ratio = kwargs.get('mlp_ratio', 4.0)
        dropout = kwargs.get('dropout', 0.1)
        self.transformer = MAETransformer(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            encoder_depth=enc_depth,
            decoder_depth=dec_depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_spatial=n_spatial,
            max_frames=1 + n_future_frames,
        )

        # Diffusion head and schedule
        self.diffusion_head = DiffusionHead(token_dim=latent_dim, hidden_dim=min(2048, hidden_dim * 2))
        self.diffusion_schedule = DiffusionSchedule(n_steps=diffusion_train_steps)

        # Deterministic head
        self.det_head = DeterministicHead(latent_dim=latent_dim)

        # Precompute deterministic loss weights (exponentially decaying)
        self._precompute_det_weights()

    def train(self, mode: bool = True) -> "OmniCast":
        """Keep the stage-one VAE frozen and in evaluation mode."""
        super().train(mode)
        self.vae.eval()
        return self

    def _precompute_det_weights(self):
        """Compute per-token exponentially decaying weights for deterministic loss."""
        weights = []
        for t in range(self.n_future_frames):
            if t < self.n_det_frames:
                weights.extend([math.exp(-t)] * self.n_spatial)
            else:
                weights.extend([0.0] * self.n_spatial)
        w = torch.tensor(weights)
        # Normalize non-zero weights to sum to 1
        nonzero_mask = w > 0
        if nonzero_mask.any():
            w[nonzero_mask] = w[nonzero_mask] / w[nonzero_mask].sum()
        self.register_buffer('det_weights', w)

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of weather frames using the frozen VAE.
        
        Args:
            frames: (B, T, V, H, W) weather states.
        Returns:
            tokens: (B, T, h*w, D) latent tokens.
        """
        B, T, V, H, W = frames.shape
        flat = frames.reshape(B * T, V, H, W)
        mu, _ = self.vae.encode(flat)           # (B*T, D, h, w)
        tokens = rearrange(mu, '(b t) d h w -> b t (h w) d', b=B, t=T)
        return tokens

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Decode latent tokens back to weather states.
        
        Args:
            tokens: (B, T, h*w, D) latent tokens.
            h, w: Spatial dimensions of latent grid.
        Returns:
            frames: (B, T, V, H, W) decoded weather states.
        """
        B, T, N, D = tokens.shape
        latent = rearrange(tokens, 'b t (h w) d -> (b t) d h w', h=h, w=w)
        recon = self.vae.decode(latent)       # (B*T, V, H', W')
        return rearrange(recon, '(b t) v h w -> b t v h w', b=B, t=T)

    def training_step(
        self,
        cond_tokens: torch.Tensor,   # (B, hw, D)
        future_tokens: torch.Tensor, # (B, T*hw, D)
    ) -> dict:
        """
        Single training step: mask future tokens, compute diffusion + deterministic loss.
        
        Args:
            cond_tokens: Latent tokens of the initial condition frame.
            future_tokens: Latent tokens of T future frames, flattened.
            
        Returns:
            Dictionary with 'loss', 'diff_loss', 'det_loss'.
        """
        B, N, D = future_tokens.shape
        device = future_tokens.device
        self.diffusion_schedule.to(device)

        # Sample gamma ~ U[0.5, 1.0]. Each example receives an independent
        # random order across both space and time, with a common packed length.
        gamma = torch.empty((), device=device).uniform_(0.5, 1.0)
        n_masked = max(1, int(torch.floor(gamma * N).item()))
        mask_order = torch.rand(B, N, device=device).argsort(dim=1)
        masked_indices = mask_order[:, :n_masked]
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_indices, True)

        # --- Forward through transformer backbone ---
        # z_i at masked positions
        z_masked = self.transformer(cond_tokens, future_tokens, mask, self.n_spatial)
        # z_masked: (B, n_masked, D)

        # Ground truth tokens at masked positions
        # MAETransformer returns masked positions in increasing sequence order.
        masked_indices = mask.nonzero(as_tuple=False)[:, 1].reshape(B, n_masked)
        x_target = torch.gather(
            future_tokens,
            1,
            masked_indices.unsqueeze(-1).expand(-1, -1, D),
        )

        # --- Diffusion loss on masked tokens ---
        # Flatten for per-token diffusion: (B*n_masked, D)
        z_flat = z_masked.reshape(-1, D)
        x_flat = x_target.reshape(-1, D)

        # Sample random diffusion timesteps
        t = torch.randint(0, self.diffusion_train_steps, (z_flat.shape[0],), device=device)
        noise = torch.randn_like(x_flat)
        x_noisy = self.diffusion_schedule.q_sample(x_flat, t, noise)

        # Predict noise
        noise_pred = self.diffusion_head(x_noisy, t, z_flat)
        diff_loss = F.mse_loss(noise_pred, noise)

        # --- Deterministic loss on masked tokens (first 10 frames only) ---
        x_det_pred = self.det_head(z_masked)  # (B, n_masked, D)
        det_residual = (x_det_pred - x_target) ** 2  # (B, n_masked, D)

        # Get weights for the masked positions
        weights = self.det_weights[masked_indices]
        # w(i) is normalized once across the first ten frames, as in Appendix
        # A.2; it is not renormalized to whichever positions happened to mask.
        det_loss = (det_residual.mean(dim=-1) * weights).sum(dim=1).mean()

        loss = diff_loss + det_loss

        return {
            'loss': loss,
            'diff_loss': diff_loss.detach(),
            'det_loss': det_loss.detach(),
        }

    @torch.no_grad()
    def generate(
        self,
        cond_tokens: torch.Tensor,  # (B, hw, D)
        n_iterations: Optional[int] = None,
        tau: float = 1.3,
        diffusion_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Iterative unmasking inference.
        
        Starts with all future tokens masked. At each iteration:
          1. Run transformer on conditioning + current state (masked/unmasked).
          2. Select a subset of masked tokens to unmask (cosine schedule + random order).
          3. Sample those tokens via diffusion conditioned on z_i.
        
        Args:
            cond_tokens: (B, hw, D) initial condition latent tokens.
            n_iterations: Number of unmasking iterations (default = n_future_frames).
            tau: Diffusion sampling temperature.
            diffusion_steps: Override inference diffusion steps.
            
        Returns:
            future_tokens: (B, T*hw, D) generated future latent tokens.
        """
        device = cond_tokens.device
        B = cond_tokens.shape[0]
        N = self.n_future_frames * self.n_spatial
        D = self.latent_dim
        n_iter = n_iterations or self.n_future_frames
        n_diff_steps = diffusion_steps or self.diffusion_infer_steps

        self.diffusion_schedule.to(device)

        # Initialize all future tokens as zeros (will be replaced by generated tokens)
        future_tokens = torch.zeros(B, N, D, device=device)
        # Every ensemble member receives an independent random unmasking order.
        is_masked = torch.ones(B, N, dtype=torch.bool, device=device)

        # Cosine schedule: determines how many tokens to unmask at each iteration
        # γ(t) goes from 1.0 → 0.0 following cos schedule
        for i in range(n_iter):
            # Target remaining tokens under the MaskGIT cosine schedule.
            ratio_next = math.cos(math.pi / 2 * (i + 1) / n_iter)
            n_remaining = int(is_masked[0].sum().item())
            if n_remaining == 0:
                break
            target_remaining = int(math.floor(ratio_next * N))
            n_to_unmask = n_remaining - target_remaining
            if n_to_unmask <= 0:
                continue

            # Random positions are selected separately for each batch member.
            random_scores = torch.rand(B, N, device=device)
            random_scores = random_scores.masked_fill(~is_masked, float("inf"))
            selected = random_scores.argsort(dim=1)[:, :n_to_unmask]

            # Forward through transformer
            z_masked = self.transformer(
                cond_tokens, future_tokens, is_masked, self.n_spatial
            )
            # z_masked: (B, n_currently_masked, D)

            # Transformer outputs masked positions in increasing token order.
            rank_within_mask = is_masked.long().cumsum(dim=1) - 1
            selected_positions = torch.gather(
                rank_within_mask, 1, selected
            )
            z_selected = torch.gather(
                z_masked,
                1,
                selected_positions.unsqueeze(-1).expand(-1, -1, D),
            )

            # Diffusion sampling for each selected token
            z_flat = z_selected.reshape(B * n_to_unmask, D)
            sampled = self.diffusion_schedule.sample(
                self.diffusion_head, z_flat, n_steps=n_diff_steps, tau=tau
            )
            sampled = sampled.reshape(B, n_to_unmask, D)

            # Place generated tokens and reveal them for the next iteration.
            future_tokens.scatter_(
                1, selected.unsqueeze(-1).expand(-1, -1, D), sampled
            )
            is_masked.scatter_(1, selected, False)

        return future_tokens

    @torch.no_grad()
    def generate_ensemble(
        self,
        cond_tokens: torch.Tensor,
        ensemble_size: int = 50,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate independent ensemble members by replicating each initial state.

        Returns:
            Tensor of shape (B, ensemble_size, T*hw, D).
        """
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        B = cond_tokens.shape[0]
        replicated = cond_tokens.repeat_interleave(ensemble_size, dim=0)
        samples = self.generate(replicated, **generate_kwargs)
        return samples.reshape(B, ensemble_size, samples.shape[1], samples.shape[2])

    @torch.no_grad()
    def generate_autoregressive(
        self,
        cond_tokens: torch.Tensor,
        n_frames: int,
        *,
        tau: float = 1.0,
        n_iterations: int = 1,
        diffusion_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Medium-range rollout used in Section 5.2 of the paper.

        The medium-range model predicts two 12-hour frames per call. The most
        recent predicted frame becomes the next initial condition until the
        requested horizon is reached.

        Returns:
            Tensor of shape (B, n_frames, hw, D).
        """
        if n_frames < 1:
            raise ValueError("n_frames must be positive")

        chunks = []
        current_condition = cond_tokens
        generated_count = 0
        while generated_count < n_frames:
            window = self.generate(
                current_condition,
                n_iterations=n_iterations,
                tau=tau,
                diffusion_steps=diffusion_steps,
            )
            window = window.reshape(
                window.shape[0],
                self.n_future_frames,
                self.n_spatial,
                self.latent_dim,
            )
            n_take = min(self.n_future_frames, n_frames - generated_count)
            chunks.append(window[:, :n_take])
            generated_count += n_take
            current_condition = window[:, -1]

        return torch.cat(chunks, dim=1)


# =============================================================================
# 7. Example Training and Inference Usage
# =============================================================================

def example_training():
    """
    Demonstrates the two-stage training pipeline of OmniCast.
    Uses small dimensions for illustration — scale up for real usage.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Hyperparameters (small for demo; paper uses much larger) ---
    V = 8          # weather variables (paper: 69)
    H, W = 32, 64  # spatial resolution (paper: 128×256 for S2S)
    D = 64          # latent token dim (paper: 1024)
    T = 6           # future frames (paper: 44)
    channel_mults = (1, 2, 4, 4)  # paper: (1, 2, 4, 4, 8) → 16× downsampling
    downsample_factor = 2 ** (len(channel_mults) - 1)  # 3 DownBlocks → 8×
    h = H // downsample_factor  # latent height (4)
    w = W // downsample_factor  # latent width  (8)
    n_spatial = h * w            # 32 tokens per frame

    # =============================
    # Stage 1: Train the VAE
    # =============================
    print("\n=== Stage 1: Training VAE ===")
    vae = WeatherVAE(
        in_channels=V, latent_dim=D, base_channels=32,
        channel_mults=channel_mults, n_blocks=1
    ).to(device)

    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=1e-5)

    for epoch in range(3):
        # Synthetic batch: (B, V, H, W)
        x = torch.randn(4, V, H, W, device=device)
        recon, z, mu, logvar = vae(x)
        loss = vae.vae_loss(x, recon, mu, logvar, kl_weight=5e-5)
        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()
        print(f"  VAE Epoch {epoch}: loss={loss.item():.4f}, latent shape={z.shape} -> h={h}, w={w}")

    # =============================
    # Stage 2: Train OmniCast
    # =============================
    print("\n=== Stage 2: Training OmniCast (Transformer + Diffusion) ===")
    vae.eval()

    model = OmniCast(
        vae=vae,
        latent_dim=D,
        hidden_dim=128,     # paper: 1024
        n_spatial=n_spatial,
        n_future_frames=T,
        n_det_frames=3,     # paper: 10
        diffusion_train_steps=100,  # paper: 1000
        diffusion_infer_steps=20,   # paper: 100
        encoder_depth=2,    # paper: 16 (small for demo)
        decoder_depth=2,    # paper: 16 (small for demo)
        n_heads=4,          # paper: 16 (small for demo)
    ).to(device)

    # Only optimize transformer, diffusion head, and deterministic head
    trainable_params = (
        list(model.transformer.parameters()) +
        list(model.diffusion_head.parameters()) +
        list(model.det_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, betas=(0.9, 0.95), weight_decay=1e-5)

    for step in range(5):
        # Synthetic sequence: (B, 1+T, V, H, W) — 1 IC frame + T future frames
        frames = torch.randn(4, 1 + T, V, H, W, device=device)

        # Encode all frames to latent tokens
        all_tokens = model.encode_frames(frames)  # (B, 1+T, h*w, D)
        cond_tokens = all_tokens[:, 0, :, :]       # (B, h*w, D)
        future_tokens = rearrange(all_tokens[:, 1:, :, :], 'b t n d -> b (t n) d')  # (B, T*h*w, D)

        # Training step
        losses = model.training_step(cond_tokens, future_tokens)
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()

        print(f"  Step {step}: total={losses['loss'].item():.4f}, "
              f"diff={losses['diff_loss'].item():.4f}, det={losses['det_loss'].item():.4f}")

    return model, h, w


def example_inference(model: OmniCast, h: int, w: int):
    """
    Demonstrates the iterative unmasking inference process.
    """
    device = next(model.parameters()).device
    V = model.vae.enc_in.in_channels
    n_downsamples = sum(block.downsample is not None for block in model.vae.enc_blocks)
    H = h * (2 ** n_downsamples)
    W = w * (2 ** n_downsamples)

    print("\n=== Inference: Iterative Unmasking ===")

    # Encode one initial condition, then independently sample two members.
    x0 = torch.randn(1, 1, V, H, W, device=device)
    ic_tokens = model.encode_frames(x0)[:, 0, :, :]  # (1, h*w, D)

    # Generate future tokens via iterative unmasking
    model.eval()
    future_tokens = model.generate_ensemble(
        cond_tokens=ic_tokens,
        ensemble_size=2,
        n_iterations=model.n_future_frames,  # 1 iteration per frame
        tau=1.3,                              # temperature from paper
        diffusion_steps=20,
    )[0]
    print(f"  Generated future tokens shape: {future_tokens.shape}")
    # future_tokens: (2, T*h*w, D)

    # Reshape and decode
    T = model.n_future_frames
    future_tokens = rearrange(future_tokens, 'b (t n) d -> b t n d', t=T, n=model.n_spatial)
    forecasts = model.decode_tokens(future_tokens, h, w)
    print(f"  Decoded forecasts shape: {forecasts.shape}")
    # forecasts: (2, T, V, H, W) — 2 ensemble members × T future frames

    return forecasts


if __name__ == '__main__':
    model, h, w = example_training()
    forecasts = example_inference(model, h, w)
    print("\nDone! OmniCast training and inference completed successfully.")
