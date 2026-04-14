from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _largest_divisor_at_most(value: int, cap: int) -> int:
    for candidate in range(min(value, cap), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=_largest_divisor_at_most(channels, 32),
        num_channels=channels,
        eps=1e-6,
        affine=True,
    )


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _cosine_mask_schedule(num_steps: int) -> List[float]:
    if num_steps <= 1:
        return [0.0]
    return [
        0.5 * (1.0 + math.cos(math.pi * step / (num_steps - 1)))
        for step in range(num_steps)
    ]


def _default_latent_dim(image_hw: Tuple[int, int]) -> int:
    if image_hw == (721, 1440):
        return 256
    return 1024


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.temb_proj: Optional[nn.Linear]
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        else:
            self.temb_proj = None

        self.norm2 = _normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                self.nin_shortcut = None
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                self.conv_shortcut = None
        else:
            self.conv_shortcut = None
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        h = _swish(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(_swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = _swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.conv_shortcut is not None:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int,
        dropout: float,
        resamp_with_conv: bool,
        in_channels: int,
        z_channels: int,
        double_z: bool,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hs.append(self.down[i_level].block[i_block](hs[-1], temb))
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = _swish(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        resolution: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int,
        dropout: float,
        resamp_with_conv: bool,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)

        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = _swish(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            zeros = torch.zeros_like(self.mean)
            self.std = zeros
            self.var = zeros

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0],
                dtype=self.mean.dtype,
                device=self.mean.device,
            )
        if other is None:
            return 0.5 * torch.mean(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=(1, 2, 3),
            )
        return 0.5 * torch.mean(
            (self.mean - other.mean).pow(2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=(1, 2, 3),
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class ContinuousUNetVAE(nn.Module):
    """
    Continuous KL-VAE that matches the paper's PDEArena-style UNet defaults.
    """

    def __init__(
        self,
        *,
        img_size: Tuple[int, int],
        in_channels: int,
        out_channels: Optional[int] = None,
        base_channels: int = 256,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4, 8),
        num_res_blocks: int = 2,
        z_channels: int = 1024,
        double_z: bool = True,
        dropout: float = 0.0,
        kl_weight: float = 5e-5,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.base_channels = base_channels
        self.channel_multipliers = tuple(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.double_z = double_z
        self.dropout = dropout
        self.kl_weight = kl_weight
        self.downsample_factor = 2 ** (len(self.channel_multipliers) - 1)

        height, width = img_size
        needs_height_adjustment = height % self.downsample_factor != 0
        if needs_height_adjustment and (height - 1) % self.downsample_factor != 0:
            raise ValueError(
                "image height must be divisible by the VAE downsample factor, "
                "or become divisible after a single-row reduction"
            )
        if width % self.downsample_factor != 0:
            raise ValueError("image width must be divisible by the VAE downsample factor")

        self._height_adjust = needs_height_adjustment
        if self._height_adjust:
            self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(2, 1), stride=1, padding=0)
            self.post_conv = nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=(2, 1), stride=1, padding=0)
        else:
            self.pre_conv = nn.Identity()
            self.post_conv = nn.Identity()

        adjusted_height = height - 1 if self._height_adjust else height
        resolution = width

        self.encoder = Encoder(
            ch=base_channels,
            ch_mult=self.channel_multipliers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=double_z,
        )
        self.decoder = Decoder(
            ch=base_channels,
            out_ch=self.out_channels,
            resolution=resolution,
            ch_mult=self.channel_multipliers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            resamp_with_conv=True,
            z_channels=z_channels,
        )

        latent_height = adjusted_height // self.downsample_factor
        latent_width = width // self.downsample_factor
        self._latent_hw = (latent_height, latent_width)

    @property
    def latent_hw(self) -> Tuple[int, int]:
        return self._latent_hw

    def encode_distribution(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        x = self.pre_conv(x)
        moments = self.encoder(x)
        return DiagonalGaussianDistribution(moments)

    def encode(
        self,
        x: torch.Tensor,
        *,
        sample: bool = False,
        return_posterior: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode_distribution(x)
        z = posterior.sample() if sample else posterior.mode()
        if return_posterior:
            return z, posterior
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)
        return self.post_conv(x_hat)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        posterior = self.encode_distribution(x)
        z = posterior.sample()
        x_hat = self.decode(z)
        rec = F.mse_loss(x_hat, x)
        kl = posterior.kl().mean()
        loss = rec + self.kl_weight * kl
        return {
            "xhat": x_hat,
            "z": z,
            "rec_loss": rec,
            "kl_loss": kl,
            "loss": loss,
        }


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(dim=dim, heads=heads, dropout=dropout)
            for _ in range(depth)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x


class MAEBackbone(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 1024,
        depth_enc: int = 16,
        depth_dec: int = 16,
        heads: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TransformerStack(dim=dim, depth=depth_enc, heads=heads, dropout=dropout)
        self.decoder = TransformerStack(dim=dim, depth=depth_dec, heads=heads, dropout=dropout)

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(tokens, key_padding_mask=key_padding_mask)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(tokens)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLNResidualBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.activation = nn.SiLU()
        self.cond = nn.Linear(cond_dim, width * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        h = self.fc2(self.activation(self.fc1(h)))
        return x + h


class PerTokenDiffusionHead(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        cond_dim: int,
        width: int = 2048,
        blocks: int = 6,
        t_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(t_embed_dim)
        self.cond_proj = nn.Linear(cond_dim + t_embed_dim, width)
        self.x_proj = nn.Linear(token_dim, width)
        self.blocks = nn.ModuleList(
            AdaLNResidualBlock(width=width, cond_dim=width)
            for _ in range(blocks)
        )
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, token_dim)

    def forward(
        self,
        x_s: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.t_embed(s.reshape(-1)).view(*s.shape, -1)
        cond = self.cond_proj(torch.cat([z, t_emb], dim=-1))
        h = self.x_proj(x_s)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(self.out_norm(h))


class DeterministicHead(nn.Module):
    def __init__(self, cond_dim: int, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, token_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass
class OmniCastConfig:
    in_ch: int
    image_hw: Tuple[int, int]
    max_future_frames: int = 44

    downsample: int = 16
    vae_base: int = 256
    vae_ch_mult: Tuple[int, ...] = (1, 2, 4, 4, 8)
    vae_num_res_blocks: int = 2
    vae_dropout: float = 0.0
    z_dim: Optional[int] = None

    model_dim: int = 1024
    depth_enc: int = 16
    depth_dec: int = 16
    heads: int = 16
    dropout: float = 0.1

    diff_train_steps: int = 1000
    diff_infer_steps: int = 100
    diff_width: int = 2048
    diff_blocks: int = 6

    aux_mse_frames: int = 10
    mask_ratio_min: float = 0.5
    mask_ratio_max: float = 1.0
    sample_temperature: float = 1.3

    token_dim: Optional[int] = None

    def __post_init__(self) -> None:
        expected_downsample = 2 ** (len(self.vae_ch_mult) - 1)
        if self.downsample != expected_downsample:
            raise ValueError(
                f"downsample={self.downsample} must match the VAE architecture's factor of {expected_downsample}"
            )

        if self.z_dim is None:
            self.z_dim = _default_latent_dim(self.image_hw)

        if self.token_dim is None:
            self.token_dim = self.z_dim

        if self.token_dim != self.z_dim:
            raise ValueError(
                "OmniCast uses the VAE latent channels directly as token vectors; "
                "token_dim must equal z_dim"
            )

        if self.max_future_frames < 1:
            raise ValueError("max_future_frames must be at least 1")

        if not (0.0 <= self.mask_ratio_min <= self.mask_ratio_max <= 1.0):
            raise ValueError("mask ratios must satisfy 0 <= min <= max <= 1")


class OmniCast(nn.Module):
    """
    Paper-faithful OmniCast implementation:
      - continuous per-frame KL-VAE
      - MAE-style encoder-decoder Transformer with learned spatial + temporal positions
      - per-token diffusion head for masked future tokens
      - weighted deterministic auxiliary loss on the first K future frames
    """

    def __init__(self, cfg: OmniCastConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.vae = ContinuousUNetVAE(
            img_size=cfg.image_hw,
            in_channels=cfg.in_ch,
            out_channels=cfg.in_ch,
            base_channels=cfg.vae_base,
            channel_multipliers=cfg.vae_ch_mult,
            num_res_blocks=cfg.vae_num_res_blocks,
            z_channels=cfg.z_dim,
            dropout=cfg.vae_dropout,
            kl_weight=5e-5,
        )

        self.latent_h, self.latent_w = self.vae.latent_hw
        self.spatial_tokens = self.latent_h * self.latent_w
        self.token_dim = cfg.token_dim

        if self.token_dim == cfg.model_dim:
            self.token_embed: nn.Module = nn.Identity()
        else:
            self.token_embed = nn.Linear(self.token_dim, cfg.model_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(cfg.max_future_frames + 1, cfg.model_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(self.spatial_tokens, cfg.model_dim))

        self.backbone = MAEBackbone(
            dim=cfg.model_dim,
            depth_enc=cfg.depth_enc,
            depth_dec=cfg.depth_dec,
            heads=cfg.heads,
            dropout=cfg.dropout,
        )
        self.diff_head = PerTokenDiffusionHead(
            token_dim=self.token_dim,
            cond_dim=cfg.model_dim,
            width=cfg.diff_width,
            blocks=cfg.diff_blocks,
        )
        self.deter_head = DeterministicHead(cond_dim=cfg.model_dim, token_dim=self.token_dim)

        betas = torch.linspace(1e-4, 0.02, cfg.diff_train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("diff_betas", betas)
        self.register_buffer("diff_alphas", alphas)
        self.register_buffer("diff_abar", abar)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)
        nn.init.normal_(self.spatial_pos, std=0.02)

    def _validate_inputs(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError("expected x to have shape (B, 1 + T, C, H, W)")
        _, _, channels, height, width = x.shape
        if channels != self.cfg.in_ch:
            raise ValueError(f"expected {self.cfg.in_ch} channels, got {channels}")
        if (height, width) != self.cfg.image_hw:
            raise ValueError(f"expected image size {self.cfg.image_hw}, got {(height, width)}")

    def _encode_frame_to_tokens(self, x_frame: torch.Tensor) -> torch.Tensor:
        z_map = self.vae.encode(x_frame, sample=False)
        return z_map.flatten(2).transpose(1, 2).contiguous()

    def _decode_tokens_to_frame(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, token_dim = tokens.shape
        if num_tokens != self.spatial_tokens:
            raise ValueError(
                f"expected {self.spatial_tokens} spatial tokens per frame, got {num_tokens}"
            )
        if token_dim != self.token_dim:
            raise ValueError(f"expected token dim {self.token_dim}, got {token_dim}")
        z_map = tokens.transpose(1, 2).reshape(batch, self.token_dim, self.latent_h, self.latent_w)
        return self.vae.decode(z_map)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embed(tokens)

    def _full_positional_embeddings(
        self,
        *,
        total_frames: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if total_frames > self.temporal_pos.shape[0]:
            raise ValueError(
                f"requested {total_frames} frames, but the model was configured for at most {self.temporal_pos.shape[0] - 1} future frames"
            )
        temporal = self.temporal_pos[:total_frames].to(dtype=dtype)
        spatial = self.spatial_pos.to(dtype=dtype)
        pos = temporal[:, None, :] + spatial[None, :, :]
        return pos.reshape(total_frames * self.spatial_tokens, self.cfg.model_dim)

    def _mask_token_embeddings(
        self,
        *,
        batch_size: int,
        total_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask_embed = self._embed_tokens(self.mask_token).to(device=device, dtype=dtype)
        return mask_embed.expand(batch_size, total_tokens, -1).clone()

    def _sample_mask(self, batch_size: int, future_frames: int, device: torch.device) -> torch.Tensor:
        total_future_tokens = future_frames * self.spatial_tokens
        mask = torch.zeros(batch_size, total_future_tokens, dtype=torch.bool, device=device)
        ratios = torch.empty(batch_size, device=device).uniform_(
            self.cfg.mask_ratio_min,
            self.cfg.mask_ratio_max,
        )
        for batch_idx in range(batch_size):
            num_masked = int(round(ratios[batch_idx].item() * total_future_tokens))
            if num_masked == 0:
                continue
            masked_positions = torch.randperm(total_future_tokens, device=device)[:num_masked]
            mask[batch_idx, masked_positions] = True
        return mask

    def _mae_representations(
        self,
        *,
        cond_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        visible_future_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = cond_tokens.shape[0]
        total_future_tokens = future_tokens.shape[1]
        total_frames = 1 + total_future_tokens // self.spatial_tokens
        total_tokens = self.spatial_tokens + total_future_tokens

        full_tokens = torch.cat([cond_tokens, future_tokens], dim=1)
        embedded = self._embed_tokens(full_tokens)
        positions = self._full_positional_embeddings(total_frames=total_frames, dtype=embedded.dtype)
        encoder_source = embedded + positions.unsqueeze(0)

        visible_full_mask = torch.cat(
            [
                torch.ones(batch_size, self.spatial_tokens, dtype=torch.bool, device=full_tokens.device),
                visible_future_mask,
            ],
            dim=1,
        )

        visible_indices: List[torch.Tensor] = []
        visible_lengths = visible_full_mask.sum(dim=1)
        max_visible = int(visible_lengths.max().item())

        encoder_tokens = torch.zeros(
            batch_size,
            max_visible,
            self.cfg.model_dim,
            device=embedded.device,
            dtype=embedded.dtype,
        )
        key_padding_mask = torch.ones(
            batch_size,
            max_visible,
            device=embedded.device,
            dtype=torch.bool,
        )

        for batch_idx in range(batch_size):
            indices = visible_full_mask[batch_idx].nonzero(as_tuple=False).flatten()
            visible_indices.append(indices)
            length = indices.numel()
            encoder_tokens[batch_idx, :length] = encoder_source[batch_idx, indices]
            key_padding_mask[batch_idx, :length] = False

        encoded = self.backbone.encode(
            encoder_tokens,
            key_padding_mask=key_padding_mask,
        )

        decoder_tokens = self._mask_token_embeddings(
            batch_size=batch_size,
            total_tokens=total_tokens,
            dtype=embedded.dtype,
            device=embedded.device,
        )
        for batch_idx, indices in enumerate(visible_indices):
            length = indices.numel()
            decoder_tokens[batch_idx, indices] = encoded[batch_idx, :length]

        decoder_tokens = decoder_tokens + positions.unsqueeze(0)
        return self.backbone.decode(decoder_tokens)

    def _q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        abar = self.diff_abar.index_select(0, timesteps.reshape(-1)).view(*timesteps.shape, 1)
        return torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * noise

    def _diffusion_loss(self, x0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x0.numel() == 0:
            return x0.new_tensor(0.0)
        timesteps = torch.randint(
            0,
            self.cfg.diff_train_steps,
            (x0.shape[0],),
            device=x0.device,
        )
        noise = torch.randn_like(x0)
        x_s = self._q_sample(x0, timesteps, noise)
        eps_hat = self.diff_head(x_s, timesteps, z)
        return F.mse_loss(eps_hat, noise)

    def _deterministic_loss(
        self,
        *,
        future_tokens: torch.Tensor,
        future_reps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        frame_ids = torch.arange(
            future_tokens.shape[1],
            device=future_tokens.device,
            dtype=torch.long,
        ) // self.spatial_tokens
        weights = torch.exp(-frame_ids.float())
        weights = torch.where(
            frame_ids < self.cfg.aux_mse_frames,
            weights,
            torch.zeros_like(weights),
        )
        weights = weights.unsqueeze(0) * mask.float()
        normalizer = weights.sum(dim=1, keepdim=True)

        valid = normalizer.squeeze(1) > 0
        if not torch.any(valid):
            return future_tokens.new_tensor(0.0)

        weights = weights / normalizer.clamp_min(1e-8)
        x_hat = self.deter_head(future_reps)
        mse = (future_tokens - x_hat).pow(2).sum(dim=-1)
        per_example = (weights * mse).sum(dim=1)
        return per_example[valid].mean()

    def forward(self, x: torch.Tensor, *, return_latents: bool = False) -> Dict[str, torch.Tensor]:
        self._validate_inputs(x)
        batch_size, total_frames, _, _, _ = x.shape
        future_frames = total_frames - 1
        if future_frames < 1:
            raise ValueError("OmniCast requires at least one future frame")
        if future_frames > self.cfg.max_future_frames:
            raise ValueError(
                f"received {future_frames} future frames, but max_future_frames={self.cfg.max_future_frames}"
            )

        cond_tokens = self._encode_frame_to_tokens(x[:, 0])
        future_tokens = torch.stack(
            [self._encode_frame_to_tokens(x[:, 1 + frame_idx]) for frame_idx in range(future_frames)],
            dim=1,
        ).reshape(batch_size, future_frames * self.spatial_tokens, self.token_dim)

        mask = self._sample_mask(batch_size, future_frames, device=x.device)
        reps = self._mae_representations(
            cond_tokens=cond_tokens,
            future_tokens=future_tokens,
            visible_future_mask=~mask,
        )
        future_reps = reps[:, self.spatial_tokens :, :]

        diff_loss = self._diffusion_loss(future_tokens[mask], future_reps[mask])
        deter_loss = self._deterministic_loss(
            future_tokens=future_tokens,
            future_reps=future_reps,
            mask=mask,
        )
        loss = diff_loss + deter_loss

        out = {
            "loss": loss,
            "diff_loss": diff_loss,
            "deter_loss": deter_loss,
        }
        if return_latents:
            out.update(
                {
                    "cond_tokens": cond_tokens,
                    "future_tokens": future_tokens,
                    "mask": mask,
                    "reps": reps,
                }
            )
        return out

    @torch.no_grad()
    def _p_sample_loop(self, z_cond: torch.Tensor, *, steps: int, tau: float) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be at least 1")

        x = torch.randn(
            *z_cond.shape[:-1],
            self.token_dim,
            device=z_cond.device,
            dtype=z_cond.dtype,
        )
        timestep_sequence = torch.linspace(
            self.cfg.diff_train_steps - 1,
            0,
            steps,
            device=z_cond.device,
        ).round().long()

        for timestep in timestep_sequence.tolist():
            t = int(timestep)
            t_tensor = torch.full(
                z_cond.shape[:-1],
                t,
                device=z_cond.device,
                dtype=torch.long,
            )
            beta = self.diff_betas[t]
            alpha = self.diff_alphas[t]
            abar = self.diff_abar[t]

            eps_hat = self.diff_head(x, t_tensor, z_cond)
            mean = (x - (beta / torch.sqrt(1.0 - abar)) * eps_hat) / torch.sqrt(alpha)

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + tau * torch.sqrt(beta) * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        T: int,
        *,
        unmask_iters: Optional[int] = None,
        diff_steps: Optional[int] = None,
        tau: Optional[float] = None,
        random_order: bool = True,
    ) -> torch.Tensor:
        if x0.ndim != 4:
            raise ValueError("expected x0 to have shape (B, C, H, W)")
        if T < 1:
            raise ValueError("T must be at least 1")
        if T > self.cfg.max_future_frames:
            raise ValueError(
                f"requested {T} future frames, but max_future_frames={self.cfg.max_future_frames}"
            )
        if x0.shape[1:] != (self.cfg.in_ch, *self.cfg.image_hw):
            raise ValueError(
                f"expected x0 to have shape (B, {self.cfg.in_ch}, {self.cfg.image_hw[0]}, {self.cfg.image_hw[1]})"
            )

        batch_size = x0.shape[0]
        total_future_tokens = T * self.spatial_tokens
        unmask_iters = T if unmask_iters is None else unmask_iters
        diff_steps = self.cfg.diff_infer_steps if diff_steps is None else diff_steps
        tau = self.cfg.sample_temperature if tau is None else tau

        cond_tokens = self._encode_frame_to_tokens(x0)
        future_tokens = torch.zeros(
            batch_size,
            total_future_tokens,
            self.token_dim,
            device=x0.device,
            dtype=cond_tokens.dtype,
        )
        generated = torch.zeros(
            batch_size,
            total_future_tokens,
            device=x0.device,
            dtype=torch.bool,
        )

        schedule = _cosine_mask_schedule(unmask_iters)
        if random_order:
            base_order = torch.stack(
                [torch.randperm(total_future_tokens, device=x0.device) for _ in range(batch_size)],
                dim=0,
            )
        else:
            base_order = torch.arange(total_future_tokens, device=x0.device).repeat(batch_size, 1)

        for target_mask_ratio in schedule:
            reps = self._mae_representations(
                cond_tokens=cond_tokens,
                future_tokens=future_tokens,
                visible_future_mask=generated,
            )
            future_reps = reps[:, self.spatial_tokens :, :]
            target_remaining = int(round(target_mask_ratio * total_future_tokens))

            for batch_idx in range(batch_size):
                remaining = (~generated[batch_idx]).nonzero(as_tuple=False).flatten()
                remaining_count = remaining.numel()
                if remaining_count == 0:
                    continue

                num_to_generate = max(0, remaining_count - target_remaining)
                num_to_generate = min(num_to_generate, remaining_count)
                if num_to_generate == 0:
                    continue

                order = base_order[batch_idx, remaining]
                selected = remaining[torch.argsort(order)[:num_to_generate]]
                sampled_tokens = self._p_sample_loop(
                    future_reps[batch_idx : batch_idx + 1, selected, :],
                    steps=diff_steps,
                    tau=tau,
                )
                future_tokens[batch_idx, selected] = sampled_tokens.squeeze(0)
                generated[batch_idx, selected] = True

        frames = []
        for frame_idx in range(T):
            start = frame_idx * self.spatial_tokens
            end = start + self.spatial_tokens
            frames.append(self._decode_tokens_to_frame(future_tokens[:, start:end]))
        return torch.stack(frames, dim=1)

    @torch.no_grad()
    def sample_autoregressive(
        self,
        x0: torch.Tensor,
        total_steps: int,
        *,
        chunk_steps: int = 2,
        diff_steps: Optional[int] = None,
        tau: Optional[float] = None,
        random_order: bool = True,
    ) -> torch.Tensor:
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if chunk_steps < 1:
            raise ValueError("chunk_steps must be at least 1")

        chunks: List[torch.Tensor] = []
        context = x0
        steps_remaining = total_steps
        while steps_remaining > 0:
            current_chunk = min(chunk_steps, steps_remaining)
            sampled = self.sample(
                context,
                T=current_chunk,
                diff_steps=diff_steps,
                tau=tau,
                random_order=random_order,
            )
            chunks.append(sampled)
            context = sampled[:, -1]
            steps_remaining -= current_chunk
        return torch.cat(chunks, dim=1)


def omnicast_builder(
    task: str,
    in_ch: int,
    image_hw: Tuple[int, int],
    **kwargs: Any,
) -> OmniCast:
    if task not in {"regression", "forecasting", "generation"}:
        raise ValueError(
            "OmniCast is a forecasting/generative model; use task='regression', 'forecasting', or 'generation'"
        )

    cfg = OmniCastConfig(
        in_ch=in_ch,
        image_hw=image_hw,
        max_future_frames=kwargs.get("max_future_frames", 44),
        downsample=kwargs.get("downsample", 16),
        vae_base=kwargs.get("vae_base", 256),
        vae_ch_mult=tuple(kwargs.get("vae_ch_mult", (1, 2, 4, 4, 8))),
        vae_num_res_blocks=kwargs.get("vae_num_res_blocks", 2),
        vae_dropout=kwargs.get("vae_dropout", 0.0),
        z_dim=kwargs.get("z_dim"),
        model_dim=kwargs.get("model_dim", 1024),
        depth_enc=kwargs.get("depth_enc", 16),
        depth_dec=kwargs.get("depth_dec", 16),
        heads=kwargs.get("heads", 16),
        dropout=kwargs.get("dropout", 0.1),
        diff_train_steps=kwargs.get("diff_train_steps", 1000),
        diff_infer_steps=kwargs.get("diff_infer_steps", 100),
        diff_width=kwargs.get("diff_width", 2048),
        diff_blocks=kwargs.get("diff_blocks", 6),
        aux_mse_frames=kwargs.get("aux_mse_frames", 10),
        mask_ratio_min=kwargs.get("mask_ratio_min", 0.5),
        mask_ratio_max=kwargs.get("mask_ratio_max", 1.0),
        sample_temperature=kwargs.get("sample_temperature", 1.3),
        token_dim=kwargs.get("token_dim"),
    )
    return OmniCast(cfg)
