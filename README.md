# OmniCast

This directory contains a compact PyTorch implementation of:

> Tung Nguyen, Tuan Pham, Troy Arcomano, Veerabhadra Kotamarthi, Ian Foster,
> Sandeep Madireddy, and Aditya Grover, “OmniCast: A Masked Latent Diffusion
> Model for Weather Forecasting Across Time Scales,” NeurIPS 2025.

- Paper: <https://arxiv.org/abs/2510.18707>
- Implementation: [`omnicast.py`](omnicast.py)

The code follows the paper's two-stage design: a continuous VAE compresses each
weather state independently, then an MAE-style Transformer and per-token
diffusion head jointly model future latent states across space and time.

This is a paper-aligned, self-contained implementation, not a complete
reproduction package. It does not include ERA5 data preparation, benchmark
evaluation code, pretrained weights, or the distributed training infrastructure
used for the reported results.

## Requirements

- Python 3.10+
- PyTorch
- einops

Install the two runtime dependencies with:

```bash
pip install torch einops
```

From the project root, run the reduced synthetic example with:

```powershell
.\env\Scripts\python.exe .\omnicast\omnicast.py
```

The example trains a small VAE, performs several stage-two optimization steps,
generates two independent ensemble members, and decodes them into weather
fields. It is a behavioral demonstration rather than a meaningful forecast.

## Architecture

### 1. Continuous weather VAE

`WeatherVAE` embeds each frame independently:

```text
(B, V, H, W) -> (B, D, h, w) -> (B, V, H, W)
```

The default S2S configuration follows Appendix A.1 of the paper:

| Setting | Default |
| --- | ---: |
| Input/output variables | 69 |
| Base channels | 256 |
| Channel multipliers | `(1, 2, 4, 4, 8)` |
| Residual blocks per encoder level | 2 |
| Latent channels | 1024 |
| Spatial downsampling | 16x |
| Dropout | 0.0 |
| KL weight | `5e-5` |

The encoder and decoder use PDEArena/LDM-style pre-normalized residual blocks.
The decoder reconstructs weather fields from the latent tensor alone; it does
not receive encoder skip connections. This is necessary because generated
latents have no corresponding encoder features at inference time.

For the paper's S2S data, a `69 x 128 x 256` frame becomes a
`1024 x 8 x 16` latent map, or 128 continuous tokens.

### 2. MAE encoder-decoder Transformer

`MAETransformer` processes initial-condition tokens and partially visible
future tokens.

- A mask ratio `gamma ~ U[0.5, 1.0]` is sampled during training.
- Random masks span both spatial and temporal positions.
- Each batch item receives an independent random mask with the same packed
  token count.
- The encoder sees the initial-condition tokens and visible future tokens.
- The decoder receives encoded visible tokens plus learnable `[MASK]` tokens.
- Separate spatial-plus-temporal positional embeddings are added before the
  encoder and decoder.
- Both stages use bidirectional full self-attention.

Paper-scale Transformer defaults are:

| Setting | Default |
| --- | ---: |
| Hidden dimension | 1024 |
| Encoder depth | 16 |
| Decoder depth | 16 |
| Attention heads | 16 |
| MLP expansion | 4x |
| Dropout | 0.1 |

### 3. Per-token diffusion head

`DiffusionHead` models each masked continuous token conditioned on its
Transformer representation.

- Input: noisy latent token, diffusion timestep, and Transformer output
  `z_i`.
- Network: six residual MLP blocks of width 2048.
- Each block uses adaptive LayerNorm conditioning from `z_i` and the timestep
  embedding.
- Objective: MSE between predicted and sampled Gaussian noise.

`DiffusionSchedule` uses 1000 linear training noise levels by default. Inference
uses 100 levels by default and mathematically respaces the DDPM transitions.
It does not simply skip one-step transitions from the original schedule.
Temperature `tau` scales the reverse-process noise.

The paper specifies a linear schedule but not its endpoints; this implementation
uses the conventional `beta_start=1e-4` and `beta_end=0.02`.

### 4. Auxiliary deterministic head

`DeterministicHead` directly predicts masked latent tokens from `z_i`.

Following Appendix A.2:

- only the first 10 future frames receive deterministic supervision;
- frame `k` receives weight `exp(-k)`;
- all applicable token weights are normalized once to sum to one;
- masked weights are not renormalized after mask sampling.

The complete stage-two loss is:

```text
loss = diffusion_loss + deterministic_loss
```

### 5. Iterative generation

`OmniCast.generate()` starts with every future position masked. At each
iteration it:

1. runs the MAE Transformer on the current partially generated sequence;
2. follows a cosine schedule to determine how many positions to reveal;
3. independently chooses random positions for every batch member;
4. samples those tokens with the per-token diffusion head;
5. inserts the samples into the future sequence.

The default S2S configuration generates 44 daily future frames in 44 unmasking
iterations with `tau=1.3`.

## Tensor Shapes

The public methods use the following layouts:

| Method | Input | Output |
| --- | --- | --- |
| `WeatherVAE.forward` | `(B, V, H, W)` | reconstruction, sample, mean, log-variance |
| `OmniCast.encode_frames` | `(B, T, V, H, W)` | `(B, T, h*w, D)` |
| `OmniCast.decode_tokens` | `(B, T, h*w, D)` | `(B, T, V, H, W)` |
| `OmniCast.training_step` | condition `(B, h*w, D)`, future `(B, T*h*w, D)` | loss dictionary |
| `OmniCast.generate` | `(B, h*w, D)` | `(B, T*h*w, D)` |
| `OmniCast.generate_ensemble` | `(B, h*w, D)` | `(B, E, T*h*w, D)` |
| `OmniCast.generate_autoregressive` | `(B, h*w, D)` | `(B, requested_frames, h*w, D)` |

Latent tokens are flattened frame-major: all spatial tokens for frame 0,
followed by all spatial tokens for frame 1, and so on.

## Two-Stage Training

### Stage 1: train the VAE

```python
import torch
from omnicast.omnicast import WeatherVAE

vae = WeatherVAE()
optimizer = torch.optim.Adam(
    vae.parameters(),
    lr=2e-4,
    betas=(0.9, 0.95),
    weight_decay=1e-5,
)

weather = torch.randn(2, 69, 128, 256)
reconstruction, latent, mean, logvar = vae(weather)
loss = vae.vae_loss(
    weather,
    reconstruction,
    mean,
    logvar,
    kl_weight=5e-5,
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The paper trains this stage for 200 epochs with batch size 32, 20 epochs of
linear warmup, and cosine decay over the remaining 180 epochs. The example in
`omnicast.py` intentionally uses a much smaller model and synthetic tensors.

### Stage 2: train OmniCast in latent space

```python
from einops import rearrange
from omnicast.omnicast import OmniCast

model = OmniCast(vae=vae)
optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=2e-4,
    betas=(0.9, 0.95),
    weight_decay=1e-5,
)

# frames: (B, 45, 69, 128, 256), consisting of one IC and 44 targets
all_tokens = model.encode_frames(frames)
condition = all_tokens[:, 0]
future = rearrange(all_tokens[:, 1:], "b t n d -> b (t n) d")

losses = model.training_step(condition, future)
optimizer.zero_grad()
losses["loss"].backward()
optimizer.step()
```

The VAE is frozen by `OmniCast` and remains in evaluation mode even when
`model.train()` is called. The paper trains stage two for 100 epochs with batch
size 32, 10 warmup epochs, and cosine decay over the remaining 90 epochs.

Paper-scale tensors and models require substantial accelerator memory; the code
above documents the interface and is not expected to fit on a typical CPU or
consumer GPU.

## S2S Ensemble Inference

The paper creates an ensemble by replicating each initial condition and
independently sampling every copy. `generate_ensemble()` implements that
behavior directly:

```python
# initial_frame: (B, 1, 69, 128, 256)
condition = model.encode_frames(initial_frame)[:, 0]

ensemble_tokens = model.generate_ensemble(
    condition,
    ensemble_size=50,
    n_iterations=44,
    diffusion_steps=100,
    tau=1.3,
)
# (B, 50, 44*128, 1024)
```

Each member receives independent diffusion noise and an independent random
unmasking order. To decode one batch item:

```python
members = ensemble_tokens[0].reshape(50, 44, 128, 1024)
forecasts = model.decode_tokens(members, h=8, w=16)
# (50, 44, 69, 128, 256)
```

## Medium-Range Rollout

Section 5.2 uses a separate medium-range configuration:

- native `69 x 721 x 1440` weather fields;
- `256 x 45 x 90` latent maps;
- two predicted frames per call at 12-hour intervals;
- autoregressive rollout using the most recent predicted frame;
- one unmasking iteration per prediction window;
- diffusion temperature `tau=1.0`.

Configure a separate model with `latent_dim=256`, `n_spatial=45*90`, and
`n_future_frames=2`, then call:

```python
rollout = medium_range_model.generate_autoregressive(
    condition,
    n_frames=30,       # 15 days at 12-hour intervals
    n_iterations=1,
    diffusion_steps=100,
    tau=1.0,
)
# (B, 30, 45*90, 256)
```

The latent rollout method is implemented, but this compact VAE does not include
the official medium-range model's special `721 -> 720 -> 721` row adapter.
Four downsampling stages map a 721-row input to 45 latent rows, while the
decoder naturally reconstructs 720 rows. Native WeatherBench2 decoding
therefore requires an equivalent input/output row adapter or preprocessing to
an even 720-row grid.

## Paper Evaluation Setup

The paper trains and evaluates on 69 ERA5 variables:

- four surface variables: T2m, U10, V10, and MSLP;
- geopotential, temperature, U wind, V wind, and specific humidity at 13
  pressure levels.

The S2S experiment uses ChaosBench at `128 x 256` resolution, trains on
1979-2020, validates on 2021, and tests on 2022. It forecasts days 1-44 from
00 UTC initializations.

The medium-range experiment uses WeatherBench2 at `721 x 1440`, trains on
1979-2018, validates on 2019, and tests on 2020 using 00 UTC and 12 UTC
initializations.

These datasets, splits, normalization statistics, metrics, and evaluation
pipelines are described here for context but are not included in this
repository.

## Paper Defaults vs. Included Demo

| Setting | Paper S2S model | Built-in demo |
| --- | ---: | ---: |
| Weather variables | 69 | 8 |
| Input resolution | `128 x 256` | `32 x 64` |
| Latent channels | 1024 | 64 |
| Future frames | 44 | 6 |
| Transformer depth | 16 encoder / 16 decoder | 2 / 2 |
| Attention heads | 16 | 4 |
| Diffusion levels | 1000 train / 100 infer | 100 / 20 |
| VAE/channel training data | ERA5 | random tensors |

## Reproduction Limitations

The implementation captures the model mechanics described in the paper,
including latent-only VAE decoding, independent random masks, MAE
encoder-decoder processing, weighted near-term supervision, respaced diffusion,
cosine iterative unmasking, ensemble replication, and medium-range
autoregression.

Exact paper results additionally require:

- ERA5 acquisition, variable ordering, normalization, and temporal sampling;
- WeatherBench2 and ChaosBench evaluation code;
- the full learning-rate schedules and distributed mixed-precision trainer;
- trained VAE and OmniCast checkpoints;
- the native WeatherBench2 721-row VAE input/output adapter;
- the paper's accelerator scale and ensemble evaluation setup.

Because those components are absent, successful execution of this script
validates architecture and tensor flow, not the forecast scores reported in the
paper.
