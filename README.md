# OmniCast Summary

This repository contains a compact PyTorch implementation of **OmniCast**, based on:

- Tung Nguyen, Tuan Pham, Troy Arcomano, Veerabhadra Kotamarthi, Ian Foster, Sandeep Madireddy, and Aditya Grover, **"OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales"**, NeurIPS 2025
- Paper: https://arxiv.org/abs/2510.18707
- Local implementation: `omnicast/omnicast.py`

This README summarizes:

1. The evaluation setup and headline results reported in the paper
2. The model pipeline as implemented in this repository
3. Where this code matches the paper and where it is intentionally simplified

## What OmniCast Tries to Do

OmniCast is a **probabilistic weather forecasting model** designed to work across both:

- **medium-range forecasting**
- **subseasonal-to-seasonal (S2S) forecasting**

The core idea is to avoid the error accumulation of standard autoregressive rollouts by:

- compressing weather states into a continuous latent space with a VAE
- predicting many future latent tokens jointly with a masked transformer
- using a per-token diffusion model to sample the masked latent tokens

## Datasets and Benchmarks Used in the Paper

The paper evaluates OmniCast on **69 ERA5 variables** and uses two benchmark settings.

### 1. WeatherBench2 for medium-range forecasting

- Dataset source: **ERA5 reanalysis**
- Resolution: the paper says the medium-range setting uses the **native WeatherBench2 resolution**
- Split: **train 1979-2018**, **val 2019**, **test 2020**
- Initializations: **00 UTC and 12 UTC**
- Main metrics: **ensemble RMSE**, **CRPS**, **spread-skill ratio (SSR)**

### 2. ChaosBench for S2S forecasting

- Dataset source: **ERA5 reanalysis**
- Resolution: downsampled global grids for the S2S setup
- Split: **train 1979-2020**, **val 2021**, **test 2022**
- Initializations: **00 UTC**
- Lead times: roughly **2 to 6 weeks**; the implementation comments point to **44 future frames**
- Main target variables highlighted in the paper: **T850**, **Z500**, **Q700**
- Main metrics:
- Deterministic: **RMSE**, **absolute bias**, **multi-scale SSIM**
- Physics-based: **spectral divergence (SDIV)** and **spectral residual (SRES)**
- Probabilistic: **CRPS** and **SSR**

### ERA5 variables used

The paper states that training/evaluation use **69 variables** from ERA5:

- 4 surface variables: **2m temperature (T2m)**, **10m U wind (U10)**, **10m V wind (V10)**, and **mean sea-level pressure (MSLP)**
- 5 atmospheric variables across 13 pressure levels: **geopotential (Z)**, **temperature (T)**, **U wind**, **V wind**, and **specific humidity (Q)**

## Paper-Reported Evaluation Results

### S2S results on ChaosBench

The paper's main claim is that OmniCast is strongest at the **subseasonal-to-seasonal timescale**.

Reported takeaways:

- OmniCast achieves **state-of-the-art S2S performance** across deterministic, physics-based, and probabilistic metrics
- On deterministic metrics, it is a little weaker at short lead times, but its relative performance improves as lead time grows
- Beyond the later lead-time regime, the paper says OmniCast becomes one of the **top two** methods and **matches ECMWF-ENS** on deterministic performance
- OmniCast shows the **lowest bias** among compared baselines, staying near zero across the target variables
- On physics-based metrics, the paper says OmniCast is **substantially better than other deep learning methods** and often better than all baselines
- On probabilistic metrics, the paper says **OmniCast and ECMWF-ENS are the two leading methods** across variables and lead times
- The paper also states that OmniCast **outperforms ECMWF-ENS beyond longer lead times** on the probabilistic side after trailing it earlier in the forecast

In short, the paper positions OmniCast as especially strong when the forecasting horizon becomes long enough that autoregressive error accumulation starts to dominate competing deep learning systems.

### Medium-range results on WeatherBench2

For medium-range forecasting, the paper compares OmniCast to:

- **GenCast**
- **IFS-ENS**

Reported takeaways:

- OmniCast is **competitive at medium range**
- It performs **comparably to IFS-ENS**
- It is **slightly behind GenCast**
- The abstract states that OmniCast is roughly **10x to 20x faster** than leading probabilistic methods at the medium-range timescale

So the paper's overall message is:

- OmniCast is not mainly optimized to be the absolute best short-horizon model
- but it remains strong at medium range
- and its design becomes especially valuable at S2S horizons

### Efficiency findings

The paper emphasizes efficiency as a major advantage:

- training reported as **4 days on 32 NVIDIA A100 GPUs**
- compared against heavier baselines such as GenCast and NeuralGCM
- inference described as **orders of magnitude faster** than GenCast, NeuralGCM, and IFS-ENS

The paper attributes that speedup to two choices:

- forecasting in a **compressed latent space**
- doing expensive iterative sampling only in the **small diffusion head**, not through the full transformer each diffusion step

### Ablation findings reported in the paper

The paper highlights four main ablations:

#### 1. Auxiliary deterministic loss

- Removing the MSE objective hurts both RMSE and CRPS
- Applying MSE to **all** future frames also hurts
- The best setting is to apply deterministic supervision only to the **first 10 future frames**

#### 2. Training sequence length

- Shorter sequences or smaller step sizes help short-range behavior
- Full-sequence training performs better at S2S horizons because it reduces long-horizon error accumulation

#### 3. Unmasking order

- A fully randomized unmasking strategy works better than framewise alternatives
- The paper says this improves ensemble diversity and SSR

#### 4. Diffusion temperature

- Low temperature gives under-dispersive ensembles
- Very high temperature hurts RMSE and CRPS
- The paper reports **tau = 1.3** as the best balance

## Model Pipeline

The implementation in `omnicast/omnicast.py` follows the same high-level two-stage pipeline described in the paper.

### Stage 1: Compress each weather frame with a VAE

`WeatherVAE` is a UNet-style continuous VAE.

- Input: one weather frame with many channels and variables
- Encoder: convolution plus residual downsampling blocks
- Latent: a spatial grid of continuous latent vectors
- Decoder: upsamples latent grids back into weather fields
- Loss: reconstruction MSE plus KL penalty

Why this matters:

- forecasting directly in raw weather space is expensive
- forecasting in a compressed latent space makes long sequences feasible

### Stage 2: Predict future latent tokens with masked generative modeling

After the VAE is trained, the model freezes it and works in latent-token space.

- The initial condition frame is encoded into conditioning tokens
- Future frames are encoded into target latent tokens during training
- A random subset of future tokens is masked
- The transformer sees conditioning tokens, visible future tokens, and `[MASK]` tokens in masked future positions

The backbone is `MAETransformer`, which acts like an encoder-decoder masked transformer:

- the encoder processes only visible tokens
- the decoder reconstructs a full token sequence including masked locations
- the outputs at masked positions become conditioning vectors for token generation

### Per-token diffusion head

Each masked token is modeled as a continuous random variable.

Instead of predicting a discrete codebook ID, OmniCast uses `DiffusionHead`:

- input: noisy token plus diffusion timestep plus transformer conditioning vector
- network: small MLP with AdaLN-style conditioning
- target: predict the diffusion noise for that token

This lets the model generate continuous latent tokens without vector quantization.

### Auxiliary deterministic head

The implementation includes `DeterministicHead`, an MLP that directly predicts masked latent tokens.

Its role is to stabilize short-range forecasts:

- only early future frames receive this deterministic supervision
- the weights decay exponentially with lead time

This directly reflects one of the paper's most important ablation findings.

### Inference by iterative unmasking

Generation starts with **all future tokens masked**.

Then the model repeats:

1. Run the transformer on conditioning tokens plus the current partially filled future sequence
2. Choose some masked positions to reveal
3. Sample those positions with the diffusion head
4. Write them back into the sequence

The implementation uses:

- a **cosine mask schedule**
- **random token order**
- configurable diffusion steps
- configurable diffusion temperature

This is the key mechanism that replaces standard autoregressive frame-by-frame rollout.

## How This Repository Maps to the Paper

This repo is best read as a **clean educational implementation**, not a full reproduction package.

What it captures well:

- continuous VAE latent compression
- masked transformer over conditioning plus future tokens
- per-token diffusion sampling
- auxiliary deterministic head for early frames
- iterative unmasking inference

What is simplified in this repo:

- there are **no ERA5, WeatherBench2, or ChaosBench data loaders**
- there are **no official training scripts or checkpoints**
- `example_training()` uses **synthetic random tensors**
- the example uses much smaller dimensions than the paper
- several comments explicitly note paper settings versus demo settings

Examples of paper-vs-demo differences already called out in the code:

- paper: **69 variables**; demo: **8**
- paper: latent dim **1024**; demo: **64**
- paper: **44** future frames; demo: **6**
- paper transformer depth: **16/16**; demo: **2/2**
- paper diffusion steps: **1000 train / 100 infer**; demo: **100 / 20**
