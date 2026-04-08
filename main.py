import argparse
from typing import Tuple

import torch

from omnicast import omnicast_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test runner for the OmniCast model."
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "sample", "both"),
        default="both",
        help="Which OmniCast path to exercise.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the synthetic input.",
    )
    parser.add_argument(
        "--future-frames",
        type=int,
        default=2,
        help="Number of future frames to forecast.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of channels per frame.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
        help="Input height. Must be divisible by --downsample.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="Input width. Must be divisible by --downsample.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=16,
        help="Latent downsampling factor used by OmniCast.",
    )
    parser.add_argument(
        "--vae-base",
        type=int,
        default=16,
        help="Base channel width for the VAE backbone.",
    )
    parser.add_argument(
        "--z-dim",
        type=int,
        default=32,
        help="VAE latent channel count for the smoke test model.",
    )
    parser.add_argument(
        "--token-dim",
        type=int,
        default=8,
        help="Per-token latent dimension.",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=32,
        help="Transformer hidden width.",
    )
    parser.add_argument(
        "--depth-enc",
        type=int,
        default=1,
        help="Number of encoder transformer blocks.",
    )
    parser.add_argument(
        "--depth-dec",
        type=int,
        default=1,
        help="Number of decoder transformer blocks.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Attention heads for the tiny test model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout used in the transformer backbone.",
    )
    parser.add_argument(
        "--diff-train-steps",
        type=int,
        default=20,
        help="Training diffusion steps for the toy run.",
    )
    parser.add_argument(
        "--diff-infer-steps",
        type=int,
        default=5,
        help="Sampling diffusion steps for the toy run.",
    )
    parser.add_argument(
        "--aux-mse-frames",
        type=int,
        default=2,
        help="Number of early frames included in the deterministic auxiliary loss.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.75,
        help="Fixed mask ratio used during the forward smoke test.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.3,
        help="Sampling temperature for diffusion decoding.",
    )
    parser.add_argument(
        "--unmask-iters",
        type=int,
        default=None,
        help="Iterative unmasking steps used by sample(). Defaults to T.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_image_size(image_hw: Tuple[int, int], downsample: int) -> None:
    height, width = image_hw
    if height % downsample != 0 or width % downsample != 0:
        raise ValueError(
            f"Image size {image_hw} must be divisible by downsample={downsample}."
        )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    image_hw = (args.height, args.width)
    validate_image_size(image_hw, args.downsample)

    model = omnicast_builder(
        task="forecasting",
        in_ch=args.channels,
        image_hw=image_hw,
        downsample=args.downsample,
        vae_base=args.vae_base,
        z_dim=args.z_dim,
        token_dim=args.token_dim,
        model_dim=args.model_dim,
        depth_enc=args.depth_enc,
        depth_dec=args.depth_dec,
        heads=args.heads,
        dropout=args.dropout,
        diff_train_steps=args.diff_train_steps,
        diff_infer_steps=args.diff_infer_steps,
        aux_mse_frames=args.aux_mse_frames,
        mask_ratio_min=args.mask_ratio,
        mask_ratio_max=args.mask_ratio,
    )
    return model.to(device)


def make_synthetic_batch(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.randn(
        args.batch_size,
        1 + args.future_frames,
        args.channels,
        args.height,
        args.width,
        device=device,
    )


def run_forward_test(model: torch.nn.Module, batch: torch.Tensor) -> None:
    model.train()
    outputs = model(batch)
    print("Forward test")
    print(f"  loss:       {outputs['loss'].item():.6f}")
    print(f"  diff_loss:  {outputs['diff_loss'].item():.6f}")
    print(f"  deter_loss: {outputs['deter_loss'].item():.6f}")


@torch.no_grad()
def run_sample_test(
    model: torch.nn.Module,
    batch: torch.Tensor,
    future_frames: int,
    diff_steps: int,
    tau: float,
    unmask_iters: int | None,
) -> None:
    model.eval()
    context = batch[:, 0]
    samples = model.sample(
        context,
        T=future_frames,
        unmask_iters=unmask_iters,
        diff_steps=diff_steps,
        tau=tau,
    )
    print("Sample test")
    print(f"  output shape: {tuple(samples.shape)}")
    print(f"  mean:         {samples.mean().item():.6f}")
    print(f"  std:          {samples.std().item():.6f}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)

    model = build_model(args, device)
    batch = make_synthetic_batch(args, device)

    print("OmniCast smoke test")
    print(f"  device:      {device}")
    print(f"  input shape: {tuple(batch.shape)}")
    print(f"  parameters:  {count_parameters(model):,}")

    if args.mode in {"forward", "both"}:
        run_forward_test(model, batch)

    if args.mode in {"sample", "both"}:
        run_sample_test(
            model=model,
            batch=batch,
            future_frames=args.future_frames,
            diff_steps=args.diff_infer_steps,
            tau=args.tau,
            unmask_iters=args.unmask_iters,
        )


if __name__ == "__main__":
    main()
