#!/usr/bin/env python3
"""Real per-era SD1.5 UNet LoRA training (PEFT).

Adapted from data/train_lora_complete.py with:
  - MPS / CUDA / CPU device selection
  - CLI: --era, --max-steps, --epochs
  - Outputs under models/era_lora/<slug>/

Example (Mac MPS, short but real):
  python scripts/train_lora_era.py --era modern_impressionist --max-steps 80 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]

ERA_ALIASES = {
    "modern": "modern_impressionist",
    "modern_impressionist": "modern_impressionist",
    "modern/impressionist": "modern_impressionist",
    "baroque": "baroque",
    "romantic": "romantic",
    "renaissance": "renaissance",
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_slug(era: str) -> str:
    key = era.strip().lower().replace(" ", "_")
    if key in ERA_ALIASES:
        return ERA_ALIASES[key]
    return key.replace("/", "_")


class PortraitDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = 512):
        self.image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        if not self.image_files:
            raise FileNotFoundError(f"No images in {image_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_files[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        cap_path = path.with_suffix(".txt")
        if cap_path.is_file():
            caption = cap_path.read_text(encoding="utf-8").strip()
        else:
            caption = "portrait painting"
        return {"image": image, "caption": caption, "filename": path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", required=True, help="Era slug or name, e.g. baroque")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Folder of jpg+txt pairs (default: data/lora_training_data/by_era/<slug>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (default: models/era_lora/<slug>)",
    )
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="Stop after this many optimizer steps (short real runs)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=40,
        help="Checkpoint frequency in steps (0=only final)",
    )
    args = parser.parse_args()

    slug = resolve_slug(args.era)
    dataset_dir = args.dataset or (
        REPO / "data" / "lora_training_data" / "by_era" / slug
    )
    output_root = args.output or (REPO / "models" / "era_lora" / slug)
    output_root.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    # float32 is more stable for MPS training; fp16 for CUDA.
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(args.seed)

    print("=" * 72)
    print(f"Era LoRA train  slug={slug}")
    print(f"  dataset={dataset_dir}")
    print(f"  output={output_root}")
    print(f"  device={device} dtype={dtype}")
    print(
        f"  epochs={args.epochs} max_steps={args.max_steps} "
        f"lr={args.lr} rank={args.lora_rank}"
    )
    print("=" * 72)

    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(device, dtype=dtype)
    text_encoder.to(device)
    # UNet stays float32 on MPS for grad stability; CUDA can use fp16 weights + scaler optional.
    unet.to(device, dtype=torch.float32 if device == "mps" else dtype)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.train()

    dataset = PortraitDataset(dataset_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        (p for p in unet.parameters() if p.requires_grad), lr=args.lr
    )

    meta = {
        "slug": slug,
        "era_arg": args.era,
        "dataset_dir": str(dataset_dir),
        "n_images": len(dataset),
        "base_model": args.model,
        "device": device,
        "dtype": str(dtype),
        "epochs_planned": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "lora_rank": args.lora_rank,
        "image_size": args.image_size,
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_root / "train_config.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    global_step = 0
    history: list[dict] = []
    t0 = time.perf_counter()
    stop = False

    for epoch in range(args.epochs):
        if stop:
            break
        epoch_loss = 0.0
        n_batches = 0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in bar:
            images = batch["image"].to(device, dtype=dtype if device != "mps" else torch.float32)
            # VAE expects same dtype as its weights
            if device == "mps":
                images = images.to(dtype=torch.float32)
            captions = list(batch["caption"])

            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += float(loss.item())
            n_batches += 1
            bar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if args.save_every_steps and global_step % args.save_every_steps == 0:
                ckpt = output_root / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(ckpt)
                print(f"\n💾 checkpoint → {ckpt}")

            if global_step >= args.max_steps:
                stop = True
                break

        avg = epoch_loss / max(1, n_batches)
        history.append({"epoch": epoch + 1, "avg_loss": avg, "steps": global_step})
        print(f"Epoch {epoch + 1} avg_loss={avg:.4f} steps={global_step}")

    final_dir = output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(final_dir)

    elapsed = time.perf_counter() - t0
    result = {
        **meta,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(elapsed, 1),
        "steps_completed": global_step,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "history": history,
        "final_adapter": str(final_dir.relative_to(REPO)),
        "honest_note": (
            "This is a short but real PEFT UNet LoRA run (noise-prediction MSE), "
            "not a fake/stub checkpoint. Quality is limited by steps/images; "
            "Colab GPU recommended for fuller epochs."
        ),
    }
    (output_root / "train_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("\n✅ Done")
    print(f"  steps={global_step} elapsed={elapsed:.1f}s")
    print(f"  final={final_dir}")
    print(f"  meta={output_root / 'train_result.json'}")
    return 0


if __name__ == "__main__":
    # Avoid tokenizer fork warnings / MPS quirks with over-parallelism.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
