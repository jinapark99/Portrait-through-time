#!/usr/bin/env python3
"""Colab / cloud GPU runner for remaining era LoRAs.

Paste into a Colab cell after uploading an era zip, or run:
  python scripts/colab_train_era_lora.py --era baroque --max-steps 400 --epochs 20

Expects dataset at ./by_era/<slug>/ with jpg+txt pairs
(or pass --dataset). Saves to ./era_lora/<slug>/final/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


COLAB_SETUP = r'''
# --- Colab cell 1: deps ---
!pip install -q diffusers==0.32.2 transformers accelerate peft pillow torchvision tqdm

# --- Colab cell 2: upload / unzip era pack ---
from google.colab import files
import zipfile, os
uploaded = files.upload()  # upload baroque.zip etc. OR by_era.zip
for name in uploaded:
    with zipfile.ZipFile(name) as z:
        z.extractall(".")
    print("extracted", name)

# --- Colab cell 3: train one era ---
!python scripts/colab_train_era_lora.py --era baroque --max-steps 400 --epochs 20 --device cuda
# then:
!python scripts/colab_train_era_lora.py --era romantic --max-steps 400 --epochs 20 --device cuda

# Download adapters:
from google.colab import files
files.download("era_lora/baroque/final/adapter_model.safetensors")
files.download("era_lora/baroque/final/adapter_config.json")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", required=True)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (auto if omitted)")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-colab-help", action="store_true")
    args = parser.parse_args()

    if args.print_colab_help:
        print(COLAB_SETUP)
        return 0

    # Prefer the fuller local trainer when present (repo checkout).
    repo_trainer = Path(__file__).resolve().parent / "train_lora_era.py"
    if repo_trainer.is_file():
        slug = args.era
        cmd = [
            sys.executable,
            str(repo_trainer),
            "--era",
            slug,
            "--max-steps",
            str(args.max_steps),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--lora-rank",
            str(args.lora_rank),
        ]
        if args.dataset:
            cmd += ["--dataset", str(args.dataset)]
        if args.output:
            cmd += ["--output", str(args.output)]
        else:
            # Colab-friendly default relative path
            out = Path("era_lora") / slug.replace("/", "_").replace(" ", "_").lower()
            # train_lora_era maps aliases; pass through
            cmd += ["--output", str(out)]
        print("Delegating to", repo_trainer)
        return subprocess.call(cmd)

    print(
        "train_lora_era.py not found next to this script. "
        "Clone the Portrait repo or copy scripts/train_lora_era.py into the runtime.",
        file=sys.stderr,
    )
    print(json.dumps({"colab_help": True}))
    print(COLAB_SETUP)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
