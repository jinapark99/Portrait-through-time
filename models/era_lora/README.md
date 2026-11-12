# Era-specific style LoRAs

Per-era PEFT UNet adapters for SD1.5 (`r=4`), trained from
`data/labels/portraits_104_era_emotion.csv` splits.

| Era | Slug | Images | Status |
|-----|------|-------:|--------|
| Modern/Impressionist | `modern_impressionist` | 40 | Trained (MPS short-run) |
| Baroque | `baroque` | 20 | Pending Colab — see `docs/era-lora-colab.md` |
| Romantic | `romantic` | 20 | Pending Colab |
| Renaissance | `renaissance` | 15 | Optional |
| Rococo | `rococo` | 8 | Skipped (too few) |
| Medieval/Gothic | `medieval_gothic` | 1 | Skipped |

## Layout

```
models/era_lora/<slug>/final/{adapter_config.json,adapter_model.safetensors}
models/era_lora/<slug>/train_result.json
```

## Train

```bash
python scripts/split_era_lora_datasets.py
python scripts/train_lora_era.py --era modern_impressionist --max-steps 80 --epochs 8
```

## Honesty note

The older `lora_trained_model/` weights are a **mixed** 104-image style LoRA.
They are **not** era-specific. Claim “era LoRAs” only for adapters under this folder
with a real `train_result.json` (steps, device, n_images).
