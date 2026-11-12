# Era LoRA — Colab / cloud GPU continuation

Mac MPS can finish a **short** real run (e.g. Modern/Impressionist, ~80 steps).  
Use this for the remaining primary eras (**Baroque**, **Romantic**) with fuller steps.

## Pack a dataset zip (local)

```bash
python scripts/split_era_lora_datasets.py
cd data/lora_training_data/by_era
zip -r ../../../baroque_era.zip baroque
zip -r ../../../romantic_era.zip romantic
```

## Colab

1. Runtime → GPU (T4+)
2. Upload `scripts/train_lora_era.py`, `scripts/colab_train_era_lora.py`, and an era zip
3. Install + train:

```python
!pip install -q diffusers transformers accelerate peft pillow torchvision tqdm
```

```bash
python scripts/colab_train_era_lora.py --era baroque --max-steps 400 --epochs 20
python scripts/colab_train_era_lora.py --era romantic --max-steps 400 --epochs 20
```

4. Download `era_lora/<slug>/final/{adapter_config.json,adapter_model.safetensors}`
5. Place under repo: `models/era_lora/<slug>/final/`

## Local wiring

`app/generation.py` resolves `models/era_lora/<slug>/final` by matched era, else falls back to the old mixed LoRA / prompt-only.
