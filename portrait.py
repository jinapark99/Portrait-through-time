"""
LoRA Fine-tuning Script for Portrait Generation
Complete training script based on Hugging Face official methods
"""

import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F
from tqdm import tqdm

print("🚀 LoRA Fine-tuning for Portrait Generation")
print("="*80)

# Configuration
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DATASET_DIR = Path("lora_training_data/images")
OUTPUT_DIR = Path("lora_output")
OUTPUT_DIR.mkdir(exist_ok=True)

LEARNING_RATE = 1e-4
NUM_EPOCHS = 20  # Start with fewer epochs for CPU training
BATCH_SIZE = 1
LORA_RANK = 4
IMAGE_SIZE = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n💻 Device: {device}")
if device == "cpu":
    print("⚠️  Warning: Training on CPU will be slow. Consider using GPU or Colab.")
    print("   Reducing epochs to 20 for CPU training.")

# Dataset
class PortraitDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        self.caption_files = sorted(list(self.image_dir.glob("*.txt")))
        
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print(f"📊 Found {len(self.image_files)} training images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_files[idx]).convert("RGB")
        image = self.transform(image)
        
        # Load caption
        with open(self.caption_files[idx], 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        return {"image": image, "caption": caption}

# Load models
print("\n📦 Loading Stable Diffusion models...")
print("   (This will download ~5GB on first run)")

try:
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = StableDiffusionPipeline.from_pretrained(MODEL_NAME, subfolder="vae").vae
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("\n💡 Make sure you have stable internet and enough disk space (~5GB)")
    exit(1)

# Move to device
vae = vae.to(device)
text_encoder = text_encoder.to(device)
unet = unet.to(device)

# Freeze VAE and text encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Apply LoRA to UNet
print("\n🔧 Applying LoRA to UNet...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none"
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# Dataset and DataLoader
dataset = PortraitDataset(DATASET_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

# Training loop
print("\n🎓 Starting training...")
print("="*80)

global_step = 0
for epoch in range(NUM_EPOCHS):
    unet.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch in progress_bar:
        # Get data
        images = batch["image"].to(device)
        captions = batch["caption"]
        
        # Encode text
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        with torch.no_grad():
            encoder_hidden_states = text_encoder(text_input_ids)[0]
        
        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample random timestep
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Calculate loss
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_dir = OUTPUT_DIR / f"checkpoint-{epoch+1}"
        checkpoint_dir.mkdir(exist_ok=True)
        unet.save_pretrained(checkpoint_dir)
        print(f"💾 Checkpoint saved: {checkpoint_dir}")

# Save final model
final_dir = OUTPUT_DIR / "final"
final_dir.mkdir(exist_ok=True)
unet.save_pretrained(final_dir)
print(f"\n✅ Training complete! Final model saved to: {final_dir}")

print("\n" + "="*80)
print("📝 Next steps:")
print("   1. Test the model with your generation script")
print("   2. Compare results before/after LoRA")
print(f"\n💾 Model saved to: {final_dir}")

