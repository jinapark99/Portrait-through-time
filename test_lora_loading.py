#!/usr/bin/env python3
"""
🔧 LoRA 로딩 테스트 스크립트
다양한 방법으로 LoRA를 로딩해서 어떤 방식이 작동하는지 확인합니다.
"""

import torch
import os
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

def test_lora_loading_methods():
    """다양한 LoRA 로딩 방법을 테스트"""
    print("🔧 LoRA 로딩 방법 테스트 시작...")
    
    # M1 GPU 설정
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"📱 사용 디바이스: {device}")
    
    # 기본 파이프라인 로드
    print("🎨 기본 파이프라인 로드 중...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device in ["mps", "cuda"] else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # 테스트할 LoRA 경로들
    lora_paths = [
        "lora_trained_model/final",
        "lora_trained_model/checkpoint-20",
        "lora_trained_model/checkpoint-15"
    ]
    
    for lora_path in lora_paths:
        print(f"\n🔍 테스트 중: {lora_path}")
        
        if not os.path.exists(lora_path):
            print(f"❌ 경로가 존재하지 않습니다: {lora_path}")
            continue
        
        # 방법 1: 기본 load_lora_weights
        try:
            print("  방법 1: load_lora_weights() 시도...")
            pipe.load_lora_weights(lora_path)
            print("  ✅ 방법 1 성공!")
        except Exception as e:
            print(f"  ❌ 방법 1 실패: {e}")
        
        # 방법 2: weight_name 명시
        try:
            print("  방법 2: weight_name 명시 시도...")
            pipe.load_lora_weights(lora_path, weight_name="adapter_model.safetensors")
            print("  ✅ 방법 2 성공!")
        except Exception as e:
            print(f"  ❌ 방법 2 실패: {e}")
        
        # 방법 3: adapter_name 명시
        try:
            print("  방법 3: adapter_name 명시 시도...")
            pipe.load_lora_weights(lora_path, adapter_name="portrait_style")
            print("  ✅ 방법 3 성공!")
        except Exception as e:
            print(f"  ❌ 방법 3 실패: {e}")
        
        # 방법 4: PEFT 직접 사용
        try:
            print("  방법 4: PEFT 직접 사용 시도...")
            # UNet에 LoRA 적용
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            print("  ✅ 방법 4 성공!")
        except Exception as e:
            print(f"  ❌ 방법 4 실패: {e}")
        
        # 방법 5: 파일 직접 로드
        try:
            print("  방법 5: 파일 직접 로드 시도...")
            adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                pipe.load_lora_weights(adapter_path)
                print("  ✅ 방법 5 성공!")
            else:
                print("  ❌ adapter_model.safetensors 파일이 없습니다.")
        except Exception as e:
            print(f"  ❌ 방법 5 실패: {e}")
    
    print("\n🎯 테스트 완료!")
    print("성공한 방법을 사용해서 실제 이미지 생성을 테스트해보겠습니다...")

def test_actual_generation():
    """실제 이미지 생성으로 LoRA 적용 확인"""
    print("\n🎨 실제 이미지 생성 테스트...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 파이프라인 로드
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device in ["mps", "cuda"] else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # LoRA 로드 (가장 성공한 방법 사용)
    lora_path = "lora_trained_model/final"
    try:
        pipe.load_lora_weights(lora_path)
        print(f"✅ LoRA 로드 성공: {lora_path}")
    except Exception as e:
        print(f"❌ LoRA 로드 실패: {e}")
        return
    
    # 테스트 이미지 로드
    from PIL import Image
    test_image = Image.open("IMG_5241.JPG").convert("RGB").resize((512, 512))
    
    # 다양한 프롬프트로 테스트
    test_prompts = [
        "portrait painting, joyful expression",
        "medieval portrait, renaissance style, happy face",
        "classical portrait painting, cheerful expression",
        "portrait, contemplative expression, melancholic expression"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 테스트 프롬프트 {i+1}: {prompt}")
        
        try:
            result = pipe(
                prompt=prompt,
                image=test_image,
                strength=0.7,
                guidance_scale=7.5,
                num_inference_steps=10,  # 빠른 테스트
                generator=torch.Generator(device=device)
            )
            
            output_path = f"lora_test_{i+1}.png"
            result.images[0].save(output_path)
            print(f"✅ 이미지 생성 완료: {output_path}")
            
        except Exception as e:
            print(f"❌ 이미지 생성 실패: {e}")

if __name__ == "__main__":
    test_lora_loading_methods()
    test_actual_generation()





