"""D1-style portrait generation: img2img + era prompt + optional style LoRA.

IP-Adapter FaceID is attempted when deps are present; otherwise clearly stubbed.
Prefer a real image out on MPS / CUDA / CPU over a perfect FaceID path.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = os.environ.get(
    "PORTRAIT_BASE_MODEL", "runwayml/stable-diffusion-v1-5"
)
DEFAULT_LORA_PATHS = (
    REPO_ROOT / "lora_trained_model" / "final",
    REPO_ROOT / "lora_trained_model 2" / "final",
)
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted face, deformed, ugly, watermark, text, "
    "photo, selfie, modern clothing, oversaturated"
)


@dataclass
class GenerationMeta:
    mode: str  # img2img | faceid | placeholder
    device: str
    faceid_available: bool
    faceid_used: bool
    faceid_reason: str
    lora_loaded: bool
    lora_path: str | None
    lora_scale: float
    strength: float
    steps: int
    guidance_scale: float
    base_model: str
    elapsed_sec: float
    prompt: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    image: Image.Image
    meta: GenerationMeta

    def to_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return buf.getvalue()


def pick_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _dtype_for(device: str):
    import torch

    # MPS float16 is usually fine for SD1.5; fall back to float32 on CPU.
    if device in {"mps", "cuda"}:
        return torch.float16
    return torch.float32


def resolve_lora_path(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    env = os.environ.get("PORTRAIT_LORA_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p
    for candidate in DEFAULT_LORA_PATHS:
        if candidate.exists() and (
            (candidate / "adapter_model.safetensors").exists()
            or (candidate / "pytorch_lora_weights.safetensors").exists()
            or list(candidate.glob("*.safetensors"))
        ):
            return candidate
    return None


def faceid_status() -> tuple[bool, str]:
    """Return (available, reason). FaceID needs insightface + IP-Adapter extras."""
    if os.environ.get("PORTRAIT_DISABLE_FACEID", "").strip() in {"1", "true", "yes"}:
        return False, "disabled by PORTRAIT_DISABLE_FACEID"
    try:
        import insightface  # noqa: F401
    except Exception as exc:
        return False, f"insightface not available: {exc}"
    try:
        # Optional IP-Adapter FaceID helper; often missing on Mac MVP installs.
        from diffusers import IPAdapterFaceIDPlusPipeline  # type: ignore  # noqa: F401
    except Exception:
        try:
            import ip_adapter  # noqa: F401
        except Exception as exc:
            return (
                False,
                "IP-Adapter FaceID extras missing "
                f"(install insightface + ip-adapter / FaceID weights): {exc}",
            )
    return True, "insightface + FaceID pipeline importable"


class PortraitGenerator:
    """Lazy-loaded img2img generator with optional LoRA and FaceID stub."""

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        lora_path: str | Path | None = None,
        lora_scale: float = 0.55,
        device: str | None = None,
        allow_placeholder: bool = True,
    ) -> None:
        self.base_model = base_model
        self.lora_path = resolve_lora_path(lora_path)
        self.lora_scale = lora_scale
        self.device = device or pick_device()
        self.allow_placeholder = allow_placeholder
        self._pipe = None
        self._load_error: str | None = None
        self._lora_loaded = False

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            self._load_pipeline()
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Failed to load generation pipeline")
            raise

    @staticmethod
    def _attach_lora(pipe: Any, lora_path: Path, lora_scale: float) -> None:
        """Load PEFT-style UNet LoRA adapters used by this repo."""
        # Prefer PEFT when adapter_config.json is present (our trained format).
        if (Path(lora_path) / "adapter_config.json").is_file():
            from peft import PeftModel

            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
            try:
                pipe.unet.set_adapter(pipe.unet.active_adapter)
            except Exception:
                pass
            # Approximate strength via scaling if supported.
            try:
                for module in pipe.unet.modules():
                    if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                        for k in list(module.scaling.keys()):
                            module.scaling[k] = lora_scale
                    elif hasattr(module, "scaling") and isinstance(module.scaling, (int, float)):
                        module.scaling = lora_scale
            except Exception:
                pass
            return

        try:
            pipe.load_lora_weights(str(lora_path), prefix=None)
        except TypeError:
            pipe.load_lora_weights(str(lora_path))
        if hasattr(pipe, "set_adapters"):
            try:
                adapters = list(pipe.get_active_adapters() or [])
                if adapters:
                    pipe.set_adapters(adapters, adapter_weights=[lora_scale] * len(adapters))
            except Exception:
                pass

    def _load_pipeline(self) -> None:
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

        dtype = _dtype_for(self.device)
        logger.info(
            "Loading SD img2img base=%s device=%s dtype=%s",
            self.base_model,
            self.device,
            dtype,
        )
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.base_model,
                dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except TypeError:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.base_model,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        if self.lora_path is not None:
            try:
                self._attach_lora(pipe, self.lora_path, self.lora_scale)
                self._lora_loaded = True
                logger.info("LoRA loaded from %s (scale≈%s)", self.lora_path, self.lora_scale)
            except Exception as exc:
                logger.warning("LoRA load failed (%s): %s", self.lora_path, exc)
                self._lora_loaded = False

        if self.device == "mps":
            pipe = pipe.to("mps")
            # Reduce memory spikes on Apple Silicon.
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        elif self.device == "cuda":
            pipe = pipe.to("cuda")
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        else:
            pipe = pipe.to("cpu")

        self._pipe = pipe

    def generate(
        self,
        selfie: Image.Image | Path | str,
        prompt: str,
        *,
        strength: float = 0.72,
        steps: int = 24,
        guidance_scale: float = 7.0,
        seed: int | None = 42,
        use_faceid: bool = True,
    ) -> GenerationResult:
        """Generate a styled portrait. Falls back to placeholder if models fail."""
        t0 = time.perf_counter()
        init = _as_rgb_512(selfie)
        faceid_ok, faceid_reason = faceid_status()
        notes: list[str] = []
        faceid_used = False
        mode = "img2img"

        if use_faceid and faceid_ok:
            try:
                image = self._generate_faceid(
                    init, prompt, strength=strength, steps=steps, guidance_scale=guidance_scale, seed=seed
                )
                faceid_used = True
                mode = "faceid"
            except Exception as exc:
                notes.append(f"FaceID attempted but failed; falling back to img2img: {exc}")
                faceid_reason = f"{faceid_reason}; runtime error: {exc}"
                image = None
        else:
            image = None
            if use_faceid and not faceid_ok:
                notes.append(f"FaceID stubbed: {faceid_reason}")

        if image is None:
            try:
                self.ensure_loaded()
                image = self._generate_img2img(
                    init,
                    prompt,
                    strength=strength,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
                mode = "img2img"
            except Exception as exc:
                if not self.allow_placeholder:
                    raise
                notes.append(f"img2img unavailable ({exc}); returning stylized placeholder")
                image = _placeholder_portrait(init, prompt)
                mode = "placeholder"
                self._load_error = str(exc)

        elapsed = time.perf_counter() - t0
        meta = GenerationMeta(
            mode=mode,
            device=self.device if mode != "placeholder" else pick_device(),
            faceid_available=faceid_ok,
            faceid_used=faceid_used,
            faceid_reason=faceid_reason,
            lora_loaded=self._lora_loaded,
            lora_path=str(self.lora_path) if self.lora_path else None,
            lora_scale=self.lora_scale,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            base_model=self.base_model,
            elapsed_sec=round(elapsed, 3),
            prompt=prompt,
            notes=notes,
        )
        return GenerationResult(image=image, meta=meta)

    def _generate_img2img(
        self,
        init: Image.Image,
        prompt: str,
        *,
        strength: float,
        steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> Image.Image:
        import torch

        assert self._pipe is not None
        generator = None
        if seed is not None:
            # MPS generators are device-specific in recent torch.
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except Exception:
                generator = torch.Generator().manual_seed(seed)

        # Cross-attention / LoRA scale for older diffusers APIs.
        cross_attention_kwargs = None
        if self._lora_loaded:
            cross_attention_kwargs = {"scale": self.lora_scale}

        result = self._pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=init,
            strength=max(0.15, min(0.95, strength)),
            guidance_scale=guidance_scale,
            num_inference_steps=max(4, steps),
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return result.images[0]

    def _generate_faceid(self, *args: Any, **kwargs: Any) -> Image.Image:
        """Reserved for IP-Adapter FaceID. Raises until extras are installed."""
        raise RuntimeError(
            "FaceID path not wired for this install — use img2img. "
            "Install insightface + IP-Adapter FaceID weights to enable."
        )


def _as_rgb_512(image: Image.Image | Path | str) -> Image.Image:
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image
    img = img.convert("RGB")
    # Center-crop to square then resize — keeps faces better than stretch.
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((512, 512), Image.Resampling.LANCZOS)


def _placeholder_portrait(init: Image.Image, prompt: str) -> Image.Image:
    """Lightweight stylized fallback when SD weights cannot run."""
    from PIL import ImageEnhance, ImageFilter, ImageDraw, ImageFont

    img = init.copy().resize((512, 512))
    img = ImageEnhance.Color(img).enhance(1.35)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    # Warm painterly wash.
    wash = Image.new("RGB", img.size, (42, 28, 18))
    img = Image.blend(img, wash, 0.22)
    draw = ImageDraw.Draw(img)
    label = "placeholder · SD unavailable"
    era_hint = (prompt[:48] + "…") if len(prompt) > 48 else prompt
    draw.rectangle((0, 460, 512, 512), fill=(20, 14, 10))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((12, 468), label, fill=(230, 210, 180), font=font)
    draw.text((12, 488), era_hint, fill=(180, 160, 130), font=font)
    return img


# Process-wide singleton for FastAPI warm reuse.
_GENERATOR: PortraitGenerator | None = None


def get_generator(**kwargs: Any) -> PortraitGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = PortraitGenerator(**kwargs)
    return _GENERATOR
