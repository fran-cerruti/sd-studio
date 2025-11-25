#!/usr/bin/env python3
"""
Configuración centralizada para SD-Studio
Optimizaciones específicas para GTX 1060 3GB
"""

from pathlib import Path
import torch

# Directorios
MODELS_DIR = Path("../models")
OUTPUTS_DIR = Path("../outputs")
HUGGINGFACE_CACHE = Path.home() / ".cache" / "huggingface"

def auto_detect_safetensors():
    """Auto-detecta archivos .safetensors en MODELS_DIR"""
    detected = []
    if MODELS_DIR.exists():
        for safetensor_file in MODELS_DIR.glob("*.safetensors"):
            model_id = safetensor_file.stem
            detected.append({
                "id": model_id,
                "file": safetensor_file.name,
                "name": model_id.replace("_", " ").title(),
                "type": "general",
                "description": "Auto-detected model",
                "size_gb": safetensor_file.stat().st_size / (1024**3),
                "emoji": "🎨"
            })
    return detected

# Configuración GTX 1060 3GB
GTX_1060_CONFIG = {
    "max_resolution": 768,
    "recommended_resolution": 512,
    "max_batch_size": 1,
    "torch_dtype": torch.float16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Optimizaciones que sabemos que funcionan
OPTIMIZATIONS = {
    "enable_attention_slicing": True,
    "attention_slice_size": 1,
    "enable_vae_slicing": True,
    "enable_cpu_offload": True,  # model_cpu_offload para HF, sequential para safetensors
    "safety_checker": None,
    "requires_safety_checker": False
}

# Catálogo unificado de modelos
MODELS_CATALOG = {
    "local": [
        {
            "id": "cyberrealistic_v90",
            "file": "cyberrealistic_v90.safetensors",
            "name": "CyberRealistic v9.0",
            "type": "photorealistic",
            "description": "Fotorealismo cyberpunk de alta calidad",
            "size_gb": 2.0,
            "emoji": "📸"
        },
        {
            "id": "dreamshaper_8",
            "file": "dreamshaper_8.safetensors",
            "name": "DreamShaper 8",
            "type": "photorealistic",
            "description": "Versátil fotorealista, muy popular",
            "size_gb": 2.0,
            "emoji": "📸"
        },
        {
            "id": "vision_v4",
            "file": "vision_v4.safetensors",
            "name": "Vision v4",
            "type": "photorealistic",
            "description": "Fotorealismo de alta calidad",
            "size_gb": 2.0,
            "emoji": "📸"
        },
        {
            "id": "toonyou_beta6",
            "file": "toonyou_beta6.safetensors",
            "name": "ToonYou Beta 6",
            "type": "animation",
            "description": "Estilo anime/toon de alta calidad",
            "size_gb": 2.2,
            "emoji": "🎨"
        },
        {
            "id": "dreamshaper_pixelart_v10",
            "file": "dreamshaperPixelart_v10.safetensors",
            "name": "DreamShaper Pixel Art v10",
            "type": "animation",
            "description": "Pixel art estilizado",
            "size_gb": 2.0,
            "emoji": "🎨"
        },
        {
            "id": "pixnite_purepixel_v10",
            "file": "pixnite15PurePixel_v10.safetensors",
            "name": "Pixnite Pure Pixel v10",
            "type": "animation",
            "description": "Pixel art puro de alta calidad",
            "size_gb": 2.0,
            "emoji": "🎨"
        }
    ],
    "huggingface": [
        {
            "id": "majicmix_v6",
            "repo": "digiplay/majicMIX_realistic_v6",
            "name": "majicMIX realistic v6",
            "type": "photorealistic",
            "description": "⭐ Fotorealismo de alta calidad (recomendado)",
            "variant": "fp16",
            "size_gb": 2.24,
            "emoji": "📸"
        },
        {
            "id": "urpm_v13",
            "repo": "stablediffusionapi/urpm-v13",
            "name": "URPM v1.3",
            "type": "photorealistic",
            "description": "Fotorealista merge de alta calidad",
            "variant": None,
            "size_gb": 4.5,
            "emoji": "📸"
        },
        {
            "id": "epicrealism",
            "repo": "emilianJR/epiCRealism",
            "name": "epiCRealism",
            "type": "photorealistic",
            "description": "Fotorealismo extremo de alta calidad",
            "variant": None,
            "size_gb": 4.0,
            "emoji": "📸"
        },
        {
            "id": "dreamshaper_hf",
            "repo": "Lykon/DreamShaper-8",
            "name": "DreamShaper 8 (HuggingFace)",
            "type": "photorealistic",
            "description": "Versátil fotorealista (versión HF)",
            "variant": None,
            "size_gb": 4.0,
            "emoji": "📸"
        }
    ]
}

# Negative prompts optimizados por tipo
NEGATIVE_PROMPTS = {
    "photorealistic": (
        "cartoon, anime, 3d render, cgi, painting, drawing, illustration, sketch, "
        "digital art, artificial, synthetic, fake, unrealistic, low quality, blurry, "
        "out of focus, distorted, deformed, disfigured, bad anatomy, bad proportions, "
        "extra limbs, cloned face, mutated, gross proportions, malformed limbs, "
        "missing arms, missing legs, extra arms, extra legs, fused fingers, "
        "too many fingers, long neck, duplicate, morbid, mutilated, poorly drawn hands, "
        "poorly drawn face, mutation, ugly, bad hands, text, watermark, signature, "
        "username, logo, oversaturated, overexposed, underexposed, noise, grain, "
        "jpeg artifacts, compression artifacts, low resolution, pixelated"
    ),
    "animation": (
        "photorealistic, realistic, photo, photograph, 3d render, "
        "low quality, worst quality, blurry, out of focus, "
        "bad anatomy, bad proportions, extra limbs, deformed, disfigured, "
        "ugly, poorly drawn, mutation, mutated, "
        "text, watermark, signature, username, logo, "
        "jpeg artifacts, compression artifacts, noise, grain"
    )
}

# Configuración de ControlNet
CONTROLNET_MODELS = {
    "canny": {
        "repo": "lllyasviel/sd-controlnet-canny",
        "name": "Canny Edge Detection",
        "description": "Detección de bordes para mantener estructura",
        "preprocessor": "canny"
    },
    "openpose": {
        "repo": "lllyasviel/sd-controlnet-openpose",
        "name": "OpenPose",
        "description": "Detección de pose humana",
        "preprocessor": "openpose"
    },
    "depth": {
        "repo": "lllyasviel/sd-controlnet-depth",
        "name": "Depth Map",
        "description": "Mapa de profundidad",
        "preprocessor": "depth"
    }
}

# Parámetros por defecto
DEFAULT_PARAMS = {
    "width": 512,
    "height": 512,
    "steps": 30,
    "cfg_scale": 7.0,
    "controlnet_scale": 1.0
}

# Resoluciones disponibles
RESOLUTION_PRESETS = {
    "256": (256, 256),
    "384": (384, 384),
    "512": (512, 512),
    "576": (576, 576),
    "640": (640, 640),
    "768": (768, 768)
}
