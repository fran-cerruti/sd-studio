#!/usr/bin/env python3
"""
Templates de configuración para diferentes perfiles de GPU
Usado por setup.py para generar config.py optimizado
"""

def generate_config(hw_info, profile_config):
    """
    Genera config.py completo según perfil de hardware
    
    Args:
        hw_info: Dict con información de hardware detectado
        profile_config: Dict con configuración del perfil
    
    Returns:
        String con contenido completo de config.py
    """
    
    opts = profile_config['optimizations']
    
    template = f'''#!/usr/bin/env python3
"""
Configuración centralizada para SD-Studio
Auto-generado por setup.py

Hardware detectado:
- GPU: {hw_info['name']}
- VRAM: {hw_info['vram_gb']:.1f} GB
- Arquitectura: {hw_info['arch']} (SM {hw_info['compute']})
- Perfil: {hw_info['profile']}
- Fecha: {hw_info.get('date', 'N/A')}
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
            detected.append({{
                "id": model_id,
                "file": safetensor_file.name,
                "name": model_id.replace("_", " ").title(),
                "type": "general",
                "description": "Auto-detected model",
                "size_gb": safetensor_file.stat().st_size / (1024**3),
                "emoji": "🎨"
            }})
    return detected

# Configuración de hardware (auto-detectada por setup.py)
HARDWARE_CONFIG = {{
    "gpu_name": "{hw_info['name']}",
    "vram_gb": {hw_info['vram_gb']:.1f},
    "compute_capability": "{hw_info['compute']}",
    "architecture": "{hw_info['arch']}",
    "profile": "{hw_info['profile']}",
    "max_resolution": {profile_config['max_resolution']},
    "recommended_resolution": {profile_config['recommended_resolution']},
    "torch_dtype": torch.float16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}}

# Alias para compatibilidad con código existente
GTX_1060_CONFIG = HARDWARE_CONFIG

# Optimizaciones (auto-configuradas por setup.py)
OPTIMIZATIONS = {{
    "enable_attention_slicing": {str(opts['enable_attention_slicing'])},
    "attention_slice_size": 1,
    "enable_vae_slicing": {str(opts['enable_vae_slicing'])},
    "enable_cpu_offload": {str(opts['enable_cpu_offload'])},
    "safety_checker": None,
    "requires_safety_checker": False
}}

# Catálogo unificado de modelos
MODELS_CATALOG = {{
    "local": [
        {{
            "id": "cyberrealistic_v90",
            "file": "cyberrealistic_v90.safetensors",
            "name": "CyberRealistic v9.0",
            "type": "photorealistic",
            "description": "Fotorealismo cyberpunk de alta calidad",
            "size_gb": 2.0,
            "emoji": "📸"
        }},
        {{
            "id": "dreamshaper_8",
            "file": "dreamshaper_8.safetensors",
            "name": "DreamShaper 8",
            "type": "photorealistic",
            "description": "Versátil fotorealista, muy popular",
            "size_gb": 2.0,
            "emoji": "📸"
        }},
        {{
            "id": "vision_v4",
            "file": "vision_v4.safetensors",
            "name": "Vision v4",
            "type": "photorealistic",
            "description": "Fotorealismo de alta calidad",
            "size_gb": 2.0,
            "emoji": "📸"
        }},
        {{
            "id": "toonyou_beta6",
            "file": "toonyou_beta6.safetensors",
            "name": "ToonYou Beta 6",
            "type": "animation",
            "description": "Estilo anime/toon de alta calidad",
            "size_gb": 2.2,
            "emoji": "🎨"
        }},
        {{
            "id": "dreamshaper_pixelart_v10",
            "file": "dreamshaperPixelart_v10.safetensors",
            "name": "DreamShaper Pixel Art v10",
            "type": "animation",
            "description": "Pixel art estilizado",
            "size_gb": 2.0,
            "emoji": "🎨"
        }},
        {{
            "id": "pixnite_purepixel_v10",
            "file": "pixnite15PurePixel_v10.safetensors",
            "name": "Pixnite Pure Pixel v10",
            "type": "animation",
            "description": "Pixel art puro de alta calidad",
            "size_gb": 2.0,
            "emoji": "🎨"
        }}
    ],
    "huggingface": [
        {{
            "id": "majicmix_v6",
            "repo": "digiplay/majicMIX_realistic_v6",
            "name": "majicMIX realistic v6",
            "type": "photorealistic",
            "description": "⭐ Fotorealismo de alta calidad (recomendado)",
            "variant": "fp16",
            "size_gb": 2.24,
            "emoji": "📸"
        }},
        {{
            "id": "urpm_v13",
            "repo": "stablediffusionapi/urpm-v13",
            "name": "URPM v1.3",
            "type": "photorealistic",
            "description": "Fotorealista merge de alta calidad",
            "variant": None,
            "size_gb": 4.5,
            "emoji": "📸"
        }},
        {{
            "id": "epicrealism",
            "repo": "emilianJR/epiCRealism",
            "name": "epiCRealism",
            "type": "photorealistic",
            "description": "Fotorealismo extremo de alta calidad",
            "variant": None,
            "size_gb": 4.0,
            "emoji": "📸"
        }},
        {{
            "id": "dreamshaper_hf",
            "repo": "Lykon/DreamShaper-8",
            "name": "DreamShaper 8 (HuggingFace)",
            "type": "photorealistic",
            "description": "Versátil fotorealista (versión HF)",
            "variant": None,
            "size_gb": 4.0,
            "emoji": "📸"
        }}
    ]
}}

# Negative prompts optimizados por tipo
NEGATIVE_PROMPTS = {{
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
}}

# Configuración de ControlNet
CONTROLNET_MODELS = {{
    "canny": {{
        "repo": "lllyasviel/sd-controlnet-canny",
        "name": "Canny Edge Detection",
        "description": "Detección de bordes para mantener estructura",
        "preprocessor": "canny"
    }},
    "openpose": {{
        "repo": "lllyasviel/sd-controlnet-openpose",
        "name": "OpenPose",
        "description": "Detección de pose humana",
        "preprocessor": "openpose"
    }},
    "depth": {{
        "repo": "lllyasviel/sd-controlnet-depth",
        "name": "Depth Map",
        "description": "Mapa de profundidad",
        "preprocessor": "depth"
    }}
}}

# Parámetros por defecto
DEFAULT_PARAMS = {{
    "width": {profile_config['recommended_resolution']},
    "height": {profile_config['recommended_resolution']},
    "steps": 30,
    "cfg_scale": 7.0,
    "controlnet_scale": 1.0
}}

# Resoluciones disponibles (limitadas según perfil)
RESOLUTION_PRESETS = {{
    "256": (256, 256),
    "384": (384, 384),
    "512": (512, 512),
    "576": (576, 576),
    "640": (640, 640),
    "768": (768, 768)'''
    
    # Agregar resoluciones adicionales para perfiles con más VRAM
    if profile_config['max_resolution'] >= 1024:
        template += ''',
    "1024": (1024, 1024)'''
    
    if profile_config['max_resolution'] >= 1536:
        template += ''',
    "1536": (1536, 1536)'''
    
    template += '''
}
'''
    
    return template
