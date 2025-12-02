#!/usr/bin/env python3
"""
Configuración centralizada para SD-Studio
Auto-generado por setup.py

Hardware detectado:
- GPU: NVIDIA GeForce GTX 1060 3GB
- VRAM: 3.0 GB
- Arquitectura: Pascal (SM 6.1)
- Perfil: LOW_VRAM
- Fecha: 2025-11-27 10:43:41
"""

from pathlib import Path
import torch

# Directorios con búsqueda inteligente
def _find_or_create_dir(dir_name: str) -> Path:
    """
    Busca directorio en orden:
    1. Dentro del proyecto (./dir_name)
    2. Un nivel arriba (../dir_name)
    3. Si no existe, lo crea dentro del proyecto
    """
    # Opción 1: Dentro del proyecto
    local_dir = Path(__file__).parent / dir_name
    if local_dir.exists():
        return local_dir
    
    # Opción 2: Un nivel arriba (compartido)
    parent_dir = Path(__file__).parent.parent / dir_name
    if parent_dir.exists():
        return parent_dir
    
    # Opción 3: Crear dentro del proyecto
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir

MODELS_DIR = _find_or_create_dir("models")
OUTPUTS_DIR = _find_or_create_dir("outputs")
HUGGINGFACE_CACHE = Path.home() / ".cache" / "huggingface"

def auto_detect_safetensors():
    """Auto-detecta archivos .safetensors en MODELS_DIR"""
    detected = []
    if MODELS_DIR.exists():
        for safetensor_file in MODELS_DIR.glob("*.safetensors"):
            model_id = safetensor_file.stem
            # Inferir tipo basado en nombre
            name_lower = model_id.lower()
            if any(kw in name_lower for kw in ['toon', 'anime', 'pixel', 'cartoon', 'comic']):
                model_type = "animation"
                emoji = "🎨"
            else:
                model_type = "photorealistic"
                emoji = "📸"
            
            detected.append({
                "id": model_id,
                "file": safetensor_file.name,
                "name": model_id.replace("_", " ").title(),
                "type": model_type,
                "description": "Auto-detected local model",
                "size_gb": round(safetensor_file.stat().st_size / (1024**3), 2),
                "emoji": emoji
            })
    return detected

def auto_detect_huggingface():
    """
    Auto-detecta SOLO modelos de Stable Diffusion descargados y completos
    Detecta el variant correcto (fp16, bf16, etc.) leyendo los archivos
    Filtra LLMs, ControlNet y otros modelos no-SD
    """
    detected = []
    if not HUGGINGFACE_CACHE.exists():
        return detected
    
    hub_dir = HUGGINGFACE_CACHE / "hub"
    if not hub_dir.exists():
        return detected
    
    for model_dir in hub_dir.glob("models--*"):
        try:
            # Buscar snapshot más reciente
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                continue
            
            # Obtener último snapshot (más reciente)
            snapshots = sorted(snapshots_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if not snapshots:
                continue
            
            snapshot = snapshots[0]
            
            # VERIFICAR QUE SEA UN MODELO SD (tiene model_index.json)
            model_index = snapshot / "model_index.json"
            if not model_index.exists():
                continue  # No es un modelo SD, skip (puede ser LLM, ControlNet, etc.)
            
            # VERIFICAR QUE ESTÉ COMPLETO (tiene unet/)
            unet_dir = snapshot / "unet"
            if not unet_dir.exists():
                continue  # Modelo incompleto o descarga a medias, skip
            
            # DETECTAR VARIANT leyendo archivos del unet
            variant = None
            for file in unet_dir.iterdir():
                if file.is_symlink() or file.is_file():
                    filename = file.name
                    if filename.startswith("diffusion_pytorch_model"):
                        # Extraer variant del nombre: diffusion_pytorch_model.fp16.safetensors
                        if ".fp16." in filename:
                            variant = "fp16"
                        elif ".bf16." in filename:
                            variant = "bf16"
                        # Si no tiene variant, queda None (modelo full precision)
                        break
            
            # Extraer repo (formato: models--user--model)
            parts = model_dir.name.split("--")
            if len(parts) >= 3:
                user = parts[1]
                model = "--".join(parts[2:])  # Por si el nombre tiene --
                repo = f"{user}/{model}"
                model_id = f"{user}_{model}".replace("-", "_")
                
                # Inferir tipo basado en nombre
                name_lower = model.lower()
                if any(kw in name_lower for kw in ['toon', 'anime', 'pixel', 'cartoon', 'comic']):
                    model_type = "animation"
                    emoji = "🎨"
                else:
                    model_type = "photorealistic"
                    emoji = "📸"
                
                detected.append({
                    "id": model_id,
                    "repo": repo,
                    "name": model.replace("-", " ").replace("_", " ").title(),
                    "type": model_type,
                    "description": "Auto-detected HuggingFace model",
                    "variant": variant,  # Detectado correctamente
                    "size_gb": 0.0,
                    "emoji": emoji
                })
        except Exception as e:
            # Si hay error con un modelo, continuar con el siguiente
            continue
    
    return detected

# Configuración de hardware (auto-detectada por setup.py)
HARDWARE_CONFIG = {
    "gpu_name": "NVIDIA GeForce GTX 1060 3GB",
    "vram_gb": 3.0,
    "compute_capability": "6.1",
    "architecture": "Pascal",
    "profile": "LOW_VRAM",
    "max_resolution": 768,
    "recommended_resolution": 512,
    "torch_dtype": torch.float16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Alias para compatibilidad con código existente
GTX_1060_CONFIG = HARDWARE_CONFIG

# Optimizaciones (auto-configuradas por setup.py)
OPTIMIZATIONS = {
    "enable_attention_slicing": True,
    "attention_slice_size": 1,
    "enable_vae_slicing": True,
    "enable_cpu_offload": True,
    "safety_checker": None,
    "requires_safety_checker": False
}

# Catálogo de modelos (auto-generado dinámicamente)
def get_models_catalog():
    """Genera catálogo de modelos detectando automáticamente locales y HuggingFace"""
    return {
        "local": auto_detect_safetensors(),
        "huggingface": auto_detect_huggingface()
    }

# Para compatibilidad con código existente
MODELS_CATALOG = get_models_catalog()

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

# Resoluciones disponibles (limitadas según perfil)
RESOLUTION_PRESETS = {
    "256": (256, 256),
    "384": (384, 384),
    "512": (512, 512),
    "576": (576, 576),
    "640": (640, 640),
    "768": (768, 768)
}
