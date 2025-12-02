#!/usr/bin/env python3
"""
Model Manager - Gestión unificada de modelos SD
Maneja carga, descarga y cache de modelos .safetensors y HuggingFace
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import gc
from diffusers import (
    StableDiffusionPipeline, 
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
from config import (
    MODELS_DIR, get_models_catalog, GTX_1060_CONFIG, OPTIMIZATIONS, 
    NEGATIVE_PROMPTS
)


class ModelManager:
    """Gestor centralizado de modelos"""
    
    def __init__(self):
        self.current_model = {
            "loaded": False,
            "model_id": None,
            "model_name": None,
            "model_type": None,
            "source": None,
            "pipeline": None
        }
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un modelo por su ID (detección dinámica)"""
        catalog = get_models_catalog()
        
        # Buscar en catálogo local
        for model in catalog["local"]:
            if model["id"] == model_id:
                return {"source": "local", **model}
        
        # Buscar en HuggingFace
        for model in catalog["huggingface"]:
            if model["id"] == model_id:
                return {"source": "huggingface", **model}
        
        return None
    
    def list_available_models(self) -> Dict[str, list]:
        """Lista modelos disponibles (detección automática dinámica)"""
        catalog = get_models_catalog()
        
        # Todos los modelos detectados ya están disponibles
        available_local = [
            {**model, "available": True, "source": "local"}
            for model in catalog["local"]
        ]
        
        available_hf = [
            {**model, "available": True, "source": "huggingface"}
            for model in catalog["huggingface"]
        ]
        
        return {
            "local": available_local,
            "huggingface": available_hf
        }
    
    def load_model(self, model_id: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Carga un modelo específico
        
        Args:
            model_id: ID del modelo a cargar
            force_reload: Forzar recarga incluso si ya está cargado
        
        Returns:
            Dict con información del modelo cargado
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        # Si ya está cargado y no se fuerza reload
        if not force_reload and self.current_model["loaded"] and self.current_model["model_id"] == model_id:
            return {
                "success": True,
                "message": "Modelo ya está cargado",
                "model": {
                    "id": self.current_model["model_id"],
                    "name": self.current_model["model_name"],
                    "type": self.current_model["model_type"]
                }
            }
        
        # Descargar modelo anterior si existe
        if self.current_model["loaded"]:
            self.unload_model()
        
        # Limpieza agresiva antes de cargar
        self._aggressive_cleanup()
        
        # Cargar según fuente
        if model_info["source"] == "local":
            pipe = self._load_local_model(model_info)
        else:
            pipe = self._load_huggingface_model(model_info)
        
        # Actualizar estado
        self.current_model.update({
            "loaded": True,
            "model_id": model_id,
            "model_name": model_info["name"],
            "model_type": model_info["type"],
            "source": model_info["source"],
            "pipeline": pipe
        })
        
        return {
            "success": True,
            "message": f"Modelo {model_info['name']} cargado exitosamente",
            "model": {
                "id": model_id,
                "name": model_info["name"],
                "type": model_info["type"],
                "description": model_info["description"]
            }
        }
    
    def _load_local_model(self, model_info: Dict[str, Any]):
        """Carga un modelo local .safetensors"""
        model_path = MODELS_DIR / model_info["file"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        pipe = StableDiffusionPipeline.from_single_file(
            str(model_path),
            torch_dtype=GTX_1060_CONFIG["torch_dtype"],
            safety_checker=OPTIMIZATIONS["safety_checker"],
            load_safety_checker=False,
            use_safetensors=False
        )
        
        # Desactivar safety checker embebido
        if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
            pipe.safety_checker = None
        
        # Optimizaciones para 3GB VRAM (sequential para safetensors)
        pipe.enable_sequential_cpu_offload()
        
        if OPTIMIZATIONS["enable_attention_slicing"]:
            pipe.enable_attention_slicing(OPTIMIZATIONS["attention_slice_size"])
        
        if OPTIMIZATIONS["enable_vae_slicing"]:
            pipe.enable_vae_slicing()
        
        # Scheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        
        return pipe
    
    def _load_huggingface_model(self, model_info: Dict[str, Any]):
        """Carga un modelo de HuggingFace"""
        kwargs = {
            "torch_dtype": GTX_1060_CONFIG["torch_dtype"],
            "safety_checker": OPTIMIZATIONS["safety_checker"],
            "requires_safety_checker": OPTIMIZATIONS["requires_safety_checker"],
            "use_safetensors": True
        }
        
        if model_info.get("variant"):
            kwargs["variant"] = model_info["variant"]
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_info["repo"],
            **kwargs
        )
        
        # Optimizaciones (model_cpu_offload para HuggingFace)
        pipe.enable_model_cpu_offload()
        
        if OPTIMIZATIONS["enable_attention_slicing"]:
            pipe.enable_attention_slicing(OPTIMIZATIONS["attention_slice_size"])
        
        if OPTIMIZATIONS["enable_vae_slicing"]:
            pipe.enable_vae_slicing()
        
        # Scheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        
        return pipe
    
    def unload_model(self):
        """Descarga el modelo actual y libera memoria"""
        if not self.current_model["loaded"]:
            return {"success": True, "message": "No hay modelo cargado"}
        
        del self.current_model["pipeline"]
        self._aggressive_cleanup()
        
        self.current_model.update({
            "loaded": False,
            "model_id": None,
            "model_name": None,
            "model_type": None,
            "source": None,
            "pipeline": None
        })
        
        return {"success": True, "message": "Modelo descargado y memoria liberada"}
    
    def get_current_model(self) -> Dict[str, Any]:
        """Retorna información del modelo actual"""
        return {
            "loaded": self.current_model["loaded"],
            "model_id": self.current_model["model_id"],
            "model_name": self.current_model["model_name"],
            "model_type": self.current_model["model_type"],
            "source": self.current_model["source"]
        }
    
    def get_pipeline(self):
        """Retorna el pipeline actual (para generación)"""
        if not self.current_model["loaded"]:
            raise RuntimeError("No hay modelo cargado")
        return self.current_model["pipeline"]
    
    def get_negative_prompt(self, custom: Optional[str] = None) -> str:
        """
        Retorna negative prompt apropiado
        
        Args:
            custom: Negative prompt personalizado (opcional)
        
        Returns:
            Negative prompt a usar
        """
        if custom:
            return custom
        
        if not self.current_model["loaded"]:
            return NEGATIVE_PROMPTS["photorealistic"]
        
        return NEGATIVE_PROMPTS[self.current_model["model_type"]]
    
    @staticmethod
    def cleanup():
        """Limpieza básica de memoria"""
        torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def _aggressive_cleanup():
        """Limpieza super agresiva"""
        import time
        for i in range(5):
            ModelManager.cleanup()
            torch.cuda.empty_cache()
            if i < 4:
                time.sleep(0.5)
        torch.cuda.reset_peak_memory_stats()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
