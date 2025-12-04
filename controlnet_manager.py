#!/usr/bin/env python3
"""
ControlNet Manager - Gestión de generación con ControlNet
Wrapper sobre la implementación probada de test_controlnet.py
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
import gc
import uuid

from config import CONTROLNET_MODELS, OPTIMIZATIONS, GTX_1060_CONFIG, OUTPUTS_DIR


class ControlNetManager:
    """Gestor de ControlNet para generación con estructura"""
    
    def __init__(self, model_manager):
        """
        Args:
            model_manager: Instancia de ModelManager
        """
        self.model_manager = model_manager
        self.controlnet_cache = {}
        self.current_pipeline = None
        self.current_controlnet_type = None
        self.current_model_id = None  # Track del modelo usado en el pipeline
    
    def load_controlnet_pipeline(
        self,
        controlnet_type: str = "canny",
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Carga pipeline con ControlNet
        
        Args:
            controlnet_type: Tipo de ControlNet (canny, openpose, depth)
            force_reload: Forzar recarga
        
        Returns:
            Dict con información de carga
        """
        if controlnet_type not in CONTROLNET_MODELS:
            raise ValueError(f"ControlNet tipo '{controlnet_type}' no soportado")
        
        # Verificar que hay modelo SD cargado
        if not self.model_manager.current_model["loaded"]:
            raise RuntimeError("Debes cargar un modelo SD primero")
        
        current_model_id = self.model_manager.current_model["model_id"]
        
        # Si ya está cargado el mismo tipo Y el mismo modelo base
        if (not force_reload and 
            self.current_controlnet_type == controlnet_type and 
            self.current_model_id == current_model_id):
            return {
                "success": True,
                "message": f"ControlNet {controlnet_type} ya está cargado"
            }
        
        # Limpiar pipeline anterior
        if self.current_pipeline:
            del self.current_pipeline
            self._cleanup()
        
        # Cargar ControlNet si no está en cache
        if controlnet_type not in self.controlnet_cache:
            controlnet_info = CONTROLNET_MODELS[controlnet_type]
            controlnet = ControlNetModel.from_pretrained(
                controlnet_info["repo"],
                torch_dtype=GTX_1060_CONFIG["torch_dtype"],
                use_safetensors=True
            )
            self.controlnet_cache[controlnet_type] = controlnet
        else:
            controlnet = self.controlnet_cache[controlnet_type]
        
        # Obtener modelo base del model_manager
        base_model_info = self.model_manager.get_model_info(
            self.model_manager.current_model["model_id"]
        )
        
        # Cargar pipeline con ControlNet
        if base_model_info["source"] == "local":
            pipe = self._load_controlnet_local(base_model_info, controlnet)
        else:
            pipe = self._load_controlnet_huggingface(base_model_info, controlnet)
        
        self.current_pipeline = pipe
        self.current_controlnet_type = controlnet_type
        self.current_model_id = current_model_id  # Guardar el model_id usado
        
        return {
            "success": True,
            "message": f"ControlNet {controlnet_type} cargado exitosamente",
            "controlnet_type": controlnet_type,
            "model": self.model_manager.current_model["model_name"]
        }
    
    def _load_controlnet_local(self, model_info: Dict[str, Any], controlnet):
        """Carga pipeline ControlNet desde modelo local"""
        from config import MODELS_DIR
        
        model_path = MODELS_DIR / model_info["file"]
        
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            str(model_path),
            controlnet=controlnet,
            torch_dtype=GTX_1060_CONFIG["torch_dtype"],
            safety_checker=None,
            use_safetensors=True
        )
        
        # Optimizaciones
        pipe.enable_sequential_cpu_offload()
        if OPTIMIZATIONS["enable_attention_slicing"]:
            pipe.enable_attention_slicing(OPTIMIZATIONS["attention_slice_size"])
        if OPTIMIZATIONS["enable_vae_slicing"]:
            pipe.enable_vae_slicing()
        
        return pipe
    
    def _load_controlnet_huggingface(self, model_info: Dict[str, Any], controlnet):
        """Carga pipeline ControlNet desde HuggingFace"""
        kwargs = {
            "controlnet": controlnet,
            "torch_dtype": GTX_1060_CONFIG["torch_dtype"],
            "safety_checker": None,
            "requires_safety_checker": False,
            "use_safetensors": True
        }
        
        if model_info.get("variant"):
            kwargs["variant"] = model_info["variant"]
        
        try:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_info["repo"],
                **kwargs
            )
        except (OSError, EnvironmentError) as e:
            if "safetensors" in str(e).lower():
                kwargs["use_safetensors"] = False
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_info["repo"],
                    **kwargs
                )
            else:
                raise
        
        # Optimizaciones
        pipe.enable_sequential_cpu_offload()
        if OPTIMIZATIONS["enable_attention_slicing"]:
            pipe.enable_attention_slicing(OPTIMIZATIONS["attention_slice_size"])
        if OPTIMIZATIONS["enable_vae_slicing"]:
            pipe.enable_vae_slicing()
        
        return pipe
    
    def preprocess_image(
        self,
        image: Image.Image,
        controlnet_type: str = "canny",
        target_width: int = 512,
        target_height: int = 512,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> tuple[Image.Image, Image.Image, int, int]:
        """
        Preprocesa imagen para ControlNet
        
        Args:
            image: Imagen PIL de entrada
            controlnet_type: Tipo de preprocessor
            target_width: Ancho objetivo (forzado, no mantiene aspect ratio)
            target_height: Alto objetivo (forzado, no mantiene aspect ratio)
            low_threshold: Umbral bajo para Canny (default 100)
            high_threshold: Umbral alto para Canny (default 200)
        
        Returns:
            (imagen_redimensionada, imagen_procesada, final_width, final_height)
        """
        # Asegurar que dimensiones sean múltiplos de 8 (requerido por SD)
        final_width = (target_width // 8) * 8
        final_height = (target_height // 8) * 8
        
        # Redimensionar a dimensiones exactas (forzar, ignorar aspect ratio)
        resized_image = image.resize((final_width, final_height), Image.LANCZOS)
        
        # Aplicar preprocessor sobre la imagen ya redimensionada
        if controlnet_type == "canny":
            detector = CannyDetector()
            processed = detector(resized_image, low_threshold=low_threshold, high_threshold=high_threshold)
        else:
            # Por ahora solo soportamos Canny
            # TODO: Agregar OpenPose y Depth
            raise NotImplementedError(f"Preprocessor {controlnet_type} no implementado aún")
        
        return resized_image, processed, final_width, final_height
    
    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.5,
        controlnet_scale: float = 1.0,
        seed: Optional[int] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Genera imagen con ControlNet
        
        Args:
            prompt: Prompt de generación
            control_image: Imagen de control (ya procesada)
            negative_prompt: Negative prompt (opcional)
            width: Ancho
            height: Alto
            steps: Pasos de inferencia
            cfg_scale: CFG scale
            controlnet_scale: Escala de ControlNet
            seed: Seed (opcional)
            save: Guardar imagen
        
        Returns:
            Dict con imagen y metadata
        """
        if not self.current_pipeline:
            raise RuntimeError("Debes cargar un pipeline ControlNet primero")
        
        # Negative prompt
        if not negative_prompt:
            negative_prompt = self.model_manager.get_negative_prompt()
        
        # Seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        device = "cpu"  # Generator siempre en CPU para compatibilidad
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generar
        with torch.inference_mode():
            result = self.current_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                controlnet_conditioning_scale=controlnet_scale,
                generator=generator
            )
        
        image = result.images[0]
        del result
        self._cleanup()
        
        # Guardar
        filepath = None
        if save:
            filepath = self._save_image(
                image, prompt, width, height, steps, cfg_scale, seed
            )
        
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "controlnet_scale": controlnet_scale,
            "controlnet_type": self.current_controlnet_type,
            "seed": seed,
            "model_id": self.model_manager.current_model["model_id"],
            "model_name": self.model_manager.current_model["model_name"],
            "filepath": str(filepath.relative_to(OUTPUTS_DIR.parent)) if filepath else None
        }
        
        return {
            "success": True,
            "image": image,
            "metadata": metadata
        }
    
    def generate_batch(
        self,
        prompts: List[str],
        control_image: Image.Image,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.5,
        controlnet_scale: float = 1.0,
        seeds: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Genera batch de imágenes con ControlNet
        
        Args:
            prompts: Lista de prompts
            control_image: Imagen de control (misma para todos)
            ... (resto de parámetros)
        
        Returns:
            Lista de resultados
        """
        if seeds is None:
            seeds = [torch.randint(0, 2**32, (1,)).item() for _ in prompts]
        
        results = []
        for i, prompt in enumerate(prompts):
            result = self.generate(
                prompt=prompt,
                control_image=control_image,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                controlnet_scale=controlnet_scale,
                seed=seeds[i],
                save=True
            )
            results.append(result)
        
        return results
    
    def _save_image(
        self,
        image: Image.Image,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int
    ) -> Path:
        """Guarda imagen con nombre descriptivo"""
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_manager.current_model["model_name"]
        safe_model_name = model_name.lower().replace(" ", "_").replace("/", "_")
        
        filename = f"controlnet_{self.current_controlnet_type}_{safe_model_name}_{width}x{height}_{timestamp}_{seed}.png"
        filepath = OUTPUTS_DIR / filename
        
        image.save(filepath)
        return filepath
    
    def unload(self):
        """Descarga pipeline ControlNet"""
        if self.current_pipeline:
            del self.current_pipeline
            self._cleanup()
        
        self.current_pipeline = None
        self.current_controlnet_type = None
        self.current_model_id = None
        
        return {"success": True, "message": "ControlNet pipeline descargado"}
    
    def clear_cache(self):
        """Limpia cache de ControlNet models"""
        for key in list(self.controlnet_cache.keys()):
            del self.controlnet_cache[key]
        self.controlnet_cache = {}
        self._cleanup()
    
    @staticmethod
    def _cleanup():
        """Limpieza de memoria"""
        torch.cuda.empty_cache()
        gc.collect()
