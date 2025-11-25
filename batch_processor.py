#!/usr/bin/env python3
"""
Batch Processor - Motor de generación batch avanzado
Soporta variaciones de prompts, parámetros y seeds
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import torch
import uuid
import gc
from PIL import Image

from config import OUTPUTS_DIR, GTX_1060_CONFIG


class BatchProcessor:
    """Procesador de generación batch con variaciones"""
    
    def __init__(self, model_manager):
        """
        Args:
            model_manager: Instancia de ModelManager
        """
        self.model_manager = model_manager
        self.current_job = None
    
    def create_batch_job(
        self,
        prompts: List[str],
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.0,
        seeds: Optional[List[int]] = None,
        variations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crea un trabajo batch
        
        Args:
            prompts: Lista de prompts a generar
            negative_prompt: Negative prompt (usa default del modelo si None)
            width: Ancho de imagen
            height: Alto de imagen
            steps: Pasos de inferencia
            cfg_scale: CFG scale
            seeds: Lista de seeds (opcional, se generan random si None)
            variations: Variaciones de parámetros (opcional)
                {
                    "cfg_scales": [5.0, 7.0, 9.0],
                    "steps": [20, 30, 40],
                    "resolutions": [(512, 512), (768, 768)]
                }
        
        Returns:
            Job dict con configuración
        """
        job_id = str(uuid.uuid4())
        
        # Calcular total de imágenes
        total_images = len(prompts)
        
        if variations:
            if "cfg_scales" in variations:
                total_images *= len(variations["cfg_scales"])
            if "steps" in variations:
                total_images *= len(variations["steps"])
            if "resolutions" in variations:
                total_images *= len(variations["resolutions"])
        
        # Generar seeds si no se proveen
        if seeds is None:
            seeds = [torch.randint(0, 2**32, (1,)).item() for _ in range(len(prompts))]
        
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "current_image": 0,
            "total_images": total_images,
            "images": [],
            "metadata": [],
            "error": None,
            "started_at": None,
            "completed_at": None,
            "config": {
                "prompts": prompts,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seeds": seeds,
                "variations": variations
            }
        }
        
        return job
    
    def process_batch(
        self,
        job: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Procesa un trabajo batch
        
        Args:
            job: Job dict creado por create_batch_job
            progress_callback: Función callback para reportar progreso
        
        Returns:
            Job actualizado con resultados
        """
        self.current_job = job
        job["status"] = "processing"
        job["started_at"] = datetime.now().isoformat()
        
        try:
            pipe = self.model_manager.get_pipeline()
            config = job["config"]
            
            # Obtener negative prompt
            negative_prompt = config["negative_prompt"]
            if not negative_prompt:
                negative_prompt = self.model_manager.get_negative_prompt()
            
            device = GTX_1060_CONFIG["device"]
            
            # Generar imágenes
            image_counter = 0
            
            for prompt_idx, prompt in enumerate(config["prompts"]):
                seed = config["seeds"][prompt_idx]
                
                # Si hay variaciones, iterar sobre ellas
                if config["variations"]:
                    self._process_variations(
                        pipe, prompt, negative_prompt, seed,
                        config, job, image_counter, progress_callback
                    )
                else:
                    # Generación simple
                    self._generate_image(
                        pipe, prompt, negative_prompt, seed,
                        config["width"], config["height"],
                        config["steps"], config["cfg_scale"],
                        job, image_counter, progress_callback
                    )
                    image_counter += 1
            
            # Completado
            job["status"] = "completed"
            job["progress"] = 100.0
            job["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = datetime.now().isoformat()
            self._cleanup()
        
        self.current_job = None
        return job
    
    def _process_variations(
        self,
        pipe,
        prompt: str,
        negative_prompt: str,
        seed: int,
        config: Dict[str, Any],
        job: Dict[str, Any],
        start_counter: int,
        progress_callback: Optional[Callable]
    ):
        """Procesa variaciones de parámetros"""
        variations = config["variations"]
        counter = start_counter
        
        # Valores base
        cfg_scales = variations.get("cfg_scales", [config["cfg_scale"]])
        steps_list = variations.get("steps", [config["steps"]])
        resolutions = variations.get("resolutions", [(config["width"], config["height"])])
        
        # Iterar sobre todas las combinaciones
        for cfg in cfg_scales:
            for steps in steps_list:
                for width, height in resolutions:
                    self._generate_image(
                        pipe, prompt, negative_prompt, seed,
                        width, height, steps, cfg,
                        job, counter, progress_callback
                    )
                    counter += 1
    
    def _generate_image(
        self,
        pipe,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        job: Dict[str, Any],
        image_number: int,
        progress_callback: Optional[Callable]
    ):
        """Genera una imagen individual"""
        device = GTX_1060_CONFIG["device"]
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generar
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=1
            )
        
        image = result.images[0]
        del result
        self._cleanup()
        
        # Guardar
        filepath = self._save_image(
            image, prompt, width, height, steps, cfg_scale, seed
        )
        
        # Metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "model_id": self.model_manager.current_model["model_id"],
            "model_name": self.model_manager.current_model["model_name"]
        }
        
        # Actualizar job
        job["images"].append(str(filepath.relative_to(OUTPUTS_DIR.parent)))
        job["metadata"].append(metadata)
        job["current_image"] = image_number + 1
        job["progress"] = ((image_number + 1) / job["total_images"]) * 100
        
        # Callback
        if progress_callback:
            progress_callback(job)
        
        # Limpiar imagen
        del image
        self._cleanup()
    
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
        
        # Truncar prompt para filename
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '_')).strip()
        safe_prompt = safe_prompt.replace(" ", "_")
        
        filename = f"{safe_model_name}_{width}x{height}_s{steps}_cfg{cfg_scale}_{timestamp}_{seed}.png"
        filepath = OUTPUTS_DIR / filename
        
        image.save(filepath)
        return filepath
    
    def get_current_job(self) -> Optional[Dict[str, Any]]:
        """Retorna el job actual"""
        return self.current_job
    
    @staticmethod
    def _cleanup():
        """Limpieza de memoria"""
        torch.cuda.empty_cache()
        gc.collect()


class BatchJobQueue:
    """Cola de trabajos batch"""
    
    def __init__(self, batch_processor: BatchProcessor):
        self.processor = batch_processor
        self.queue: List[Dict[str, Any]] = []
        self.completed: List[Dict[str, Any]] = []
    
    def add_job(self, job: Dict[str, Any]):
        """Agrega job a la cola"""
        self.queue.append(job)
    
    def process_next(self, progress_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """Procesa el siguiente job en la cola"""
        if not self.queue:
            return None
        
        job = self.queue.pop(0)
        result = self.processor.process_batch(job, progress_callback)
        self.completed.append(result)
        return result
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Estado de la cola"""
        return {
            "queued": len(self.queue),
            "completed": len(self.completed),
            "current": self.processor.get_current_job()
        }
