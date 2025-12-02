#!/usr/bin/env python3
"""
SD-Studio API v2 - Backend FastAPI completo
Generación simple, batch y ControlNet
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import torch
import uuid
import asyncio
from PIL import Image
import io
import base64
import pynvml
import json

# Importar módulos propios
from model_manager import ModelManager
from batch_processor import BatchProcessor, BatchJobQueue
from controlnet_manager import ControlNetManager
from config import OUTPUTS_DIR

app = FastAPI(
    title="SD-Studio API",
    version="2.0.0",
    description="Framework personal de Stable Diffusion con batch y ControlNet"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar NVML para stats de GPU
try:
    pynvml.nvmlInit()
except Exception as e:
    print(f"Warning: Could not initialize NVML: {e}")
    print("GPU stats will use PyTorch fallback")

# Instancias globales
model_manager = ModelManager()
batch_processor = BatchProcessor(model_manager)
batch_queue = BatchJobQueue(batch_processor)
controlnet_manager = ControlNetManager(model_manager)

# Estado global para generaciones
generation_jobs = {}  # job_id -> job_data

# Archivos de configuración
PROMPTS_FILE = Path("saved_prompts.json")
LOGS_FILE = Path("generation_logs.jsonl")


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class LoadModelRequest(BaseModel):
    model_id: str

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = None
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    steps: int = Field(30, ge=10, le=100)
    cfg_scale: float = Field(7.0, ge=1.0, le=20.0)
    iterations: int = Field(1, ge=1, le=10)
    seed: Optional[int] = None

class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., min_items=1, max_items=50)
    negative_prompt: Optional[str] = None
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    steps: int = Field(30, ge=10, le=100)
    cfg_scale: float = Field(7.0, ge=1.0, le=20.0)
    seeds: Optional[List[int]] = None
    variations: Optional[Dict[str, Any]] = None

class ControlNetGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = None
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    steps: int = Field(30, ge=10, le=100)
    cfg_scale: float = Field(7.5, ge=1.0, le=20.0)
    controlnet_scale: float = Field(1.0, ge=0.0, le=2.0)
    controlnet_type: str = Field("canny", pattern="^(canny|openpose|depth)$")
    seed: Optional[int] = None

class LoadControlNetRequest(BaseModel):
    controlnet_type: str = Field("canny", pattern="^(canny|openpose|depth)$")


# ============================================================================
# ENDPOINTS - GENERAL
# ============================================================================

@app.get("/api")
def api_info():
    """API information endpoint"""
    return {
        "service": "SD-Studio API",
        "version": "2.0.0",
        "status": "running",
        "features": ["simple_generation", "batch_generation", "controlnet"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.current_model["loaded"],
        "controlnet_loaded": controlnet_manager.current_pipeline is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/gpu-info")
def get_gpu_info():
    """Información de la GPU usando NVML (misma fuente que nvtop)"""
    if not torch.cuda.is_available():
        return {
            "success": True,
            "cuda_available": False
        }
    
    try:
        # Usar NVML para stats reales (misma fuente que nvtop)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        
        # Uso de GPU (%)
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_percent = utilization.gpu
        except:
            gpu_percent = None
        
        # Temperatura (°C)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temp = None
        
        return {
            "success": True,
            "cuda_available": True,
            "device_name": name.decode('utf-8') if isinstance(name, bytes) else name,
            "total_memory_gb": round(info.total / 1024**3, 2),
            "allocated_memory_gb": round(info.used / 1024**3, 2),
            "free_memory_gb": round(info.free / 1024**3, 2),
            "gpu_utilization_percent": gpu_percent,
            "temperature_celsius": temp
        }
    except Exception as e:
        # Fallback a torch si NVML falla
        return {
            "success": True,
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "allocated_memory_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "error": f"NVML error, using PyTorch fallback: {str(e)}"
        }


# ============================================================================
# ENDPOINTS - MODELOS
# ============================================================================

@app.get("/models")
def list_models():
    """Lista todos los modelos disponibles"""
    models = model_manager.list_available_models()
    current = model_manager.get_current_model()
    
    return {
        "success": True,
        "models": models,
        "current_model": current
    }

@app.post("/load-model")
async def load_model(request: LoadModelRequest):
    """Carga un modelo específico"""
    try:
        result = model_manager.load_model(request.model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

@app.post("/unload-model")
def unload_model():
    """Descarga el modelo actual"""
    return model_manager.unload_model()


# ============================================================================
# ENDPOINTS - GENERACIÓN SIMPLE
# ============================================================================

@app.post("/generate")
async def generate_image(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Genera imagen(s) con el modelo cargado"""
    if not model_manager.current_model["loaded"]:
        raise HTTPException(status_code=400, detail="No hay modelo cargado. Usa /load-model primero")
    
    job_id = str(uuid.uuid4())
    
    job = {
        "job_id": job_id,
        "type": "simple",
        "status": "processing",
        "progress": 0.0,
        "current_iteration": 0,
        "total_iterations": request.iterations,
        "images": [],
        "error": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "request": request.dict()
    }
    
    generation_jobs[job_id] = job
    background_tasks.add_task(process_simple_generation, job)
    
    return {
        "success": True,
        "job_id": job_id,
        "message": f"Generación iniciada: {request.iterations} imagen(es)"
    }

def process_simple_generation(job: Dict[str, Any]):
    """Procesa generación simple"""
    try:
        request = GenerationRequest(**job["request"])
        pipe = model_manager.get_pipeline()
        negative_prompt = model_manager.get_negative_prompt(request.negative_prompt)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for i in range(request.iterations):
            job["current_iteration"] = i + 1
            
            # Seed
            if request.seed is not None:
                seed = request.seed + i
            else:
                seed = torch.randint(0, 2**32, (1,)).item()
            
            generator = torch.Generator(device=device).manual_seed(seed)
            
            # Generar
            with torch.inference_mode():
                result = pipe(
                    prompt=request.prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=request.steps,
                    guidance_scale=request.cfg_scale,
                    width=request.width,
                    height=request.height,
                    generator=generator,
                    num_images_per_prompt=1
                )
            
            image = result.images[0]
            del result
            ModelManager.cleanup()
            
            # Guardar
            filepath = save_image(
                image, request.prompt, request.width, request.height,
                request.steps, request.cfg_scale, seed
            )
            
            job["images"].append(str(filepath.relative_to(OUTPUTS_DIR.parent)))
            job["progress"] = ((i + 1) / request.iterations) * 100
            
            del image
            ModelManager.cleanup()
        
        job["status"] = "completed"
        job["progress"] = 100.0
        job["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        ModelManager.cleanup()


# ============================================================================
# ENDPOINTS - GENERACIÓN BATCH
# ============================================================================

@app.post("/batch/generate")
async def generate_batch(request: BatchGenerationRequest, background_tasks: BackgroundTasks):
    """Genera batch de imágenes con variaciones opcionales"""
    if not model_manager.current_model["loaded"]:
        raise HTTPException(status_code=400, detail="No hay modelo cargado")
    
    # Crear job
    job = batch_processor.create_batch_job(
        prompts=request.prompts,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        cfg_scale=request.cfg_scale,
        seeds=request.seeds,
        variations=request.variations
    )
    
    job_id = job["job_id"]
    generation_jobs[job_id] = job
    
    # Procesar en background
    background_tasks.add_task(process_batch_generation, job)
    
    return {
        "success": True,
        "job_id": job_id,
        "total_images": job["total_images"],
        "message": f"Batch iniciado: {job['total_images']} imágenes"
    }

def process_batch_generation(job: Dict[str, Any]):
    """Procesa generación batch"""
    def progress_callback(updated_job):
        # Actualizar job en dict global
        generation_jobs[updated_job["job_id"]] = updated_job
    
    try:
        batch_processor.process_batch(job, progress_callback)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()

@app.get("/batch/queue")
def get_batch_queue():
    """Estado de la cola batch"""
    return {
        "success": True,
        "queue": batch_queue.get_queue_status()
    }


# ============================================================================
# ENDPOINTS - CONTROLNET
# ============================================================================

@app.post("/controlnet/load")
async def load_controlnet(request: LoadControlNetRequest):
    """Carga pipeline ControlNet"""
    try:
        result = controlnet_manager.load_controlnet_pipeline(request.controlnet_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando ControlNet: {str(e)}")

@app.post("/controlnet/preprocess")
async def preprocess_controlnet_image(
    file: UploadFile = File(...),
    controlnet_type: str = Form("canny"),
    width: int = Form(512),
    height: int = Form(512),
    low_threshold: int = Form(100),
    high_threshold: int = Form(200)
):
    """Preprocesa imagen para ControlNet y retorna preview"""
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocesar con thresholds personalizados
        resized, processed, final_width, final_height = controlnet_manager.preprocess_image(
            image, controlnet_type, width, height, 
            low_threshold=low_threshold, high_threshold=high_threshold
        )
        
        # Convertir a base64 para preview
        buffered = io.BytesIO()
        processed.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "processed_image": f"data:image/png;base64,{img_str}",
            "dimensions": {
                "width": processed.width,
                "height": processed.height
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.post("/controlnet/generate")
async def generate_controlnet(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(512),
    height: int = Form(512),
    steps: int = Form(30),
    cfg_scale: float = Form(7.5),
    controlnet_scale: float = Form(1.0),
    controlnet_type: str = Form("canny"),
    seed: Optional[int] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Genera imagen con ControlNet"""
    if not controlnet_manager.current_pipeline:
        raise HTTPException(status_code=400, detail="Debes cargar ControlNet primero")
    
    try:
        # Leer y preprocesar imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        _, control_image, final_width, final_height = controlnet_manager.preprocess_image(
            image, controlnet_type, width, height
        )
        
        # Generar usando las dimensiones finales calculadas
        result = controlnet_manager.generate(
            prompt=prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            width=final_width,
            height=final_height,
            steps=steps,
            cfg_scale=cfg_scale,
            controlnet_scale=controlnet_scale,
            seed=seed,
            save=True
        )
        
        return {
            "success": True,
            "image_path": result["metadata"]["filepath"],
            "metadata": result["metadata"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando: {str(e)}")

@app.post("/controlnet/unload")
def unload_controlnet():
    """Descarga pipeline ControlNet"""
    return controlnet_manager.unload()


# ============================================================================
# ENDPOINTS - STATUS
# ============================================================================

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Obtiene estado de un job específico"""
    if job_id not in generation_jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    job = generation_jobs[job_id]
    
    return {
        "success": True,
        "job": {
            "job_id": job["job_id"],
            "type": job.get("type", "unknown"),
            "status": job["status"],
            "progress": job["progress"],
            "images": job["images"],
            "error": job.get("error"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at")
        }
    }

@app.get("/status")
def get_all_status():
    """Estado general de generaciones"""
    active_jobs = [j for j in generation_jobs.values() if j["status"] == "processing"]
    completed_jobs = [j for j in generation_jobs.values() if j["status"] == "completed"]
    failed_jobs = [j for j in generation_jobs.values() if j["status"] == "failed"]
    
    return {
        "success": True,
        "active": len(active_jobs),
        "completed": len(completed_jobs),
        "failed": len(failed_jobs),
        "jobs": {
            "active": active_jobs,
            "completed": completed_jobs[-10:],  # Últimos 10
            "failed": failed_jobs[-10:]
        }
    }


# ============================================================================
# UTILIDADES
# ============================================================================

def save_image(
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
    model_name = model_manager.current_model["model_name"]
    safe_model_name = model_name.lower().replace(" ", "_").replace("/", "_")
    
    filename = f"{safe_model_name}_{width}x{height}_s{steps}_cfg{cfg_scale}_{timestamp}_{seed}.png"
    filepath = OUTPUTS_DIR / filename
    
    image.save(filepath)
    return filepath


# ============================================================================
# ENDPOINTS ADICIONALES PARA FRONTEND
# ============================================================================

@app.get("/prompts")
async def get_saved_prompts():
    """Obtener prompts guardados"""
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE, 'r') as f:
            prompts = json.load(f)
        return {"success": True, "prompts": prompts}
    return {"success": True, "prompts": []}

@app.post("/prompts")
async def save_prompt(prompt: Dict[str, Any]):
    """Guardar un nuevo prompt"""
    prompts = []
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE, 'r') as f:
            prompts = json.load(f)
    
    prompts.append(prompt)
    
    with open(PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    return {"success": True, "message": "Prompt guardado"}

@app.delete("/prompts/{index}")
async def delete_prompt(index: int):
    """Eliminar un prompt guardado"""
    if not PROMPTS_FILE.exists():
        raise HTTPException(status_code=404, detail="No hay prompts guardados")
    
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
    
    if index < 0 or index >= len(prompts):
        raise HTTPException(status_code=404, detail="Prompt no encontrado")
    
    prompts.pop(index)
    
    with open(PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    return {"success": True, "message": "Prompt eliminado"}

@app.post("/log")
async def log_generation(log_data: Dict[str, Any]):
    """Registrar una generación en el log"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **log_data
    }
    
    with open(LOGS_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    return {"success": True}

@app.get("/logs")
async def get_logs(limit: int = 100):
    """Obtener últimos logs de generación"""
    if not LOGS_FILE.exists():
        return {"success": True, "logs": []}
    
    logs = []
    with open(LOGS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    
    # Retornar los últimos N logs
    return {"success": True, "logs": logs[-limit:]}

# Endpoint de galería
@app.get("/gallery")
async def get_gallery(sort: str = "date_desc"):
    """Obtener todas las imágenes de la galería"""
    if not OUTPUTS_DIR.exists():
        return {"success": True, "images": []}
    
    images = []
    for img_path in OUTPUTS_DIR.rglob("*.png"):
        images.append({
            "path": f"outputs/{img_path.relative_to(OUTPUTS_DIR)}",
            "name": img_path.name,
            "size": img_path.stat().st_size,
            "modified": img_path.stat().st_mtime
        })
    
    # Ordenar
    if sort == "date_desc":
        images.sort(key=lambda x: x["modified"], reverse=True)
    elif sort == "date_asc":
        images.sort(key=lambda x: x["modified"])
    elif sort == "name_asc":
        images.sort(key=lambda x: x["name"])
    elif sort == "name_desc":
        images.sort(key=lambda x: x["name"], reverse=True)
    
    return {"success": True, "images": images}

@app.delete("/gallery/{image_path:path}")
async def delete_image(image_path: str):
    """Eliminar una imagen del disco"""
    # Decodificar y sanitizar path
    img_path = OUTPUTS_DIR / image_path.replace("outputs/", "")
    
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not str(img_path.resolve()).startswith(str(OUTPUTS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Invalid path")
    
    img_path.unlink()
    return {"success": True, "message": "Image deleted"}

# Servir archivos estáticos (outputs e imágenes)
# IMPORTANTE: Estos deben ir ANTES del endpoint raíz
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root():
    """Redirect a frontend"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/frontend/index.html")


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Configuración del servidor
    HOST = "0.0.0.0"
    PORT = 8080
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*80)
    print("SD-STUDIO API v2.0 - Framework Personal de Stable Diffusion")
    print("="*80)
    print(f"📁 Outputs: {OUTPUTS_DIR.resolve()}")
    print(f"\n🌐 Acceso local:")
    print(f"   http://localhost:{PORT}")
    print(f"   http://127.0.0.1:{PORT}")
    print(f"\n🌐 Acceso desde red:")
    print(f"   http://{local_ip}:{PORT}")
    print(f"\n📚 Documentación:")
    print(f"   http://localhost:{PORT}/docs")
    print(f"\n✨ Features:")
    print(f"   • Generación simple")
    print(f"   • Generación batch con variaciones")
    print(f"   • ControlNet (Canny, OpenPose, Depth)")
    print(f"   • Modelos locales (.safetensors) + HuggingFace")
    print(f"   • Optimizado para GTX 1060 3GB")
    print("="*80)
    print("\nPresiona Ctrl+C para detener\n")
    
    uvicorn.run(app, host=HOST, port=PORT)
