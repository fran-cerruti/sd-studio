# SD-Studio

A lightweight, optimized Stable Diffusion web interface with ControlNet support. Built for efficiency on consumer GPUs.

## Features

- **Simple Generation**: Text-to-image with customizable parameters
- **Batch Processing**: Generate multiple variations automatically
- **ControlNet**: Image-guided generation with Canny edge detection
- **Gallery**: Browse, search, and manage generated images
- **GPU Monitoring**: Real-time VRAM, utilization, and temperature stats
- **Network Access**: Use from any device on your local network

## Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (tested on GTX 1060 3GB+)
- **VRAM**: Minimum 3GB (4GB+ recommended)
- **OS**: Linux (Ubuntu/Debian tested), Windows with WSL2

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd image_gallery
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Models

Create a `models/` directory and download Stable Diffusion models:

```bash
mkdir -p models
```

**Recommended models:**
- [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [Realistic Vision](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE)
- Any `.safetensors` model compatible with Stable Diffusion 1.5

Place `.safetensors` files in the `models/` directory.

### 5. Start Server

```bash
python generation_api_v2.py
```

The server will start on `http://localhost:8080`

### 6. Access Interface

- **Local**: Open browser to `http://localhost:8080`
- **Network**: From other devices, use `http://YOUR_IP:8080`

## Configuration

### GPU Optimizations

The application automatically applies optimizations for low-VRAM GPUs:
- Attention slicing
- VAE slicing
- Sequential CPU offload for `.safetensors` models
- Model CPU offload for HuggingFace models

### Firewall (Linux)

To access from other devices on your network:

```bash
sudo ufw allow 8080/tcp
```

### Finding Your IP

```bash
ip addr show | grep "inet 192"
```

## Usage

### Generate Tab

1. Select or load a model
2. Enter your prompt
3. Adjust parameters (resolution, steps, CFG scale)
4. Click "Generate"

### Batch Tab

1. Add multiple prompts
2. Configure variations (CFG, steps, resolutions)
3. Generate all combinations automatically

### ControlNet Tab

1. Upload or select an image from gallery
2. Adjust Canny edge detection parameters
3. Preview edges
4. Generate with prompt guidance

### Gallery Tab

1. Browse all generated images
2. Search by filename
3. Sort by date or name
4. View fullscreen and navigate
5. Delete unwanted images

## Project Structure

```
image_gallery/
├── generation_api_v2.py    # Main API server
├── config.py                # Configuration and optimizations
├── model_manager.py         # Model loading and management
├── batch_processor.py       # Batch generation engine
├── controlnet_manager.py    # ControlNet integration
├── frontend/                # Web interface
│   ├── index.html
│   ├── app.js
│   └── favicon.ico
├── models/                  # Place .safetensors models here
├── outputs/                 # Generated images
└── requirements.txt         # Python dependencies
```

## API Endpoints

- `GET /` - Web interface
- `POST /generate` - Simple generation
- `POST /batch/generate` - Batch generation
- `POST /controlnet/generate` - ControlNet generation
- `GET /gallery` - List generated images
- `GET /gpu-info` - GPU statistics
- `GET /models` - List available models
- `POST /load-model` - Load a model

Full API documentation available at `http://localhost:8080/docs`

## Troubleshooting

### Out of Memory

- Reduce image resolution (512x512 or lower)
- Reduce number of inference steps
- Close other GPU applications
- Use models optimized for low VRAM

### Slow Generation

- Ensure CUDA is properly installed
- Check GPU utilization in header stats
- Consider enabling xformers (uncomment in requirements.txt)

### Model Not Loading

- Verify model file is in `models/` directory
- Check file format is `.safetensors`
- Ensure model is compatible with Stable Diffusion 1.5

### Network Access Issues

- Verify firewall allows port 8080
- Check devices are on same network
- Use IP address, not hostname

## License

Private project. All rights reserved.

## Credits

Built with:
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Vue.js](https://vuejs.org/)
