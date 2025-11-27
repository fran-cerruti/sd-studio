#!/usr/bin/env python3
"""
SD-Studio Hardware Setup
Detecta tu GPU y optimiza la configuración automáticamente

Uso:
    python setup.py

Requisitos previos:
    pip install torch pynvml
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# Colores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Imprime header con formato"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """Imprime advertencia"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    """Imprime error"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Imprime información"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def detect_hardware():
    """
    Detecta hardware y retorna información completa
    
    Returns:
        Dict con información de GPU o None si falla
    """
    try:
        import torch
        import pynvml
    except ImportError as e:
        print_error(f"Dependencia faltante: {e}")
        print_info("Ejecuta: pip install torch pynvml")
        return None
    
    if not torch.cuda.is_available():
        print_error("No se detectó GPU CUDA")
        print_info("Este script requiere una GPU NVIDIA con CUDA")
        return None
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Información GPU
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        
        props = torch.cuda.get_device_properties(0)
        compute = f"{props.major}.{props.minor}"
        
        # Determinar arquitectura
        arch_map = {
            '3.0': 'Kepler', '3.5': 'Kepler', '3.7': 'Kepler',
            '5.0': 'Maxwell', '5.2': 'Maxwell', '5.3': 'Maxwell',
            '6.0': 'Pascal', '6.1': 'Pascal', '6.2': 'Pascal',
            '7.0': 'Volta', '7.2': 'Volta', '7.5': 'Turing',
            '8.0': 'Ampere', '8.6': 'Ampere', '8.9': 'Ada Lovelace',
            '9.0': 'Hopper'
        }
        arch = arch_map.get(compute, 'Unknown')
        
        # Determinar perfil
        if vram_gb < 4:
            profile = "LOW_VRAM"
        elif vram_gb < 7:
            profile = "MEDIUM_VRAM"
        else:
            profile = "HIGH_VRAM"
        
        # Compatibilidad xformers (Volta+ = SM 7.0+)
        xformers_compatible = float(compute) >= 7.0
        
        return {
            'name': name,
            'vram_gb': vram_gb,
            'compute': compute,
            'arch': arch,
            'profile': profile,
            'xformers_compatible': xformers_compatible,
            'torch_version': torch.__version__,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    except Exception as e:
        print_error(f"Error detectando hardware: {e}")
        return None

def get_profile_config(profile, xformers_compatible):
    """
    Retorna configuración según perfil
    
    Args:
        profile: Perfil de VRAM (LOW_VRAM, MEDIUM_VRAM, HIGH_VRAM)
        xformers_compatible: Si la GPU soporta xformers
    
    Returns:
        Dict con configuración del perfil
    """
    configs = {
        "LOW_VRAM": {
            "max_resolution": 768,
            "recommended_resolution": 512,
            "optimizations": {
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "enable_cpu_offload": True,
            },
            "requirements_extra": [],
            "description": "Optimizado para GPUs con < 4GB VRAM (GTX 1050 Ti, GTX 1060 3GB)"
        },
        "MEDIUM_VRAM": {
            "max_resolution": 1024,
            "recommended_resolution": 768,
            "optimizations": {
                "enable_attention_slicing": False,
                "enable_vae_slicing": True,
                "enable_cpu_offload": False,
            },
            "requirements_extra": ["xformers"] if xformers_compatible else [],
            "description": "Optimizado para GPUs con 4-6GB VRAM (GTX 1060 6GB, GTX 1650, RTX 2060)"
        },
        "HIGH_VRAM": {
            "max_resolution": 1536,
            "recommended_resolution": 1024,
            "optimizations": {
                "enable_attention_slicing": False,
                "enable_vae_slicing": False,
                "enable_cpu_offload": False,
            },
            "requirements_extra": ["xformers"] if xformers_compatible else [],
            "description": "Optimizado para GPUs con > 6GB VRAM (RTX 3060+, RTX 4060+)"
        }
    }
    return configs.get(profile, configs["LOW_VRAM"])

def show_recommendations(hw_info, config):
    """
    Muestra recomendaciones al usuario
    
    Args:
        hw_info: Dict con información de hardware
        config: Dict con configuración del perfil
    """
    print_header("HARDWARE DETECTADO")
    
    print(f"GPU: {Colors.BOLD}{hw_info['name']}{Colors.END}")
    print(f"VRAM: {Colors.BOLD}{hw_info['vram_gb']:.1f} GB{Colors.END}")
    print(f"Arquitectura: {Colors.BOLD}{hw_info['arch']} (SM {hw_info['compute']}){Colors.END}")
    print(f"PyTorch: {hw_info['torch_version']}")
    
    print_header("PERFIL RECOMENDADO")
    
    print(f"Perfil: {Colors.BOLD}{hw_info['profile']}{Colors.END}")
    print(f"Descripción: {config['description']}")
    
    print(f"\n{Colors.BOLD}Optimizaciones:{Colors.END}")
    for opt, enabled in config['optimizations'].items():
        status = f"{Colors.GREEN}✓ Activado{Colors.END}" if enabled else f"{Colors.YELLOW}✗ Desactivado{Colors.END}"
        opt_name = opt.replace('enable_', '').replace('_', ' ').title()
        print(f"  • {opt_name}: {status}")
    
    print(f"\n{Colors.BOLD}Resoluciones:{Colors.END}")
    print(f"  • Recomendada: {config['recommended_resolution']}x{config['recommended_resolution']}")
    print(f"  • Máxima: {config['max_resolution']}x{config['max_resolution']}")
    
    print(f"\n{Colors.BOLD}Dependencias adicionales:{Colors.END}")
    if config['requirements_extra']:
        for req in config['requirements_extra']:
            print(f"  • {req}")
            if req == "xformers":
                print_info("    xformers mejora velocidad ~30% y reduce VRAM ~40%")
    else:
        print("  • Ninguna")
    
    # Advertencias específicas
    if not hw_info['xformers_compatible']:
        print()
        print_warning(f"xformers NO compatible con {hw_info['arch']} (requiere Volta+ / SM 7.0+)")
        print(f"  → No se instalará xformers (normal para GPUs Pascal como GTX 10xx)")
        print(f"  → Tu GPU funcionará correctamente sin xformers")

def confirm_changes():
    """
    Pide confirmación al usuario
    
    Returns:
        bool: True si el usuario confirma, False si cancela
    """
    print_header("CONFIRMAR CAMBIOS")
    
    print("Este script modificará:")
    print("  1. config.py - Configuración de optimizaciones")
    print("  2. requirements.txt - Dependencias (si es necesario)")
    print("\nSe crearán backups automáticos en .setup_backup/")
    print("Puedes restaurar los archivos originales desde ahí si es necesario.")
    
    while True:
        response = input(f"\n{Colors.BOLD}¿Aplicar cambios? [Y/n]: {Colors.END}").strip().lower()
        if response in ['y', 'yes', 'sí', 'si', '']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Por favor responde 'y' (sí) o 'n' (no)")

def backup_files():
    """
    Crea backups de archivos antes de modificar
    """
    backup_dir = Path(".setup_backup")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = ["config.py", "requirements.txt"]
    
    for file in files_to_backup:
        file_path = Path(file)
        if file_path.exists():
            backup_path = backup_dir / f"{file}.backup"
            shutil.copy(file_path, backup_path)
            print_success(f"Backup: {file} → .setup_backup/{file}.backup")

def apply_config_changes(hw_info, config):
    """
    Modifica config.py usando template
    
    Args:
        hw_info: Dict con información de hardware
        config: Dict con configuración del perfil
    """
    from config_templates import generate_config
    
    new_config = generate_config(hw_info, config)
    
    with open("config.py", "w") as f:
        f.write(new_config)
    
    print_success("config.py actualizado")

def apply_requirements_changes(config):
    """
    Modifica requirements.txt si es necesario
    
    Args:
        config: Dict con configuración del perfil
    """
    if not config['requirements_extra']:
        print_info("requirements.txt sin cambios")
        return
    
    # Leer requirements actual
    with open("requirements.txt", "r") as f:
        lines = f.readlines()
    
    # Agregar dependencias nuevas
    added = []
    for req in config['requirements_extra']:
        req_line = f"{req}  # Added by setup.py\n"
        # Verificar si ya existe (sin importar comentarios)
        req_exists = any(line.strip().startswith(req) for line in lines)
        if not req_exists:
            lines.append(req_line)
            added.append(req)
    
    if added:
        # Escribir
        with open("requirements.txt", "w") as f:
            f.writelines(lines)
        
        print_success(f"requirements.txt actualizado (+{', '.join(added)})")
    else:
        print_info("requirements.txt ya contiene las dependencias necesarias")

def reinstall_dependencies():
    """
    Reinstala dependencias del venv
    
    Returns:
        bool: True si la instalación fue exitosa
    """
    print_header("REINSTALANDO DEPENDENCIAS")
    
    import subprocess
    
    print("Ejecutando: pip install -r requirements.txt")
    print("(Esto puede tomar varios minutos...)\n")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=False,  # Mostrar output en tiempo real
        text=True
    )
    
    if result.returncode == 0:
        print()
        print_success("Dependencias instaladas correctamente")
        return True
    else:
        print()
        print_error("Error instalando dependencias")
        print_warning("Puedes intentar manualmente: pip install -r requirements.txt")
        return False

def generate_report(hw_info, config):
    """
    Genera reporte de configuración
    
    Args:
        hw_info: Dict con información de hardware
        config: Dict con configuración del perfil
    """
    report_path = Path(".setup_backup") / "setup_report.txt"
    
    with open(report_path, "w") as f:
        f.write("SD-STUDIO SETUP REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fecha: {hw_info['date']}\n\n")
        
        f.write("HARDWARE:\n")
        f.write(f"  GPU: {hw_info['name']}\n")
        f.write(f"  VRAM: {hw_info['vram_gb']:.1f} GB\n")
        f.write(f"  Arquitectura: {hw_info['arch']} (SM {hw_info['compute']})\n")
        f.write(f"  PyTorch: {hw_info['torch_version']}\n\n")
        
        f.write("PERFIL:\n")
        f.write(f"  Tipo: {hw_info['profile']}\n")
        f.write(f"  Descripción: {config['description']}\n\n")
        
        f.write("OPTIMIZACIONES:\n")
        for opt, enabled in config['optimizations'].items():
            status = "Activado" if enabled else "Desactivado"
            f.write(f"  {opt}: {status}\n")
        
        f.write(f"\nRESOLUCIONES:\n")
        f.write(f"  Recomendada: {config['recommended_resolution']}x{config['recommended_resolution']}\n")
        f.write(f"  Máxima: {config['max_resolution']}x{config['max_resolution']}\n")
        
        if config['requirements_extra']:
            f.write(f"\nDEPENDENCIAS ADICIONALES:\n")
            for req in config['requirements_extra']:
                f.write(f"  {req}\n")
    
    print_success(f"Reporte guardado: .setup_backup/setup_report.txt")

def main():
    """Función principal del script"""
    print_header("SD-STUDIO HARDWARE SETUP")
    
    print("Este script detectará tu GPU y optimizará la configuración.")
    print("Es seguro ejecutarlo múltiples veces.")
    print("Se crearán backups automáticos antes de cualquier cambio.\n")
    
    # 1. Detectar hardware
    print("Detectando hardware...")
    hw_info = detect_hardware()
    
    if not hw_info:
        print()
        print_error("No se pudo detectar el hardware")
        print_info("Asegúrate de tener una GPU NVIDIA con CUDA instalado")
        sys.exit(1)
    
    # 2. Obtener configuración recomendada
    config = get_profile_config(hw_info['profile'], hw_info['xformers_compatible'])
    
    # 3. Mostrar recomendaciones
    show_recommendations(hw_info, config)
    
    # 4. Confirmar
    if not confirm_changes():
        print()
        print_info("Cancelado por el usuario.")
        print("No se realizaron cambios.")
        sys.exit(0)
    
    # 5. Aplicar cambios
    print_header("APLICANDO CAMBIOS")
    
    try:
        backup_files()
        apply_config_changes(hw_info, config)
        apply_requirements_changes(config)
        generate_report(hw_info, config)
    except Exception as e:
        print()
        print_error(f"Error aplicando cambios: {e}")
        print_warning("Puedes restaurar desde .setup_backup/ si es necesario")
        sys.exit(1)
    
    # 6. Reinstalar dependencias (opcional)
    print()
    response = input(f"{Colors.BOLD}¿Reinstalar dependencias ahora? [Y/n]: {Colors.END}").strip().lower()
    if response in ['y', 'yes', 'sí', 'si', '']:
        success = reinstall_dependencies()
        if not success:
            print()
            print_warning("Continúa manualmente con: pip install -r requirements.txt")
    else:
        print()
        print_info("Recuerda ejecutar: pip install -r requirements.txt")
    
    # 7. Resumen final
    print_header("SETUP COMPLETADO")
    print_success("Configuración optimizada para tu hardware")
    print(f"\nPerfil aplicado: {Colors.BOLD}{hw_info['profile']}{Colors.END}")
    print(f"GPU: {Colors.BOLD}{hw_info['name']}{Colors.END}")
    print(f"Resolución recomendada: {Colors.BOLD}{config['recommended_resolution']}x{config['recommended_resolution']}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Próximos pasos:{Colors.END}")
    print("  1. Descarga modelos en ../models/")
    print("  2. Ejecuta: python generation_api_v2.py")
    print("  3. Abre: http://localhost:8080")
    
    print(f"\n{Colors.BOLD}Información adicional:{Colors.END}")
    print("  • Backups: .setup_backup/")
    print("  • Reporte: .setup_backup/setup_report.txt")
    print("  • Puedes ejecutar este script nuevamente si cambias de GPU")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Cancelado por el usuario (Ctrl+C){Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Error inesperado: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
