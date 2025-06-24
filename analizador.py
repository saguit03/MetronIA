#!/usr/bin/env python3
"""
Analizador de interpretaciones musicales para MetronIA.

Este script analiza la diferencia entre un audio de referencia y uno en vivo,
generando un CSV con los resultados detallados de onsets y gráficas de análisis.

Uso:
    python analizador.py <ruta_referencia> <ruta_en_vivo> [nombre_analisis]

Args:
    ruta_referencia: Ruta al archivo de audio de referencia
    ruta_en_vivo: Ruta al archivo de audio en vivo/interpretación
    nombre_analisis: Nombre para el análisis (opcional)

Ejemplos:
    python analizador.py audio/reference.wav audio/live.wav
    python analizador.py audio/reference.wav audio/live.wav mi_analisis
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from analyzers import MusicAnalyzer
# from analyzers.timeline import play_audio

def validate_arguments() -> tuple[str, str, Optional[str]]:
    """
    Valida los argumentos de línea de comandos.
    
    Returns:
        Tupla con (ruta_referencia, ruta_en_vivo, nombre_analisis)
    """
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("❌ Error: Número incorrecto de argumentos")
        print("\n📖 Uso:")
        print("    python analizador.py <ruta_referencia> <ruta_en_vivo> [nombre_analisis]")
        print("\n📝 Ejemplos:")
        print("    python analizador.py audio/reference.wav audio/live.wav")
        print("    python analizador.py audio/reference.wav audio/live.wav mi_analisis")
        sys.exit(1)

    ruta_referencia = sys.argv[1]
    ruta_en_vivo = sys.argv[2]

    # Validar que los archivos existen
    if not os.path.exists(ruta_referencia):
        print(f"❌ Error: El archivo de referencia '{ruta_referencia}' no existe")
        sys.exit(1)

    if not os.path.exists(ruta_en_vivo):
        print(f"❌ Error: El archivo en vivo '{ruta_en_vivo}' no existe")
        sys.exit(1)

    nombre_analisis = sys.argv[3] if len(sys.argv) == 4 else Path(ruta_referencia).stem.split('_')[0]
    # Validar nombre del análisis si se proporciona
    if nombre_analisis and not nombre_analisis.replace('_', '').replace('-', '').isalnum():
        print(f"❌ Error: El nombre del análisis '{nombre_analisis}' contiene caracteres no válidos")
        print("💡 Use solo letras, números, guiones (-) y guiones bajos (_)")
        sys.exit(1)

    return ruta_referencia, ruta_en_vivo, nombre_analisis


def generate_analysis_name(live_path: str) -> str:
    """
    Genera un nombre de análisis basado en el archivo en vivo y timestamp.
    
    Args:
        live_path: Ruta del archivo en vivo
        
    Returns:
        Nombre generado para el análisis
    """
    # Obtener nombre del archivo sin extensión
    live_name = Path(live_path).stem

    # Obtener timestamp actual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combinar para crear nombre único
    analysis_name = f"{live_name}_{timestamp}"

    return analysis_name


def analyze_audio_files(ref_path: str, live_path: str, analysis_name: str) -> Dict[str, Any]:
    """
    Realiza el análisis completo entre los dos archivos de audio.
    
    Args:
        ref_path: Ruta al archivo de referencia
        live_path: Ruta al archivo en vivo
        analysis_name: Nombre del análisis
        
    Returns:
        Diccionario con todos los resultados del análisis
    """

    try:
        # Crear analizador
        analyzer = MusicAnalyzer()

        save_dir = Path(f"results_{analysis_name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        analysis_result = analyzer.comprehensive_analysis(
            reference_path=ref_path,
            live_path=live_path,
            save_name=analysis_name,
            save_dir=save_dir
        )

        print(f"✅ Análisis completado exitosamente")
        return analysis_result

    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def move_plots_to_analysis_directory(analysis_name: str, analysis_dir: Path):
    """
    Mueve las gráficas generadas al directorio de análisis.
    
    Args:
        analysis_name: Nombre del análisis usado para las gráficas
        analysis_dir: Directorio de destino
    """
    print(f"📊 Organizando gráficas de análisis...")

    plots_dir = Path("analysis_plots")
    if not plots_dir.exists():
        print(f"⚠️ No se encontró el directorio de gráficas: {plots_dir}")
        return

    # Buscar archivos que contengan el nombre del análisis
    moved_files = 0
    for plot_file in plots_dir.glob(f"*{analysis_name}*"):
        if plot_file.is_file():
            destination = analysis_dir / plot_file.name
            try:
                plot_file.rename(destination)
                moved_files += 1
                print(f"   📈 Movido: {plot_file.name}")
            except Exception as e:
                print(f"   ⚠️ No se pudo mover {plot_file.name}: {e}")

    if moved_files > 0:
        print(f"✅ {moved_files} gráfica(s) movida(s) a: {analysis_dir}")
    else:
        print(f"⚠️ No se encontraron gráficas para mover")


def main():
    """Función principal del analizador."""

    print("=" * 70)
    print("🎵 MetronIA - Análisis de Sincronía de ritmos en audios")
    print("=" * 70)

    ref_path, live_path, analysis_name = validate_arguments()
    
    analysis_result = analyze_audio_files(ref_path, live_path, analysis_name)

    # Note: fig and ax are no longer returned to prevent memory leaks
    # play_audio(ref_path, analysis_result['fig'], analysis_result['ax'])

if __name__ == "__main__":
    main()
