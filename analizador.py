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
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from analyzers import MusicAnalyzer

DEBUG = False

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

    if not os.path.exists(ruta_referencia):
        print(f"❌ Error: El archivo de referencia '{ruta_referencia}' no existe")
        sys.exit(1)

    if not os.path.exists(ruta_en_vivo):
        print(f"❌ Error: El archivo en vivo '{ruta_en_vivo}' no existe")
        sys.exit(1)

    nombre_analisis = sys.argv[3] if len(sys.argv) == 4 else Path(ruta_referencia).stem.split('_')[0]
    
    if nombre_analisis:
        nombre_analisis = re.sub(r'[^a-zA-Z0-9_-]', '_', nombre_analisis)

    return ruta_referencia, ruta_en_vivo, nombre_analisis


def generate_analysis_name(live_path: str) -> str:
    """
    Genera un nombre de análisis basado en el archivo en vivo y timestamp.
    
    Args:
        live_path: Ruta del archivo en vivo
        
    Returns:
        Nombre generado para el análisis
    """
    live_name = Path(live_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_name = f"{live_name}_{timestamp}"

    return analysis_name


def main():
    """Función principal del analizador."""

    print("=" * 70)
    print("🎵 MetronIA - Análisis de Sincronía de ritmos en audios")
    print("=" * 70)
    if DEBUG:
        ref_path, live_path, analysis_name = "audio/veneciana-reference.mp3", "audio/veneciana-live.mp3", "veneciana"
    else:
        ref_path, live_path, analysis_name = validate_arguments()
    
    try:
        analyzer = MusicAnalyzer()

        save_dir = Path(f"results") / analysis_name
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

if __name__ == "__main__":
    main()
