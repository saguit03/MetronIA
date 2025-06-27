#!/usr/bin/env python3
"""
Analizador de interpretaciones musicales para MetronIA.

Este script analiza la diferencia entre un audio de referencia y uno en vivo,
generando un CSV con los resultados detallados de onsets y gráficas de análisis.

Uso:
    python analizador.py [nombre_analisis] <ruta_referencia> <live_paths> 

Args:
    ruta_referencia: Ruta al archivo de audio de referencia
    ruta_en_vivo: Ruta al archivo de audio en vivo
    nombre_analisis: Nombre para el análisis

Ejemplos:
    python analizador.py mi_analisis audio/reference.wav audio/live.wav
    python analizador.py mi_analisis audio/reference.wav audio/live-1.wav audio/live-2.wav audio/live-3.wav
"""

import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from analyzers import MetronIA


def validate_arguments() -> tuple[str, str, Optional[str]]:
    if len(sys.argv) < 3:
        print("❌ Error: Número incorrecto de argumentos")
        print("\n📖 Uso:")
        print("    python analizador.py <ruta_referencia> <ruta_en_vivo> [nombre_analisis]")
        print("\n📝 Ejemplos:")
        print("    python analizador.py mi_analisis audio/reference.wav audio/live.wav")
        print(
            "    python analizador.py mi_analisis audio/reference.wav audio/live-1.wav audio/live-2.wav audio/live-3.wav")
        sys.exit(1)

    analysis_name = sys.argv[1]
    if analysis_name:
        analysis_name = re.sub(r'[^a-zA-Z0-9_-]', '_', analysis_name)

    ref_path = sys.argv[2]
    if not os.path.exists(ref_path):
        print(f"❌ Error: El archivo de referencia '{ref_path}' no existe")
        sys.exit(1)

    live_paths = sys.argv[3:]
    for ruta in live_paths:
        if not os.path.exists(ruta):
            live_paths.remove(ruta)

    if len(live_paths) == 0:
        print("❌ Error: No se proporcionaron archivos en vivo válidos")
        sys.exit(1)
    return analysis_name, ref_path, live_paths


def generate_analysis_name(live_path: str) -> str:
    live_name = Path(live_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_name = f"{live_name}_{timestamp}"
    return analysis_name


def main():
    print("=" * 70)
    print("🎵 MetronIA - Análisis de Sincronía de ritmos en audios")
    print("=" * 70)

    analysis_name, ref_path, live_paths = validate_arguments()

    try:
        analyzer = MetronIA()

        save_dir = Path(f"results") / analysis_name
        save_dir.mkdir(parents=True, exist_ok=True)

        analysis_progress = tqdm(live_paths, desc="Procesando archivos MIDI", unit="archivo", dynamic_ncols=True)
        for live_path in analysis_progress:
            analysis_progress.set_description(desc=f"Analizando {live_path}")
            if len(live_paths) > 1:
                save_live_dir = save_dir / Path(live_path).stem
            else:
                save_live_dir = save_dir
            save_live_dir.mkdir(parents=True, exist_ok=True)
            analyzer.comprehensive_analysis(
                reference_path=ref_path,
                live_path=live_path,
                save_name=analysis_name,
                save_dir=save_live_dir,
                verbose=True
            )
            tqdm.write(f"✅ Análisis completado para '{live_path}'")

        print(f"Análisis completado exitosamente para '{analysis_name}' con los archivos:")
        analysis_progress.close()
        for live_path in live_paths:
            print(f"  - {live_path}")
        print(f"Los resultados se guardaron en: {save_dir}")

    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
