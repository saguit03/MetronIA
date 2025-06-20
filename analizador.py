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

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from analyzers import MusicAnalyzer
from utils.audio_utils import check_extension

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


def create_results_directory() -> Path:
    """
    Crea el directorio de resultados si no existe.
    
    Returns:
        Path del directorio de resultados
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def print_analysis_summary(analysis_result: Dict[str, Any], analysis_name: str):
    """
    Imprime un resumen del análisis realizado.
    
    Args:
        analysis_result: Resultados del análisis
        analysis_name: Nombre del análisis
    """
    print(f"\n📊 RESUMEN DEL ANÁLISIS '{analysis_name}'")
    print("=" * 60)
    
    # Resumen de onsets
    dtw_onsets = analysis_result.get('dtw_onsets')
    if dtw_onsets:
        total_matches = len(dtw_onsets.matches)
        correct_matches = len([m for m in dtw_onsets.matches if m.classification.value == 'correct'])
        late_matches = len([m for m in dtw_onsets.matches if m.classification.value == 'late'])
        early_matches = len([m for m in dtw_onsets.matches if m.classification.value == 'early'])
        missing_onsets = len(dtw_onsets.missing_onsets)
        extra_onsets = len(dtw_onsets.extra_onsets)
        
        print(f"🎯 Análisis de Onsets:")
        print(f"   ✅ Correctos: {correct_matches}")
        print(f"   ⏰ Tarde: {late_matches}")
        print(f"   ⚡ Adelantados: {early_matches}")
        print(f"   ❌ Perdidos: {missing_onsets}")
        print(f"   ➕ Extra: {extra_onsets}")
        print(f"   📈 Total emparejados: {total_matches}")
        
        if total_matches > 0:
            accuracy = (correct_matches / total_matches) * 100
            print(f"   🎯 Precisión: {accuracy:.1f}%")
    
    # Resumen de tempo
    tempo_result = analysis_result.get('tempo')
    if tempo_result:
        print(f"\n🎵 Análisis de Tempo:")
        print(f"   📄 Referencia: {tempo_result.tempo_ref:.1f} BPM")
        print(f"   🎤 En vivo: {tempo_result.tempo_live:.1f} BPM")
        print(f"   📊 Diferencia: {tempo_result.difference:.1f} BPM")
        print(f"   ✅ Similar: {'Sí' if tempo_result.is_similar else 'No'}")
    
    # Resumen de beat spectrum
    beat_result = analysis_result.get('beat_spectrum')
    if beat_result:
        print(f"\n🎼 Análisis de Beat Spectrum:")
        print(f"   📊 Diferencia máxima: {beat_result.max_difference:.3f}")
        print(f"   ✅ Similar: {'Sí' if beat_result.is_similar else 'No'}")

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

        save_dir = Path("results") / analysis_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Realizar análisis completo
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
    
if __name__ == "__main__":
    main()
