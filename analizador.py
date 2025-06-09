#!/usr/bin/env python3
"""
Analizador de interpretaciones musicales para MetronIA.

Este script analiza la diferencia entre un audio de referencia y uno en vivo,
generando un CSV con los resultados y gráficas de análisis.

Uso:
    python analizador.py <ruta_referencia> <ruta_en_vivo> <nombre_analisis>

Args:
    ruta_referencia: Ruta relativa al archivo de audio de referencia
    ruta_en_vivo: Ruta relativa al archivo de audio en vivo/interpretación
    nombre_analisis: Nombre para el directorio de resultados (sin espacios)

Ejemplo:
    python analizador.py audio/reference.wav audio/live.wav mi_analisis
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from analyzers import MusicAnalyzer
# Imports del proyecto
from analyzers import analyze_performance


def validate_arguments() -> tuple[str, str, str]:
    """
    Valida los argumentos de línea de comandos.
    
    Returns:
        Tupla con (ruta_referencia, ruta_en_vivo, nombre_analisis)
    """
    if len(sys.argv) != 4:
        print("❌ Error: Número incorrecto de argumentos")
        print("\n📖 Uso:")
        print("    python analizador.py <ruta_referencia> <ruta_en_vivo> <nombre_analisis>")
        print("\n📝 Ejemplo:")
        print("    python analizador.py audio/reference.wav audio/live.wav mi_analisis")
        sys.exit(1)
    
    ruta_referencia = sys.argv[1]
    ruta_en_vivo = sys.argv[2]
    nombre_analisis = sys.argv[3]
    
    # Validar que los archivos existen
    if not os.path.exists(ruta_referencia):
        print(f"❌ Error: El archivo de referencia '{ruta_referencia}' no existe")
        sys.exit(1)
    
    if not os.path.exists(ruta_en_vivo):
        print(f"❌ Error: El archivo en vivo '{ruta_en_vivo}' no existe")
        sys.exit(1)
    
    # Validar nombre del análisis (sin caracteres problemáticos)
    if not nombre_analisis.replace('_', '').replace('-', '').isalnum():
        print(f"❌ Error: El nombre del análisis '{nombre_analisis}' contiene caracteres no válidos")
        print("💡 Use solo letras, números, guiones (-) y guiones bajos (_)")
        sys.exit(1)
    
    return ruta_referencia, ruta_en_vivo, nombre_analisis


def create_analysis_directory(nombre_analisis: str) -> Path:
    """
    Crea el directorio de análisis dentro de results/.
    
    Args:
        nombre_analisis: Nombre del análisis
        
    Returns:
        Path del directorio creado
    """
    results_dir = Path("results")
    analysis_dir = results_dir / nombre_analisis
    
    # Crear directorios
    results_dir.mkdir(exist_ok=True)
    analysis_dir.mkdir(exist_ok=True)
    
    return analysis_dir


def save_analysis_to_csv(analysis_data: Dict[str, Any], output_file: Path, 
                        ref_path: str, live_path: str, analysis_name: str):
    """
    Guarda los resultados de análisis en formato CSV.
    
    Args:
        analysis_data: Diccionario con los resultados del análisis
        output_file: Ruta del archivo CSV de salida
        ref_path: Ruta del archivo de referencia
        live_path: Ruta del archivo en vivo
        analysis_name: Nombre del análisis
    """
    print(f"💾 Guardando resultados en CSV...")
    
    # Extraer datos para CSV
    analyzer = MusicAnalyzer()
    csv_row = analyzer.extract_analysis_for_csv(
        beat_result=analysis_data['beat_spectrum'],
        onset_result=analysis_data['onsets'],
        tempo_result=analysis_data['tempo'],
        segment_result=analysis_data['segments'],
        dtw_regular=analysis_data['dtw_regular'],
        rhythm_errors=analysis_data['rhythm_errors'],
        mutation_category="manual_analysis",
        mutation_name=analysis_name
    )
    
    # Añadir información adicional del análisis
    csv_row['reference_file'] = ref_path
    csv_row['live_file'] = live_path
    csv_row['analysis_name'] = analysis_name
    
    # Reordenar columnas para que la información del análisis esté al principio
    ordered_data = {
        'analysis_name': csv_row['analysis_name'],
        'reference_file': csv_row['reference_file'],
        'live_file': csv_row['live_file'],
        'mutation_category': csv_row['mutation_category'],
        'mutation_name': csv_row['mutation_name'],
    }
    
    # Añadir el resto de datos (excluyendo los que ya agregamos)
    for key, value in csv_row.items():
        if key not in ordered_data:
            ordered_data[key] = value
    
    # Crear DataFrame y guardar
    df = pd.DataFrame([ordered_data])
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ CSV guardado en: {output_file}")


def analyze_audio_files(ref_path: str, live_path: str, analysis_name: str) -> Dict[str, Any]:
    """
    Realiza el análisis completo entre los dos archivos de audio.
    
    Args:
        ref_path: Ruta al archivo de referencia
        live_path: Ruta al archivo en vivo
        analysis_name: Nombre del análisis para las gráficas
        
    Returns:
        Diccionario con todos los resultados del análisis
    """
    print(f"🔬 Iniciando análisis de interpretación musical...")
    print(f"   📄 Referencia: {ref_path}")
    print(f"   🎤 En vivo: {live_path}")
    
    try:
        # Realizar análisis completo con gráficas
        analysis_result = analyze_performance(
            reference_path=ref_path,
            live_path=live_path,
            save_name=analysis_name,
            config=None,  # Usar configuración por defecto
            verbose=True   # Mostrar resultados detallados por pantalla
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
        if plot_file.is_file() and plot_file.suffix in ['.png', '.jpg', '.jpeg', '.pdf']:
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
    print("🎵 ANALIZADOR METRONIA - ANÁLISIS DE INTERPRETACIÓN MUSICAL")
    print("=" * 70)
    
    # 1. VALIDAR ARGUMENTOS
    print(f"\n🔍 Validando argumentos...")
    ref_path, live_path, analysis_name = validate_arguments()
    print(f"✅ Argumentos validados correctamente")
    
    # 2. CREAR DIRECTORIO DE ANÁLISIS
    print(f"\n📁 Preparando directorio de resultados...")
    analysis_dir = create_analysis_directory(analysis_name)
    print(f"✅ Directorio creado: {analysis_dir.absolute()}")
    
    # 3. REALIZAR ANÁLISIS
    print(f"\n🎼 Cargando y analizando archivos de audio...")
    analysis_result = analyze_audio_files(ref_path, live_path, analysis_name)
    
    # 4. GUARDAR RESULTADOS EN CSV
    print(f"\n📊 Procesando resultados...")
    csv_file = analysis_dir / f"analysis_{analysis_name}.csv"
    save_analysis_to_csv(analysis_result, csv_file, ref_path, live_path, analysis_name)
    
    # 5. MOVER GRÁFICAS AL DIRECTORIO DE ANÁLISIS
    print(f"\n🖼️ Organizando gráficas...")
    move_plots_to_analysis_directory(analysis_name, analysis_dir)
    
    # 6. RESUMEN FINAL
    print(f"\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
    print(f"📂 Resultados guardados en: {analysis_dir.absolute()}")
    print(f"📊 Archivos generados:")
    print(f"   📄 CSV de resultados: {csv_file.name}")
    
    # Contar gráficas en el directorio
    plot_files = list(analysis_dir.glob("*.png")) + list(analysis_dir.glob("*.jpg"))
    if plot_files:
        print(f"   📈 Gráficas generadas: {len(plot_files)} archivo(s)")
        for plot_file in sorted(plot_files):
            print(f"      - {plot_file.name}")
    
    print(f"\n💡 Para ver los resultados, revise el directorio: {analysis_dir}")


if __name__ == "__main__":
    main()
