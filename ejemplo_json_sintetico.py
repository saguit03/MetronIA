#!/usr/bin/env python3
"""
Ejemplo simple que crea un OnsetDTWAnalysisResult de ejemplo y lo exporta a JSON.

Este script crea datos de ejemplo sintéticos para demostrar la estructura JSON
sin necesidad de archivos de audio reales.
"""

import sys
import numpy as np
from pathlib import Path

# Agregar el directorio actual al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.onset_results import OnsetDTWAnalysisResult, OnsetMatchClassified, OnsetType

def create_mock_dtw_result():
    """Crea un OnsetDTWAnalysisResult de ejemplo con datos sintéticos."""
    
    # Crear algunos matches de ejemplo
    matches = [
        OnsetMatchClassified(
            ref_onset=1.0,
            live_onset=1.02,
            ref_pitch=440.0,
            live_pitch=442.0,
            time_adjustment=20.0,  # 20ms de retraso
            pitch_similarity=0.98,
            classification=OnsetType.LATE
        ),
        OnsetMatchClassified(
            ref_onset=1.5,
            live_onset=1.48,
            ref_pitch=523.25,
            live_pitch=525.0,
            time_adjustment=-20.0,  # 20ms adelantado
            pitch_similarity=0.95,
            classification=OnsetType.EARLY
        ),
        OnsetMatchClassified(
            ref_onset=2.0,
            live_onset=2.0,
            ref_pitch=659.25,
            live_pitch=660.0,
            time_adjustment=0.0,  # Perfecto
            pitch_similarity=0.99,
            classification=OnsetType.CORRECT
        ),
        OnsetMatchClassified(
            ref_onset=2.5,
            live_onset=2.53,
            ref_pitch=880.0,
            live_pitch=882.0,
            time_adjustment=30.0,  # 30ms de retraso
            pitch_similarity=0.96,
            classification=OnsetType.LATE
        ),
        OnsetMatchClassified(
            ref_onset=3.0,
            live_onset=3.0,
            ref_pitch=1046.5,
            live_pitch=1048.0,
            time_adjustment=0.0,  # Perfecto
            pitch_similarity=0.97,
            classification=OnsetType.CORRECT
        )
    ]
    
    # Crear algunos onsets faltantes y extra
    missing_onsets = [
        (3.5, 1174.66),  # Un onset que debería estar pero no está
        (4.0, 1318.51)   # Otro onset faltante
    ]
    
    extra_onsets = [
        (2.25, 740.0),   # Un onset extra que no debería estar
        (3.75, 987.77)   # Otro onset extra
    ]
    
    # Crear un path DTW de ejemplo
    dtw_path = np.array([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
        [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]
    ])
    
    # Crear el resultado
    result = OnsetDTWAnalysisResult(
        matches=matches,
        missing_onsets=missing_onsets,
        extra_onsets=extra_onsets,
        dtw_path=dtw_path,
        alignment_cost=0.123,
        tolerance_ms=1.0
    )
    
    return result

def main():
    """Función principal del ejemplo."""
    
    print("📄 EJEMPLO: EXPORTACIÓN DE OnsetDTWAnalysisResult SINTÉTICO A JSON")
    print("=" * 70)
    
    # Crear resultado de ejemplo
    print("🔄 Creando OnsetDTWAnalysisResult de ejemplo...")
    dtw_result = create_mock_dtw_result()
    
    # Mostrar estadísticas
    print("\n📊 Estadísticas del resultado de ejemplo:")
    dtw_result.print_summary()
    
    # Exportar a JSON
    json_path = "results/ejemplo_dtw_sintetico.json"
    print(f"\n💾 Exportando a JSON: {json_path}")
    
    dtw_result.export_to_json(
        filepath=json_path,
        mutation_category="ejemplo_sintetico",
        mutation_name="datos_mock",
        reference_path="audio/ejemplo_referencia.mp3",
        live_path="audio/ejemplo_live.mp3"
    )
    
    # Verificar archivo creado
    if Path(json_path).exists():
        file_size = Path(json_path).stat().st_size
        print(f"✅ Archivo JSON creado exitosamente ({file_size:,} bytes)")
        
        # Cargar y verificar
        print(f"\n🔄 Verificando carga desde JSON...")
        loaded_result = OnsetDTWAnalysisResult.from_json(json_path)
        
        print(f"📊 Comparación original vs cargado:")
        print(f"   Total matches: {dtw_result.total_matches} -> {loaded_result.total_matches}")
        print(f"   Alignment cost: {dtw_result.alignment_cost:.3f} -> {loaded_result.alignment_cost:.3f}")
        print(f"   Tolerance: {dtw_result.tolerance_ms:.1f} -> {loaded_result.tolerance_ms:.1f}")
        
        # Mostrar contenido JSON de forma legible
        print(f"\n📋 Contenido del archivo JSON:")
        print(f"   📁 Archivo: {json_path}")
        print(f"   🔍 Para ver el contenido completo, abre el archivo en un editor de texto")
        
        # Mostrar las primeras líneas del JSON
        print(f"\n📄 Primeras líneas del JSON:")
        with open(json_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:15]):  # Mostrar primeras 15 líneas
                print(f"   {i+1:2d}: {line.rstrip()}")
            if len(lines) > 15:
                print(f"   ... ({len(lines)-15} líneas más)")
        
        print(f"\n🎉 Ejemplo completado exitosamente!")
        print(f"📁 El archivo JSON está disponible en: {json_path}")
        
    else:
        print("❌ Error: No se pudo crear el archivo JSON")

if __name__ == "__main__":
    main()
