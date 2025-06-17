#!/usr/bin/env python3
"""
Ejemplo de exportación de OnsetDTWAnalysisResult a JSON desde análisis real de audios.

Uso:
    python ejemplo_export_json.py <audio_referencia> <audio_live> [archivo_salida.json]

Ejemplos:
    python ejemplo_export_json.py audio/01-reference.mp3 audio/01-live.mp3
    python ejemplo_export_json.py audio/veneciana-reference.mp3 audio/veneciana-live.mp3 mi_analisis.json
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Agregar el directorio actual al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.music_analyzer import MusicAnalyzer
from analyzers.onset_results import OnsetDTWAnalysisResult


def validate_arguments():
    """Valida los argumentos de línea de comandos."""
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("❌ Error: Número incorrecto de argumentos")
        print("\n📖 Uso:")
        print("    python ejemplo_export_json.py <audio_referencia> <audio_live> [archivo_salida.json]")
        print("\n📝 Ejemplos:")
        print("    python ejemplo_export_json.py audio/01-reference.mp3 audio/01-live.mp3")
        print("    python ejemplo_export_json.py audio/veneciana-reference.mp3 audio/veneciana-live.mp3 mi_analisis.json")
        sys.exit(1)
    
    reference_path = sys.argv[1]
    live_path = sys.argv[2]
    
    # Validar que los archivos existen
    if not Path(reference_path).exists():
        print(f"❌ Error: No se encontró el archivo de referencia: {reference_path}")
        sys.exit(1)
    
    if not Path(live_path).exists():
        print(f"❌ Error: No se encontró el archivo live: {live_path}")
        sys.exit(1)
    
    # Determinar archivo de salida
    if len(sys.argv) == 4:
        output_path = sys.argv[3]
    else:
        # Generar nombre automático basado en los archivos de entrada
        ref_name = Path(reference_path).stem
        live_name = Path(live_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/analisis_{ref_name}_vs_{live_name}_{timestamp}.json"
    
    return reference_path, live_path, output_path


def analyze_and_export(reference_path, live_path, output_path):
    """Analiza los audios y exporta el resultado a JSON."""
    
    print(f"🎵 ANÁLISIS Y EXPORTACIÓN A JSON")
    print("=" * 50)
    print(f"📄 Referencia: {reference_path}")
    print(f"🎤 Live: {live_path}")
    print(f"📋 Salida JSON: {output_path}")
    
    try:
        # Crear el analizador
        print(f"\n🔄 Inicializando analizador...")
        analyzer = MusicAnalyzer()
        
        # Realizar análisis completo
        print(f"🔍 Analizando audios...")
        analysis_result = analyzer.comprehensive_analysis(
            reference_path=reference_path,
            live_path=live_path,
            save_name=None,  # No guardar gráficos automáticamente
            verbose=False    # No mostrar detalles por pantalla
        )
        
        # Obtener el resultado DTW de onsets
        dtw_onset_result = analysis_result.get('dtw_onsets')
        
        if dtw_onset_result is None:
            print("❌ Error: No se pudo obtener el resultado DTW de onsets")
            return False
        
        # Mostrar estadísticas del análisis
        print(f"\n📊 Estadísticas del análisis:")
        dtw_onset_result.print_summary()
        
        # Crear directorio de salida si no existe
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Exportar a JSON
        print(f"\n💾 Exportando resultado a JSON...")
        
        # Obtener información adicional para el JSON
        ref_name = Path(reference_path).stem
        live_name = Path(live_path).stem
        
        dtw_onset_result.export_to_json(
            filepath=output_path,
            mutation_category="analisis_real",
            mutation_name=f"{ref_name}_vs_{live_name}",
            reference_path=reference_path,
            live_path=live_path
        )
        
        # Verificar archivo creado
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"✅ Archivo JSON creado exitosamente!")
            print(f"   📁 Ubicación: {output_path}")
            print(f"   📏 Tamaño: {file_size:,} bytes")
            
            # Verificar que se puede cargar correctamente
            print(f"\n🔄 Verificando integridad del archivo JSON...")
            try:
                loaded_result = OnsetDTWAnalysisResult.from_json(output_path)
                print(f"✅ Archivo JSON válido y cargable")
                
                # Comparar estadísticas básicas
                print(f"\n� Verificación de datos:")
                print(f"   Total matches: {dtw_onset_result.total_matches} -> {loaded_result.total_matches}")
                print(f"   Missing onsets: {len(dtw_onset_result.missing_onsets)} -> {len(loaded_result.missing_onsets)}")
                print(f"   Extra onsets: {len(dtw_onset_result.extra_onsets)} -> {len(loaded_result.extra_onsets)}")
                print(f"   Alignment cost: {dtw_onset_result.alignment_cost:.4f} -> {loaded_result.alignment_cost:.4f}")
                
                if (dtw_onset_result.total_matches == loaded_result.total_matches and
                    len(dtw_onset_result.missing_onsets) == len(loaded_result.missing_onsets) and
                    len(dtw_onset_result.extra_onsets) == len(loaded_result.extra_onsets)):
                    print("✅ Datos verificados correctamente")
                else:
                    print("⚠️  Advertencia: Diferencias en los datos cargados")
                    
            except Exception as e:
                print(f"❌ Error al verificar el archivo JSON: {e}")
                return False
            
            return True
        else:
            print("❌ Error: No se pudo crear el archivo JSON")
            return False
            
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_json_preview(json_path, max_lines=20):
    """Muestra una vista previa del archivo JSON generado."""
    try:
        print(f"\n📄 Vista previa del JSON generado:")
        print("-" * 50)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines[:max_lines]):
            print(f"{i+1:3d}: {line.rstrip()}")
        
        if len(lines) > max_lines:
            print(f"     ... ({len(lines) - max_lines} líneas más)")
            
        print("-" * 50)
        print(f"📋 Para ver el contenido completo: cat {json_path}")
        
    except Exception as e:
        print(f"❌ Error al mostrar vista previa: {e}")


def main():
    """Función principal."""
    
    # Validar argumentos
    reference_path, live_path, output_path = validate_arguments()
    
    # Realizar análisis y exportación
    success = analyze_and_export(reference_path, live_path, output_path)
    
    if success:
        # Mostrar vista previa del JSON
        show_json_preview(output_path)
        
        print(f"\n🎉 ¡Exportación completada exitosamente!")
        print(f"📁 Archivo JSON disponible en: {output_path}")
        print(f"\n💡 Tips:")
        print(f"   - Usa 'jq' para formatear JSON: jq '.' {output_path}")
        print(f"   - Abre en VS Code para sintaxis coloreada")
        print(f"   - El archivo contiene todos los detalles del análisis DTW")
    else:
        print(f"\n❌ La exportación falló")
        sys.exit(1)


if __name__ == "__main__":
    main()
