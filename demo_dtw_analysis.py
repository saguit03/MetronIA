#!/usr/bin/env python3
"""
Demostración de la diferencia entre análisis DTW tradicional vs mejorado.

Este script muestra cómo el nuevo análisis DTW puede detectar inconsistencias
entre el alineamiento global (DTW) y los onsets individuales.
"""

import numpy as np
import librosa
from analyzers.music_analyzer import MusicAnalyzer
from analyzers.config import AudioAnalysisConfig


def compare_dtw_methods(reference_path: str, live_path: str):
    """
    Compara los métodos de análisis DTW tradicional vs mejorado.
    
    Args:
        reference_path: Ruta al audio de referencia
        live_path: Ruta al audio en vivo
    """
    print("🔍 COMPARACIÓN: DTW TRADICIONAL vs MEJORADO")
    print("=" * 60)
    
    # Crear analizador
    config = AudioAnalysisConfig()
    analyzer = MusicAnalyzer(config)
    
    try:
        # Cargar audios
        audio_ref, audio_live, sr = analyzer.load_audio_files(reference_path, live_path)
        print(f"✅ Audios cargados:")
        print(f"  📁 Referencia: {reference_path}")
        print(f"  📁 En vivo: {live_path}")
        print(f"  🔊 Sample rate: {sr} Hz")
        
        # Realizar alineamiento DTW
        ref_feat, aligned_live_feat, wp = analyzer.dtw_aligner.align_features(audio_ref, audio_live, sr)
        print(f"\n🔄 Alineamiento DTW completado")
        print(f"  📊 Características extraídas: {ref_feat.shape}")
        print(f"  🗺️ Puntos en camino DTW: {len(wp)}")
        
        # ANÁLISIS TRADICIONAL
        print(f"\n📈 ANÁLISIS DTW TRADICIONAL:")
        print("-" * 40)
        
        traditional_deviations, traditional_regular = analyzer.dtw_aligner.evaluate_dtw_path(wp)
        print(f"  📏 Desviaciones máximas: {np.max(traditional_deviations):.2f}")
        print(f"  📊 Desviaciones promedio: {np.mean(traditional_deviations):.2f}")
        status = "✅ Regular" if traditional_regular else "⚠️ Irregular"
        print(f"  🎯 Evaluación: {status}")
        
        # ANÁLISIS MEJORADO
        print(f"\n🚀 ANÁLISIS DTW MEJORADO:")
        print("-" * 40)
        
        enhanced_analysis = analyzer.dtw_aligner.evaluate_dtw_path_enhanced(wp, audio_ref, audio_live, sr)
        
        print(f"  📈 Evaluación tradicional: {'Regular' if enhanced_analysis['is_regular_traditional'] else 'Irregular'}")
        print(f"  🎯 Evaluación combinada: {'Regular' if enhanced_analysis['is_regular_combined'] else 'Irregular'}")
        
        if 'overall_assessment' in enhanced_analysis:
            print(f"  📝 Evaluación completa: {enhanced_analysis['overall_assessment']}")
            print(f"  🎵 Onsets mapeados: {enhanced_analysis['mapped_onsets']}/{enhanced_analysis['total_ref_onsets']}")
            print(f"  ✅ Onsets bien alineados: {enhanced_analysis['well_aligned_ratio']*100:.1f}%")
            print(f"  ⏱️ Desplazamiento máximo: {enhanced_analysis['max_displacement']*1000:.1f}ms")
            
            if enhanced_analysis['onset_displacements']:
                displacements = np.array(enhanced_analysis['onset_displacements'])
                print(f"  📊 Desplazamiento promedio: {np.mean(displacements)*1000:.1f}ms")
                print(f"  📏 Desviación estándar: {np.std(displacements)*1000:.1f}ms")
        
        # COMPARACIÓN DE RESULTADOS
        print(f"\n🔍 COMPARACIÓN DE RESULTADOS:")
        print("-" * 40)
        
        traditional_says = "Regular" if traditional_regular else "Irregular"
        enhanced_says = "Regular" if enhanced_analysis['is_regular_combined'] else "Irregular"
        
        print(f"  📊 Método tradicional: {traditional_says}")
        print(f"  🚀 Método mejorado: {enhanced_says}")
        
        if traditional_says != enhanced_says:
            print(f"  ⚠️ DISCREPANCIA DETECTADA!")
            print(f"     El método mejorado detectó inconsistencias entre DTW y onsets")
            print(f"     que el método tradicional no capturó.")
        else:
            print(f"  ✅ Ambos métodos coinciden en la evaluación")
        
        # RECOMENDACIONES
        print(f"\n💡 RECOMENDACIONES:")
        print("-" * 40)
        
        if enhanced_analysis.get('well_aligned_ratio', 1.0) < 0.8:
            print("  ⚠️ Bajo porcentaje de onsets bien alineados")
            print("     → Revisar timing de notas individuales")
            print("     → Posibles problemas de interpretación rítmica")
        
        if enhanced_analysis.get('max_displacement', 0) > 0.100:
            print("  ⚠️ Desplazamientos grandes detectados (>100ms)")
            print("     → Revisar notas específicas con timing problemático")
            
        if traditional_regular and not enhanced_analysis.get('is_regular_combined', True):
            print("  🎯 DTW globalmente regular pero onsets inconsistentes")
            print("     → El contenido musical es correcto pero hay errores de timing")
            print("     → Usar análisis de onsets para identificar problemas específicos")
    
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado - {e}")
    except Exception as e:
        print(f"❌ Error en el análisis: {e}")


def demo_with_sample_files():
    """Demostración con archivos de muestra si están disponibles."""
    
    # Lista de archivos de muestra para probar
    sample_pairs = [
        ("audio/01-reference.mp3", "audio/01-live.mp3"),
        ("audio/veneciana-reference.mp3", "audio/veneciana-live.mp3"),
        ("audio/MamboFestival-120.mp3", "audio/MamboFestival-130.mp3"),
    ]
    
    print("🎵 DEMOSTRACIÓN CON ARCHIVOS DE MUESTRA")
    print("=" * 60)
    
    for i, (ref_path, live_path) in enumerate(sample_pairs, 1):
        print(f"\n🎼 PRUEBA {i}:")
        try:
            compare_dtw_methods(ref_path, live_path)
            break  # Solo ejecutar la primera que funcione
        except:
            print(f"⚠️ Archivos no disponibles: {ref_path}, {live_path}")
            continue
    else:
        print("❌ No se encontraron archivos de muestra válidos")
        print("💡 Coloca archivos de audio en el directorio 'audio/' para probar")


if __name__ == "__main__":
    demo_with_sample_files()
    print(f"\n🎉 Demostración completada!")
    print(f"\n📚 RESUMEN:")
    print("El análisis DTW mejorado combina:")
    print("1. ✅ Evaluación tradicional del camino DTW")
    print("2. 🎯 Análisis de consistencia con onsets individuales")
    print("3. 📊 Métricas detalladas de desplazamiento temporal")
    print("4. 💡 Evaluación combinada más precisa")
