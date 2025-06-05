"""
Script de prueba para el nuevo analizador modular de audio.
"""

import MusicAnalyzer, AudioAnalysisConfig, analyze_performance
import sys
from pathlib import Path

def test_analyzer():
    """Prueba el nuevo analizador con archivos de ejemplo."""
    
    # Configuración personalizada
    config = AudioAnalysisConfig(
        hop_length=512,
        n_mfcc=20,
        onset_margin=0.05,
        tempo_threshold=5.0,
        plot_dpi=300
    )
    
    # Archivos de prueba
    audio_dir = Path("audio")
    if audio_dir.exists():
        # Buscar archivos de referencia y en vivo
        reference_files = list(audio_dir.glob("*reference*")) + list(audio_dir.glob("*01-reference*"))
        live_files = list(audio_dir.glob("*live*")) + list(audio_dir.glob("*01-live*"))
        
        if reference_files and live_files:
            ref_path = str(reference_files[0])
            live_path = str(live_files[0])
            
            print(f"🎵 Analizando:")
            print(f"  📁 Referencia: {ref_path}")
            print(f"  📁 En vivo: {live_path}")
            
            # Usar la función de análisis completo
            results = analyze_performance(ref_path, live_path, "test_analysis", config)
            
            print("\n🎯 Análisis completado exitosamente!")
            print(f"✅ Beat spectrum similar: {results['beat_spectrum'].is_similar}")
            print(f"✅ Tempo similar: {results['tempo'].is_similar}")
            print(f"✅ DTW regular: {results['dtw_regular']}")
            
            onset_stats = results['onsets'].stats
            print(f"🎯 Precisión de onsets: {onset_stats['correct']/(onset_stats['total_ref'] or 1)*100:.1f}%")
            
            return True
        else:
            print("⚠️ No se encontraron archivos de referencia y en vivo en la carpeta audio/")
            return False
    else:
        print("⚠️ Carpeta audio/ no encontrada")
        return False

def test_compatibility():
    """Prueba la compatibilidad con la interfaz original."""
    from analyzer import show_beat_spectrum
    
    audio_dir = Path("audio")
    if audio_dir.exists():
        reference_files = list(audio_dir.glob("*reference*")) + list(audio_dir.glob("*01-reference*"))
        live_files = list(audio_dir.glob("*live*")) + list(audio_dir.glob("*01-live*"))
        
        if reference_files and live_files:
            ref_path = str(reference_files[0])
            live_path = str(live_files[0])
            
            print("\n🔄 Probando compatibilidad con interfaz original...")
            show_beat_spectrum(ref_path, live_path, comparacion_1=True, comparacion_2=True, nombre="test_compat")
            print("✅ Compatibilidad verificada!")
            return True
    
    return False

if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS DEL ANALIZADOR MODULAR")
    print("=" * 50)
    
    # Prueba del nuevo analizador
    print("\n📊 Prueba 1: Nuevo analizador modular")
    success1 = test_analyzer()
    
    # Prueba de compatibilidad
    print("\n🔗 Prueba 2: Compatibilidad con interfaz original")
    success2 = test_compatibility()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 TODAS LAS PRUEBAS EXITOSAS!")
        print("✅ El nuevo analizador está listo para uso")
    elif success1:
        print("⚠️ Analizador funciona, pero problemas de compatibilidad")
    else:
        print("❌ Errores en las pruebas - revisar archivos de audio")
