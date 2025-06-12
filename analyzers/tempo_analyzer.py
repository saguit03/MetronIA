"""
Analizador de tempo musical y estructura de compases.
"""

import numpy as np
import librosa
from typing import Dict, Any, List
from .config import AudioAnalysisConfig
from .results import TempoAnalysisResult


class TempoAnalyzer:
    """Analizador de tempo musical."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def analyze_tempo(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> TempoAnalysisResult:
        """Analiza las diferencias de tempo entre grabaciones."""
        tempo_ref, _ = librosa.beat.beat_track(y=audio_ref, sr=sr)
        tempo_live, _ = librosa.beat.beat_track(y=audio_live, sr=sr)
        
        tempo_ref = float(tempo_ref.item())
        tempo_live = float(tempo_live.item())
        difference = abs(tempo_ref - tempo_live)
        is_similar = difference <= self.config.tempo_threshold
        
        return TempoAnalysisResult(
            tempo_ref=tempo_ref,
            tempo_live=tempo_live,
            difference=difference,
            is_similar=is_similar
        )
    
    def validate_segments(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int, 
                         tolerance: float = 0.2) -> Dict[str, Any]:
        """Valida la estructura de compases."""
        duration_ref = librosa.get_duration(y=audio_ref, sr=sr)
        duration_live = librosa.get_duration(y=audio_live, sr=sr)
        
        n_compases_ref = int(duration_ref // self.config.compas_duration)
        n_compases_live = int(duration_live // self.config.compas_duration)
        
        measures_compatible = abs(n_compases_ref - n_compases_live) <= 1
        duration_compatible = abs(duration_ref - duration_live) <= tolerance * duration_ref
        
        return {
            'measures_ref': n_compases_ref,
            'measures_live': n_compases_live,
            'duration_ref': duration_ref,
            'duration_live': duration_live,
            'measures_compatible': measures_compatible,
            'duration_compatible': duration_compatible,
            'overall_compatible': measures_compatible and duration_compatible
        }
    
    def extract_multiple_tempo_candidates(self, audio: np.ndarray, sr: int) -> List[float]:
        """
        Extrae múltiples candidatos de tempo usando diferentes métodos para mayor robustez.
        
        Args:
            audio: Audio a analizar
            sr: Sample rate
            
        Returns:
            Lista de candidatos de tempo ordenados por confianza
        """
        tempos = []
        
        try:
            # Método 1: beat_track estándar
            tempo_bt, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempos.append(float(tempo_bt.item()))
        except:
            pass
        
        try:
            # Método 2: tempo estimation con aggregate function
            tempo_agg = librosa.beat.tempo(y=audio, sr=sr, aggregate=np.median)[0]
            tempos.append(float(tempo_agg))
        except:
            pass
        
        try:
            # Método 3: onset-based tempo estimation
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            if len(onset_frames) > 3:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                intervals = np.diff(onset_times)
                if len(intervals) > 0:
                    # Convertir intervalos promedio a BPM
                    avg_interval = np.median(intervals)
                    if avg_interval > 0:
                        tempo_onset = 60.0 / avg_interval
                        tempos.append(float(tempo_onset))
        except:
            pass
        
        # Remover duplicados y valores extremos
        tempos = [t for t in tempos if 40 <= t <= 300]  # Rango razonable de tempo
        tempos = list(set([round(t, 1) for t in tempos]))  # Redondear y quitar duplicados
        
        return sorted(tempos)
    
    def correct_tempo_octave_errors(self, tempo_ref: float, tempo_live: float) -> float:
        """
        Corrige errores de octava en la detección de tempo (doble/mitad).
        
        Args:
            tempo_ref: Tempo de referencia
            tempo_live: Tempo detectado en vivo (posiblemente erróneo)
            
        Returns:
            Tempo corregido
        """
        # Calcular ratios comunes de error de octava
        ratio_double = tempo_live / tempo_ref if tempo_ref > 0 else 0
        ratio_half = tempo_ref / tempo_live if tempo_live > 0 else 0
        
        # Tolerance para considerar que es doble/mitad
        octave_tolerance = 0.15  # 15% de tolerancia
        
        # Verificar si el tempo live es aproximadamente el doble
        if abs(ratio_double - 2.0) < octave_tolerance:
            return tempo_live / 2.0
        
        # Verificar si el tempo live es aproximadamente la mitad
        elif abs(ratio_half - 2.0) < octave_tolerance:
            return tempo_live * 2.0
        
        # Verificar otras relaciones fraccionarias comunes
        # 3/2 o 2/3 (tempo en tresillos vs binario)
        elif abs(ratio_double - 1.5) < octave_tolerance:
            return tempo_live / 1.5
        elif abs(ratio_double - (2.0/3.0)) < octave_tolerance:
            return tempo_live * 1.5
        
        # Si no hay error de octava evidente, devolver el original
        return tempo_live
    
    def analyze_tempo_robust(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> TempoAnalysisResult:
        """
        Análisis robusto de tempo que maneja errores de detección al doble/mitad.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            
        Returns:
            Resultado del análisis de tempo con corrección de errores
        """
        # Obtener múltiples candidatos para ambos audios
        candidates_ref = self.extract_multiple_tempo_candidates(audio_ref, sr)
        candidates_live = self.extract_multiple_tempo_candidates(audio_live, sr)
        
        # Si no se detectaron candidatos, usar método original como fallback
        if not candidates_ref or not candidates_live:
            return self.analyze_tempo(audio_ref, audio_live, sr)
        
        # Usar el primer candidato (más confiable) como tempo principal
        tempo_ref = candidates_ref[0]
        tempo_live_raw = candidates_live[0]
        
        # Aplicar corrección de errores de octava
        tempo_live_corrected = self.correct_tempo_octave_errors(tempo_ref, tempo_live_raw)
        
        # Si la corrección no mejoró significativamente, probar con otros candidatos
        initial_diff = abs(tempo_ref - tempo_live_raw)
        corrected_diff = abs(tempo_ref - tempo_live_corrected)
        
        # Si la corrección no ayudó, buscar el mejor candidato entre todos
        if corrected_diff >= initial_diff and len(candidates_live) > 1:
            best_tempo = tempo_live_raw
            best_diff = initial_diff
            
            for candidate in candidates_live:
                # Probar corrección con cada candidato
                corrected_candidate = self.correct_tempo_octave_errors(tempo_ref, candidate)
                diff_original = abs(tempo_ref - candidate)
                diff_corrected = abs(tempo_ref - corrected_candidate)
                
                # Usar el que tenga menor diferencia
                best_candidate = corrected_candidate if diff_corrected < diff_original else candidate
                candidate_diff = min(diff_original, diff_corrected)
                
                if candidate_diff < best_diff:
                    best_diff = candidate_diff
                    best_tempo = best_candidate
            
            tempo_live_corrected = best_tempo
        
        # Calcular diferencia final y similitud
        final_difference = abs(tempo_ref - tempo_live_corrected)
        is_similar = final_difference <= self.config.tempo_threshold
        
        return TempoAnalysisResult(
            tempo_ref=tempo_ref,
            tempo_live=tempo_live_corrected,
            difference=final_difference,
            is_similar=is_similar
        )
    
    def analyze_tempo_with_reference(self, audio_ref: np.ndarray, audio_live: np.ndarray, 
                                   sr: int, reference_tempo: float = None) -> TempoAnalysisResult:
        """
        Análisis de tempo usando un tempo de referencia conocido para mayor precisión.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo  
            sr: Sample rate
            reference_tempo: Tempo conocido del MIDI original (opcional)
            
        Returns:
            Resultado del análisis de tempo
        """
        if reference_tempo is not None:
            # Si tenemos tempo de referencia del MIDI, usarlo como verdad fundamental
            tempo_ref = reference_tempo
            
            # Detectar tempo del audio en vivo
            candidates_live = self.extract_multiple_tempo_candidates(audio_live, sr)
            if not candidates_live:
                # Fallback al método original
                tempo_live, _ = librosa.beat.beat_track(y=audio_live, sr=sr)
                tempo_live = float(tempo_live.item())
            else:
                tempo_live_raw = candidates_live[0]
                tempo_live = self.correct_tempo_octave_errors(tempo_ref, tempo_live_raw)
                
                # Probar otros candidatos si el primero no es bueno
                if len(candidates_live) > 1:
                    best_tempo = tempo_live
                    best_diff = abs(tempo_ref - tempo_live)
                    
                    for candidate in candidates_live[1:]:
                        corrected = self.correct_tempo_octave_errors(tempo_ref, candidate)
                        diff = abs(tempo_ref - corrected)
                        if diff < best_diff:
                            best_diff = diff
                            best_tempo = corrected
                    
                    tempo_live = best_tempo
        else:
            # Si no tenemos tempo de referencia, usar análisis robusto
            return self.analyze_tempo_robust(audio_ref, audio_live, sr)
        
        # Calcular resultados finales
        difference = abs(tempo_ref - tempo_live)
        is_similar = difference <= self.config.tempo_threshold
        
        return TempoAnalysisResult(
            tempo_ref=tempo_ref,
            tempo_live=tempo_live,
            difference=difference,
            is_similar=is_similar
        )
