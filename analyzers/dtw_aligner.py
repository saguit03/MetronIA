"""
Alineador usando Dynamic Time Warping para análisis musical.
"""

import numpy as np
import librosa
from typing import Tuple
from .config import AudioAnalysisConfig
from .feature_extractor import AudioFeatureExtractor

import soundfile as sf
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from pydub import AudioSegment
from pyrubberband import pyrb

class DTWAligner:
    """Alineador usando Dynamic Time Warping."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        self.feature_extractor = AudioFeatureExtractor(config)
    
    def synchronize_audios(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sincroniza dos audios usando DTW y devuelve el audio en vivo alineado.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            
        Returns:
            Tuple con el audio de referencia y el audio en vivo alineado
        """
        wp, wp_s, chroma_ref, chroma_live = self.align_chroma_features(audio_ref, audio_live, sr)
        aligned_audio_live = self.sync(audio_live, wp_s, sr, len(audio_ref))
        return aligned_audio_live, wp, wp_s

    def align_chroma_features(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        chroma_ref = self.feature_extractor.extract_chroma_features(audio_ref, sr)
        chroma_live = self.feature_extractor.extract_chroma_features(audio_live, sr)
        D, wp = librosa.sequence.dtw(X=chroma_ref.T, Y=chroma_live.T, metric='cosine')
        wp_s = librosa.frames_to_time(wp, sr=sr, hop_length=self.config.hop_length)
        return wp, wp_s, chroma_ref, chroma_live
    
    def sync(self, audio, wp_s, sample_rate, out_len, n_arrows = 10):
        time_map = [(int(x*sample_rate), int(y*sample_rate)) for (x, y) in wp_s[::len(wp_s)//n_arrows]]
        time_map.append((len(audio), out_len))
        return pyrb.timemap_stretch(audio, sample_rate, time_map)

    def calculate_dtw(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int, inversed = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Extraer características
        reference_mfcc_features = self.feature_extractor.extract_mfcc_features(audio_ref, sr)
        live_mfcc_features = self.feature_extractor.extract_mfcc_features(audio_live, sr)
        
        D, wp = librosa.sequence.dtw(X=reference_mfcc_features.T, Y=live_mfcc_features.T, metric='cosine')

        if inversed:
            wp = np.array(wp[::-1])  # Asegurar orden de principio a fin
        
        return wp, reference_mfcc_features, live_mfcc_features
    
    def analyze_dtw_timing_consistency(self, wp: np.ndarray, audio_ref: np.ndarray, 
                                      audio_live: np.ndarray, sr: int) -> dict:
        """
        Analiza la consistencia temporal del camino DTW considerando onsets.
        
        Args:
            wp: Camino DTW (warping path)
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            
        Returns:
            Diccionario con métricas de consistencia temporal
        """
        # Detectar onsets en ambos audios
        onsets_ref = librosa.onset.onset_detect(y=audio_ref, sr=sr, units='time')
        onsets_live = librosa.onset.onset_detect(y=audio_live, sr=sr, units='time')
        
        # Convertir onsets a frames para mapear con DTW
        ref_onset_frames = librosa.time_to_frames(onsets_ref, sr=sr, hop_length=self.config.hop_length)
        live_onset_frames = librosa.time_to_frames(onsets_live, sr=sr, hop_length=self.config.hop_length)
        
        # Crear mapeo DTW frame a frame
        dtw_mapping = {}
        for ref_frame, live_frame in wp:
            dtw_mapping[ref_frame] = live_frame
        
        # Analizar desplazamientos de onsets según DTW
        onset_displacements = []
        mapped_onsets = 0
        
        for ref_onset_frame in ref_onset_frames:
            if ref_onset_frame in dtw_mapping:
                mapped_live_frame = dtw_mapping[ref_onset_frame]
                
                # Buscar el onset en vivo más cercano al frame mapeado
                if len(live_onset_frames) > 0:
                    closest_live_onset_idx = np.argmin(np.abs(live_onset_frames - mapped_live_frame))
                    closest_live_onset_frame = live_onset_frames[closest_live_onset_idx]
                    
                    # Calcular desplazamiento en tiempo
                    ref_time = librosa.frames_to_time(ref_onset_frame, sr=sr, hop_length=self.config.hop_length)
                    expected_live_time = librosa.frames_to_time(mapped_live_frame, sr=sr, hop_length=self.config.hop_length)
                    actual_live_time = librosa.frames_to_time(closest_live_onset_frame, sr=sr, hop_length=self.config.hop_length)
                    
                    displacement = actual_live_time - expected_live_time
                    onset_displacements.append(displacement)
                    mapped_onsets += 1
        
        # Calcular métricas
        if onset_displacements:
            onset_displacements = np.array(onset_displacements)
            mean_displacement = np.mean(onset_displacements)
            std_displacement = np.std(onset_displacements)
            max_displacement = np.max(np.abs(onset_displacements))
            
            # Clasificar como regular si la mayoría de onsets están bien alineados
            displacement_threshold = 0.050  # 50ms
            well_aligned_ratio = np.sum(np.abs(onset_displacements) <= displacement_threshold) / len(onset_displacements)
            is_onset_consistent = well_aligned_ratio >= 0.8  # 80% de onsets bien alineados
        else:
            mean_displacement = 0
            std_displacement = 0
            max_displacement = 0
            well_aligned_ratio = 1.0
            is_onset_consistent = True
        
        return {
            'onset_displacements': onset_displacements.tolist() if onset_displacements.size > 0 else [],
            'mean_displacement': mean_displacement,
            'std_displacement': std_displacement,
            'max_displacement': max_displacement,
            'mapped_onsets': mapped_onsets,
            'total_ref_onsets': len(onsets_ref),
            'total_live_onsets': len(onsets_live),
            'well_aligned_ratio': well_aligned_ratio,
            'is_onset_consistent': is_onset_consistent,
            'overall_assessment': self._assess_dtw_onset_consistency(well_aligned_ratio, max_displacement)
        }
    
    def _assess_dtw_onset_consistency(self, well_aligned_ratio: float, max_displacement: float) -> str:
        """
        Evalúa la consistencia entre DTW y onsets.
        
        Args:
            well_aligned_ratio: Proporción de onsets bien alineados
            max_displacement: Máximo desplazamiento de onset
            
        Returns:
            Evaluación textual de la consistencia
        """
        if well_aligned_ratio >= 0.95 and max_displacement <= 0.030:
            return "Excelente: DTW y onsets completamente consistentes"
        elif well_aligned_ratio >= 0.8 and max_displacement <= 0.050:
            return "Bueno: DTW y onsets mayormente consistentes"
        elif well_aligned_ratio >= 0.6 and max_displacement <= 0.100:
            return "Aceptable: DTW correcto pero algunos onsets desplazados"
        elif well_aligned_ratio >= 0.4:
            return "Problemático: DTW aparentemente regular pero muchos onsets desplazados"
        else:            return "Crítico: DTW y onsets severamente inconsistentes"
    
    def evaluate_dtw_path_enhanced(self, wp: np.ndarray, audio_ref: Optional[np.ndarray] = None, 
                                  audio_live: Optional[np.ndarray] = None, sr: Optional[int] = None) -> dict:
        """
        Evaluación mejorada del camino DTW que incluye análisis de onsets.
        
        Args:
            wp: Camino DTW
            audio_ref: Audio de referencia (opcional, para análisis de onsets)
            audio_live: Audio en vivo (opcional, para análisis de onsets)
            sr: Sample rate (opcional, para análisis de onsets)
            
        Returns:
            Diccionario con evaluación completa del DTW
        """
        # Evaluación tradicional del DTW
        wp = np.array(wp)
        ref_idxs, live_idxs = wp[:, 0], wp[:, 1]
        deltas = live_idxs - ref_idxs
        deviations = np.abs(deltas - np.mean(deltas))
        
        is_regular_traditional = np.max(deviations) <= self.config.dtw_tolerance * len(ref_idxs)
        
        result = {
            'traditional_deviations': deviations.tolist(),
            'is_regular_traditional': is_regular_traditional,
            'max_deviation_traditional': float(np.max(deviations)),
            'mean_deviation_traditional': float(np.mean(deviations))
        }
        
        # Análisis de consistencia con onsets si se proporcionan los audios
        if audio_ref is not None and audio_live is not None and sr is not None:
            onset_analysis = self.analyze_dtw_timing_consistency(wp, audio_ref, audio_live, sr)
            result.update(onset_analysis)
            
            # Evaluación combinada
            result['is_regular_combined'] = (
                is_regular_traditional and onset_analysis['is_onset_consistent']
            )
        else:
            result['is_regular_combined'] = is_regular_traditional
        
        return result
    
    def align_features_for_tempo_comparison(self, wp, reference_mfcc_features, live_mfcc_features) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        aligned_live_mfcc_features_for_beat = np.zeros_like(reference_mfcc_features)
        for ref_idx, live_idx in wp:
            if ref_idx < len(aligned_live_mfcc_features_for_beat) and live_idx < len(live_mfcc_features):
                aligned_live_mfcc_features_for_beat[ref_idx] = live_mfcc_features[live_idx]
        
        return aligned_live_mfcc_features_for_beat