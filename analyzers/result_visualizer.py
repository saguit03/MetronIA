"""
Visualizador y exportador de resultados del análisis DTW de onsets.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from .dtw_results import OnsetDTWAnalysisResult

class ResultVisualizer:
    """
    Clase para visualizar y exportar resultados del análisis DTW de onsets.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Inicializa el visualizador.
        
        Args:
            output_dir: Directorio base para guardar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Configurar estilo de matplotlib
        plt.style.use('default')
    
    def plot_onset_errors_detailed(self, dtw_onset_result: OnsetDTWAnalysisResult, 
                                  save_name: str, 
                                  dir_path: Optional[str] = None,
                                  show_plot: bool = False) -> str:
        """
        Crea un gráfico detallado de los errores de onset DTW.
        
        Args:
            dtw_onset_result: Resultado del análisis DTW
            save_name: Nombre base para guardar el archivo
            dir_path: Directorio donde guardar el archivo (opcional)
            show_plot: Si mostrar el gráfico en pantalla
            
        Returns:
            Ruta del archivo guardado
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Análisis Detallado DTW de Onsets - {save_name}', fontsize=16, fontweight='bold')
        
        # 1. Distribución de tipos de onsets
        ax1 = axes[0]
        categories = ['Correctos', 'Tarde', 'Adelantados', 'Faltantes', 'Extras']
        counts = [
            len(dtw_onset_result.correct_matches),
            len(dtw_onset_result.late_matches),
            len(dtw_onset_result.early_matches),
            len(dtw_onset_result.missing_onsets),
            len(dtw_onset_result.extra_onsets)
        ]
        colors = ['green', 'orange', 'red', 'gray', 'purple']
        
        bars = ax1.bar(categories, counts, color=colors, alpha=0.7)
        ax1.set_title('Distribución de Tipos de Onsets')
        ax1.set_ylabel('Cantidad')
        
        # Añadir valores encima de las barras
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
          # 2. Timeline de onsets
        ax2 = axes[1]
        
        # Extraer todos los matches para el timeline
        all_matches = (dtw_onset_result.correct_matches + 
                      dtw_onset_result.late_matches + 
                      dtw_onset_result.early_matches)
        
        if all_matches:
            ref_times = [m.ref_onset for m in all_matches]
            live_times = [m.live_onset for m in all_matches]
            
            # Colores según clasificación
            colors_timeline = []
            for match in all_matches:
                if match in dtw_onset_result.correct_matches:
                    colors_timeline.append('green')
                elif match in dtw_onset_result.late_matches:
                    colors_timeline.append('orange')
                else:  # early
                    colors_timeline.append('red')
            
            ax2.scatter(ref_times, live_times, c=colors_timeline, alpha=0.7, s=50)
              # Línea de referencia perfecta
            min_time = min(min(ref_times), min(live_times))
            max_time = max(max(ref_times), max(live_times))
            ax2.plot([min_time, max_time], [min_time, max_time], 'k--', alpha=0.5, label='Timing Perfecto')
            
            ax2.set_xlabel('Tiempo Referencia (s)')
            ax2.set_ylabel('Tiempo Live (s)')
            ax2.set_title('Timeline de Onsets Emparejados')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar en el directorio especificado o usar el por defecto
        if dir_path:
            output_dir = Path(dir_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{save_name}.png"
        else:
            output_path = self.output_dir / f"{save_name}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return str(output_path)
    
    def export_onset_analysis_to_csv(self, analysis_results: Dict[str, Any], 
                                   save_name: str,
                                   mutation_category: str = "",
                                   mutation_name: str = "") -> str:
        """
        Exporta el análisis completo a CSV.
        
        Args:
            analysis_results: Resultados del análisis completo de MusicAnalyzer
            save_name: Nombre base para el archivo
            mutation_category: Categoría de mutación (opcional)
            mutation_name: Nombre de mutación (opcional)
            
        Returns:
            Ruta del archivo CSV guardado
        """
        dtw_onset_result = analysis_results.get('dtw_onsets')
        beat_result = analysis_results.get('beat_spectrum')
        tempo_result = analysis_results.get('tempo')
        segment_result = analysis_results.get('segments')
        dtw_analysis = analysis_results.get('dtw_analysis', {})
        
        if not dtw_onset_result:
            raise ValueError("No se encontraron resultados DTW de onsets en analysis_results")
        
        # Obtener datos CSV del análisis DTW
        csv_data = dtw_onset_result.get_csv_data(mutation_category, mutation_name)
        
        # Agregar datos adicionales del análisis completo
        if beat_result:
            csv_data.update({
                'beat_spectrum_similar': 'Similar' if beat_result.is_similar else 'Diferencias significativas',
                'beat_spectrum_max_difference': f"{beat_result.max_difference:.3f}",
            })
        
        if tempo_result:
            csv_data.update({
                'tempo_reference_bpm': f"{tempo_result.tempo_ref:.2f}",
                'tempo_live_bpm': f"{tempo_result.tempo_live:.2f}",
                'tempo_difference_bpm': f"{tempo_result.difference:.2f}",
                'tempo_similar': 'Tempo similar' if tempo_result.is_similar else 'Diferencia significativa de tempo',
            })
        
        if segment_result:
            csv_data.update({
                'structure_measures_ref': segment_result.get('measures_ref', 0),
                'structure_measures_live': segment_result.get('measures_live', 0),
                'structure_compatible': 'Estructura compatible' if segment_result.get('overall_compatible', False) else 'Estructura incompatible',
            })
        
        if dtw_analysis:
            dtw_regular = dtw_analysis.get('is_regular_combined', dtw_analysis.get('is_regular_traditional', False))
            dtw_assessment = dtw_analysis.get('overall_assessment', 'Análisis DTW estándar')
            csv_data.update({
                'dtw_regular': dtw_assessment,
                'dtw_traditional_regular': dtw_analysis.get('is_regular_traditional', False),
                'dtw_combined_regular': dtw_regular,
            })
        
        # Agregar metadatos
        csv_data.update({
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_name': save_name,
            'sample_rate': analysis_results.get('sample_rate', 'N/A'),
        })
        
        # Crear DataFrame y guardar
        df = pd.DataFrame([csv_data])
        output_path = self.output_dir / f"{save_name}_analysis.csv"
        df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def export_detailed_matches_to_csv(self, dtw_onset_result: OnsetDTWAnalysisResult,
                                     save_name: str) -> str:
        """
        Exporta información detallada de todos los matches a CSV.
        
        Args:
            dtw_onset_result: Resultado del análisis DTW
            save_name: Nombre base para el archivo
            
        Returns:
            Ruta del archivo CSV guardado
        """
        all_data = []
        
        # Procesar matches correctos
        for match in dtw_onset_result.correct_matches:
            all_data.append({
                'classification': 'correct',
                'ref_onset': match.ref_onset,
                'live_onset': match.live_onset,
                'ref_pitch': match.ref_pitch,
                'live_pitch': match.live_pitch,
                'time_adjustment_ms': match.time_adjustment,
                'pitch_similarity': match.pitch_similarity,
                'timing_error_ms': abs(match.time_adjustment),
            })
        
        # Procesar matches tarde
        for match in dtw_onset_result.late_matches:
            all_data.append({
                'classification': 'late',
                'ref_onset': match.ref_onset,
                'live_onset': match.live_onset,
                'ref_pitch': match.ref_pitch,
                'live_pitch': match.live_pitch,
                'time_adjustment_ms': match.time_adjustment,
                'pitch_similarity': match.pitch_similarity,
                'timing_error_ms': abs(match.time_adjustment),
            })
        
        # Procesar matches adelantados
        for match in dtw_onset_result.early_matches:
            all_data.append({
                'classification': 'early',
                'ref_onset': match.ref_onset,
                'live_onset': match.live_onset,
                'ref_pitch': match.ref_pitch,
                'live_pitch': match.live_pitch,
                'time_adjustment_ms': match.time_adjustment,
                'pitch_similarity': match.pitch_similarity,
                'timing_error_ms': abs(match.time_adjustment),
            })
        
        # Procesar onsets faltantes
        for onset_time, pitch in dtw_onset_result.missing_onsets:
            all_data.append({
                'classification': 'missing',
                'ref_onset': onset_time,
                'live_onset': np.nan,
                'ref_pitch': pitch,
                'live_pitch': np.nan,
                'time_adjustment_ms': np.nan,
                'pitch_similarity': np.nan,
                'timing_error_ms': np.nan,
            })
        
        # Procesar onsets extras
        for onset_time, pitch in dtw_onset_result.extra_onsets:
            all_data.append({
                'classification': 'extra',
                'ref_onset': np.nan,
                'live_onset': onset_time,
                'ref_pitch': np.nan,
                'live_pitch': pitch,
                'time_adjustment_ms': np.nan,
                'pitch_similarity': np.nan,
                'timing_error_ms': np.nan,
            })
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(all_data)
        if not df.empty:
            # Ordenar por tiempo de referencia, luego por tiempo live
            df = df.sort_values(['ref_onset', 'live_onset'], na_position='last')
        
        output_path = self.output_dir / f"{save_name}_detailed_matches.csv"
        df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def create_summary_report(self, analysis_results: Dict[str, Any], 
                            save_name: str) -> str:
        """
        Crea un reporte resumen en texto.
        
        Args:
            analysis_results: Resultados del análisis completo
            save_name: Nombre base para el archivo
            
        Returns:
            Ruta del archivo de reporte guardado
        """
        dtw_onset_result = analysis_results.get('dtw_onsets')
        
        if not dtw_onset_result:
            raise ValueError("No se encontraron resultados DTW de onsets")
        
        stats = dtw_onset_result.get_summary_stats()
        
        report = f"""
REPORTE DE ANÁLISIS DTW DE ONSETS
{'='*50}
Archivo: {save_name}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESUMEN GENERAL:
- Total onsets referencia: {stats['total_ref_onsets']}
- Total onsets live: {stats['total_live_onsets']}
- Total emparejamientos: {stats['total_matches']}

CLASIFICACIÓN DE ONSETS:
- Correctos (ritmo consistente): {len(dtw_onset_result.correct_matches)} ({stats['consistency_rate']*100:.1f}%)
- Tarde (desajuste positivo): {len(dtw_onset_result.late_matches)} ({stats['late_rate']*100:.1f}%)
- Adelantados (desajuste negativo): {len(dtw_onset_result.early_matches)} ({stats['early_rate']*100:.1f}%)
- Faltantes: {len(dtw_onset_result.missing_onsets)} ({stats['missing_rate']*100:.1f}%)
- Extras: {len(dtw_onset_result.extra_onsets)} ({stats['extra_rate']*100:.1f}%)

ESTADÍSTICAS DE TIMING:
- Ajuste temporal promedio: {stats['mean_adjustment_ms']:.2f}ms
- Desviación estándar: {stats['std_adjustment_ms']:.2f}ms
- Ajuste máximo: {stats['max_adjustment_ms']:.2f}ms
- Ajuste mínimo: {stats['min_adjustment_ms']:.2f}ms

CALIDAD DEL ANÁLISIS:
- Costo de alineamiento DTW: {stats['alignment_cost']:.3f}
- Tolerancia utilizada: {stats['tolerance_ms']:.1f}ms

INTERPRETACIÓN:
"""
        
        # Agregar interpretación automática
        if stats['consistency_rate'] > 0.8:
            report += "- Excelente consistencia rítmica\n"
        elif stats['consistency_rate'] > 0.6:
            report += "- Buena consistencia rítmica con algunas variaciones\n"
        else:
            report += "- Consistencia rítmica mejorable\n"
        
        if stats['late_rate'] > stats['early_rate']:
            report += "- Tendencia a tocar tarde\n"
        elif stats['early_rate'] > stats['late_rate']:
            report += "- Tendencia a tocar adelantado\n"
        else:
            report += "- Errores de timing balanceados\n"
        
        if stats['missing_rate'] > 0.1:
            report += "- Atención: alto porcentaje de notas faltantes\n"
        
        if stats['extra_rate'] > 0.1:
            report += "- Atención: alto porcentaje de notas extras\n"
        
        # Guardar reporte
        output_path = self.output_dir / f"{save_name}_report.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return str(output_path)
