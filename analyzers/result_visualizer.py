"""
Visualizador y exportador de resultados del análisis DTW de onsets.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from .onset_results import OnsetDTWAnalysisResult

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
            ax2.grid(True, alpha=0.3)
            fig.legend(loc='lower right')
        
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