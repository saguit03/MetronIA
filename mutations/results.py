"""
Clases de resultado para mutaciones musicales.
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any
import pandas as pd
import re
from pathlib import Path
import numpy as np


@dataclass
class MutationResult:
    """Representa el resultado de aplicar una mutación."""
    name: str
    description: str
    function: Callable
    excerpt: Optional[pd.DataFrame] = None
    path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    
    def apply(self, original_excerpt: pd.DataFrame, tempo: int = 120, 
              save_changes: bool = False, output_dir: Optional[str] = None) -> bool:
        """
        Aplica la mutación al excerpt original.
        
        Args:
            original_excerpt: DataFrame con el excerpt musical original
            tempo: Tempo en BPM del MIDI original (usado para mutaciones que requieren tempo)
            save_changes: Si guardar automáticamente los cambios en archivos
            output_dir: Directorio donde guardar los archivos de cambios
        
        Returns:
            bool: True si la mutación fue exitosa, False en caso contrario.
        """
        try:
            # Verificar si la función acepta el parámetro tempo
            import inspect
            sig = inspect.signature(self.function)
            
            if 'tempo' in sig.parameters:
                # Función acepta tempo como parámetro
                self.excerpt = self.function(original_excerpt, tempo=tempo)
            else:
                # Función no acepta tempo
                self.excerpt = self.function(original_excerpt)
                
            if self.excerpt is not None:
                self.success = True
                
                # Guardar cambios automáticamente si se solicita
                if save_changes and output_dir:
                    self.analyze_and_save_changes(original_excerpt, output_dir, tempo)
                
                return True
            else:
                self.success = False
                self.error = "Mutation returned None"
                return False
        except Exception as e:
            self.success = False
            self.error = str(e)
            self.excerpt = None
            return False
    
    def set_path(self, path: str):
        """Establece la ruta del archivo de audio generado."""
        self.path = path
    
    def analyze_and_save_changes(self, original_excerpt: pd.DataFrame, output_dir: str, 
                                base_tempo: int = 120) -> None:
        """
        Analiza los cambios realizados por la mutación y los guarda en archivos CSV y TXT.
        
        Args:
            original_excerpt: DataFrame con el excerpt original
            output_dir: Directorio donde guardar los archivos
            base_tempo: Tempo base del MIDI original
        """
        if not self.success or self.excerpt is None:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Analizar cambios
        changes = self._analyze_changes(original_excerpt, self.excerpt)
        mutation_tempo = self.get_mutation_tempo(base_tempo)
        
        # Guardar CSV con cambios detallados
        csv_path = output_path / f"{self.name}_changes.csv"
        self._save_changes_to_csv(changes, csv_path)        
        # Guardar TXT con resumen
        txt_path = output_path / f"{self.name}_summary.txt"
        self._save_summary_to_txt(changes, mutation_tempo, base_tempo, txt_path)
        
    def _analyze_changes(self, original: pd.DataFrame, mutated: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analiza las diferencias entre el excerpt original y el mutado.
        
        Args:
            original: DataFrame original
            mutated: DataFrame mutado
            
        Returns:
            Lista de diccionarios con los cambios detectados
        """
        changes = []
        
        # Crear índices para comparar notas
        original_notes = self._create_note_index(original)
        mutated_notes = self._create_note_index(mutated)
        
        # Detectar notas modificadas, eliminadas y añadidas
        self._detect_modified_notes(original_notes, mutated_notes, changes)
        self._detect_removed_notes(original_notes, mutated_notes, changes)
        self._detect_added_notes(original_notes, mutated_notes, changes)
        
        # Detectar cambios de tempo/timing
        self._detect_tempo_changes(original_notes, mutated_notes, changes)
        
        return changes
    
    def _create_note_index(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Crea un índice de notas con sus propiedades principales.
        
        Args:
            df: DataFrame con las notas
            
        Returns:
            Diccionario indexado por posición con propiedades de cada nota
        """
        notes = {}
        for idx, row in df.iterrows():
            notes[idx] = {
                'start': row.get('start', 0.0),
                'end': row.get('end', 0.0),
                'pitch': row.get('pitch', 60),
                'velocity': row.get('velocity', 80),
                'duration': row.get('duration', row.get('end', 0.0) - row.get('start', 0.0))
            }
        return notes
    
    def _detect_modified_notes(self, original_notes: Dict, mutated_notes: Dict, 
                              changes: List[Dict[str, Any]]) -> None:
        """Detecta notas que han sido modificadas."""
        tolerance = 0.001  # Tolerancia para comparaciones de punto flotante
        
        for idx in original_notes:
            if idx in mutated_notes:
                orig = original_notes[idx]
                mut = mutated_notes[idx]
                
                # Verificar cada propiedad
                modifications = []
                
                if abs(orig['start'] - mut['start']) > tolerance:
                    modifications.append(f"start: {orig['start']:.3f} → {mut['start']:.3f}")
                
                if abs(orig['end'] - mut['end']) > tolerance:
                    modifications.append(f"end: {orig['end']:.3f} → {mut['end']:.3f}")
                
                if orig['pitch'] != mut['pitch']:
                    modifications.append(f"pitch: {orig['pitch']} → {mut['pitch']}")
                
                if orig['velocity'] != mut['velocity']:
                    modifications.append(f"velocity: {orig['velocity']} → {mut['velocity']}")
                
                if abs(orig['duration'] - mut['duration']) > tolerance:
                    modifications.append(f"duration: {orig['duration']:.3f} → {mut['duration']:.3f}")
                
                if modifications:
                    changes.append({
                        'change_type': 'modified',
                        'note_index': idx,
                        'original_pitch': orig['pitch'],
                        'mutated_pitch': mut['pitch'],
                        'original_velocity': orig['velocity'],
                        'mutated_velocity': mut['velocity'],
                        'modifications': '; '.join(modifications)
                    })
    
    def _detect_removed_notes(self, original_notes: Dict, mutated_notes: Dict, 
                             changes: List[Dict[str, Any]]) -> None:
        """Detecta notas que han sido eliminadas."""
        for idx in original_notes:
            if idx not in mutated_notes:
                orig = original_notes[idx]
                changes.append({
                    'change_type': 'removed',
                    'note_index': idx,
                    'original_start': orig['start'],
                    'original_pitch': orig['pitch'],
                    'original_velocity': orig['velocity'],
                    'original_duration': orig['duration'],
                    'modifications': 'Note removed'
                })
    
    def _detect_added_notes(self, original_notes: Dict, mutated_notes: Dict, 
                           changes: List[Dict[str, Any]]) -> None:
        """Detecta notas que han sido añadidas."""
        for idx in mutated_notes:
            if idx not in original_notes:
                mut = mutated_notes[idx]
                changes.append({
                    'change_type': 'added',
                    'note_index': idx,
                    'mutated_start': mut['start'],
                    'mutated_pitch': mut['pitch'],
                    'mutated_velocity': mut['velocity'],
                    'mutated_duration': mut['duration'],
                    'modifications': 'Note added'
                })
    
    def _detect_tempo_changes(self, original_notes: Dict, mutated_notes: Dict, 
                             changes: List[Dict[str, Any]]) -> None:
        """
        Detecta cambios de tempo/timing comparando los patrones temporales.
        
        Args:
            original_notes: Diccionario de notas originales
            mutated_notes: Diccionario de notas mutadas
            changes: Lista donde añadir los cambios detectados
        """
        if not original_notes or not mutated_notes:
            return
            
        # Obtener todos los tiempos de inicio
        orig_starts = [note['start'] for note in original_notes.values()]
        mut_starts = [note['start'] for note in mutated_notes.values()]
        
        if len(orig_starts) != len(mut_starts):
            # Diferente número de notas - puede indicar problemas de timing
            return
            
        # Calcular intervalos entre notas consecutivas
        orig_intervals = []
        mut_intervals = []
        
        for i in range(1, len(orig_starts)):
            orig_intervals.append(orig_starts[i] - orig_starts[i-1])
            mut_intervals.append(mut_starts[i] - mut_starts[i-1])
        
        # Detectar cambios significativos en los intervalos
        timing_changes = 0
        total_shift = 0.0
        
        for i, (orig_interval, mut_interval) in enumerate(zip(orig_intervals, mut_intervals)):
            if abs(orig_interval - mut_interval) > 0.01:  # Tolerancia de 10ms
                timing_changes += 1
                total_shift += abs(orig_interval - mut_interval)
        
        # Si hay cambios significativos, registrarlos
        if timing_changes > len(orig_intervals) * 0.3:  # Si más del 30% de los intervalos cambian
            # Estimar el factor de tempo
            if orig_intervals and mut_intervals:
                avg_orig_interval = sum(orig_intervals) / len(orig_intervals)
                avg_mut_interval = sum(mut_intervals) / len(mut_intervals)
                
                if avg_orig_interval > 0:
                    tempo_factor = avg_orig_interval / avg_mut_interval
                    
                    # Determinar tipo de cambio de tempo
                    if tempo_factor > 1.05:  # Más rápido
                        change_type = "tempo_faster"
                    elif tempo_factor < 0.95:  # Más lento
                        change_type = "tempo_slower"
                    else:
                        change_type = "timing_variation"
                    
                    changes.append({
                        'change_type': change_type,
                        'tempo_factor': tempo_factor,
                        'affected_intervals': timing_changes,
                        'total_intervals': len(orig_intervals),
                        'avg_shift_seconds': total_shift / max(timing_changes, 1),
                        'modifications': f"Tempo change detected: factor {tempo_factor:.3f}, {timing_changes}/{len(orig_intervals)} intervals affected"
                    })
        
        # Detectar desplazamientos constantes (acelerando/ritardando)
        shifts = []
        for i, (orig_start, mut_start) in enumerate(zip(orig_starts, mut_starts)):
            shift = mut_start - orig_start
            shifts.append(shift)
        
        # Si hay un patrón de aceleración o desaceleración
        if len(shifts) > 2:
            # Calcular la tendencia de los desplazamientos
            trend = 0
            for i in range(1, len(shifts)):
                if shifts[i] > shifts[i-1]:                    trend += 1
                elif shifts[i] < shifts[i-1]:
                    trend -= 1
            
            # Si hay una tendencia clara
            if abs(trend) > len(shifts) * 0.5:
                if trend > 0:
                    change_type = "accelerando"
                else:
                    change_type = "ritardando"
                
                changes.append({
                    'change_type': change_type,
                    'trend_strength': abs(trend) / len(shifts),
                    'max_shift': max(shifts),
                    'min_shift': min(shifts),
                    'modifications': f"{change_type.capitalize()} detected: trend strength {abs(trend)/len(shifts):.3f}"
                })

    def _save_changes_to_csv(self, changes: List[Dict[str, Any]], csv_path: Path) -> None:
        """
        Guarda los cambios detallados en un archivo CSV.
        
        Args:
            changes: Lista de cambios detectados
            csv_path: Ruta donde guardar el CSV
        """
        # Si no hay cambios detectados en notas individuales, 
        # intentar inferir el tipo de mutación del nombre
        if not changes:
            mutation_type = self._infer_mutation_type_from_name()
            changes = [{
                'change_type': mutation_type,
                'modifications': f'Global {mutation_type} mutation applied - changes affect timing/tempo rather than individual notes',
                'mutation_name': self.name,
                'mutation_description': self.description
            }]
        
        df = pd.DataFrame(changes)
        df.to_csv(csv_path, index=False, encoding='utf-8')
    
    def _infer_mutation_type_from_name(self) -> str:
        """
        Infiere el tipo de mutación basándose en el nombre y descripción.
        
        Returns:
            Tipo de mutación inferido
        """
        name_lower = self.name.lower()
        desc_lower = self.description.lower()
        
        # Detectar mutaciones de tempo
        if any(keyword in name_lower for keyword in ['tempo', 'faster', 'slower', 'speed']):
            if 'faster' in name_lower or 'fast' in name_lower:
                return 'tempo_faster'
            elif 'slower' in name_lower or 'slow' in name_lower:
                return 'tempo_slower'
            else:
                return 'tempo_change'
        
        # Detectar mutaciones de timing
        elif any(keyword in name_lower for keyword in ['accelerando', 'ritardando', 'fluctuation']):
            if 'accelerando' in name_lower:
                return 'accelerando'
            elif 'ritardando' in name_lower:
                return 'ritardando'
            elif 'fluctuation' in name_lower:
                return 'tempo_fluctuation'
            else:
                return 'timing_change'
        
        # Detectar mutaciones de altura/pitch
        elif any(keyword in name_lower for keyword in ['pitch', 'note', 'altura', 'too_late', 'too_soon']):
            if 'too_late' in name_lower or 'late' in name_lower:
                return 'note_too_late'
            elif 'too_soon' in name_lower or 'early' in name_lower:
                return 'note_too_soon'
            else:
                return 'pitch_change'
        
        # Por defecto
        else:
            return 'unknown'
    
    def _save_summary_to_txt(self, changes: List[Dict[str, Any]], mutation_tempo: int,
                            base_tempo: int, txt_path: Path) -> None:
        """
        Guarda un resumen de las modificaciones y el tempo en un archivo TXT.
        
        Args:
            changes: Lista de cambios detectados
            mutation_tempo: Tempo de la mutación
            base_tempo: Tempo base original
            txt_path: Ruta donde guardar el TXT
        """
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"RESUMEN DE MUTACIÓN: {self.name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Descripción: {self.description}\n")
            f.write(f"Estado: {'Exitosa' if self.success else 'Fallida'}\n")
            if self.error:
                f.write(f"Error: {self.error}\n")
            f.write("\n")
            
            # Información de tempo
            f.write("INFORMACIÓN DE TEMPO:\n")
            f.write(f"Tempo original: {base_tempo} BPM\n")
            f.write(f"Tempo de la mutación: {mutation_tempo} BPM\n")
            if mutation_tempo != base_tempo:
                factor = mutation_tempo / base_tempo
                f.write(f"Factor de cambio: {factor:.2f}x\n")
            f.write("\n")
            
            # Resumen de cambios
            f.write("RESUMEN DE MODIFICACIONES:\n")
            if not changes:
                f.write("No se detectaron cambios en las notas.\n")
            else:
                # Contar tipos de cambios
                modified_count = len([c for c in changes if c['change_type'] == 'modified'])
                removed_count = len([c for c in changes if c['change_type'] == 'removed'])
                added_count = len([c for c in changes if c['change_type'] == 'added'])
                
                f.write(f"Total de cambios: {len(changes)}\n")
                f.write(f"Notas modificadas: {modified_count}\n")
                f.write(f"Notas eliminadas: {removed_count}\n")
                f.write(f"Notas añadidas: {added_count}\n")
                f.write("\n")
                
                # Detalles de cambios más significativos
                f.write("DETALLES DE CAMBIOS PRINCIPALES:\n")
                for i, change in enumerate(changes[:10]):  # Mostrar solo los primeros 10
                    f.write(f"{i+1}. {change['change_type'].upper()}")
                    if change['note_index'] is not None:
                        f.write(f" (Nota #{change['note_index']})")
                    f.write(f": {change['modifications']}\n")
                
                if len(changes) > 10:
                    f.write(f"... y {len(changes) - 10} cambios más (ver CSV para detalles completos)\n")
            
            f.write(f"\nArchivos relacionados:\n")
            f.write(f"CSV detallado: {txt_path.stem}_changes.csv\n")
            if self.path:
                f.write(f"Audio generado: {self.path}\n")
    
    def get_mutation_tempo(self, base_tempo: int = 120) -> int:
        """
        Calcula el tempo correspondiente a la mutación aplicada.
        
        Args:
            base_tempo: Tempo base del MIDI original (BPM)
            
        Returns:
            int: Tempo calculado para la mutación en BPM
        """
        if not self.success or self.excerpt is None:
            return base_tempo
        
        # Detectar tipo de mutación de tempo
        mutation_type = self._detect_tempo_mutation_type()
        
        if mutation_type is None:
            return base_tempo
        
        return self._calculate_tempo_for_mutation(base_tempo, mutation_type)
    
    def _detect_tempo_mutation_type(self) -> Optional[str]:
        """
        Detecta el tipo de mutación de tempo basado en el nombre.
        
        Returns:
            str: Tipo de mutación o None si no es una mutación de tempo
        """
        tempo_patterns = {
            'faster_tempo': 'faster',
            'a_lot_faster_tempo': 'a_lot_faster',
            'slower_tempo': 'slower', 
            'a_lot_slower_tempo': 'a_lot_slower',
            'accelerando': 'accelerando',
            'ritardando': 'ritardando',
            'tempo_fluctuation': 'fluctuation'
        }
        
        for pattern, mutation_type in tempo_patterns.items():
            if pattern in self.name:
                return mutation_type
        
        return None
    
    def _calculate_tempo_for_mutation(self, base_tempo: int, mutation_type: str) -> int:
        """
        Calcula el tempo específico para cada tipo de mutación.
        
        Args:
            base_tempo: Tempo base en BPM
            mutation_type: Tipo de mutación detectada
            
        Returns:
            int: Tempo calculado en BPM
        """
        from mutations.config import (
            FASTER, A_LOT_FASTER, SLOWER, A_LOT_SLOWER, 
            ACCELERANDO, RITARDANDO
        )
        
        tempo_factors = {
            'faster': FASTER,           # 1.2
            'a_lot_faster': A_LOT_FASTER,   # 1.5
            'slower': SLOWER,           # 0.8
            'a_lot_slower': A_LOT_SLOWER,   # 0.5
            'accelerando': ACCELERANDO,     # 1.3 (tempo final)
            'ritardando': RITARDANDO,       # 0.7 (tempo final)
            'fluctuation': 1.0          # Tempo promedio (sin cambio base)
        }
        
        factor = tempo_factors.get(mutation_type, 1.0)
        
        # Para accelerando y ritardando, usar el tempo final
        if mutation_type in ['accelerando', 'ritardando']:
            calculated_tempo = int(base_tempo * factor)
        else:
            # Para cambios constantes de tempo
            calculated_tempo = int(base_tempo * factor)
        
        # Asegurar que el tempo esté en un rango razonable (40-200 BPM)
        return max(40, min(200, calculated_tempo))
    
    def is_tempo_mutation(self) -> bool:
        """
        Verifica si la mutación afecta el tempo.
        
        Returns:
            bool: True si es una mutación de tempo
        """
        return self._detect_tempo_mutation_type() is not None
    
    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"{status} {self.name}: {self.description}"
