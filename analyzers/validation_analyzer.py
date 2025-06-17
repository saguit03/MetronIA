"""
Sistema de validación de análisis de mutaciones usando matrices de confusión.

Este módulo permite evaluar qué tan bien el analizador detecta los cambios introducidos
por las mutaciones, comparando los cambios esperados (changes.csv) con los errores
detectados (analysis.csv).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class ValidationResult:
    """Resultado de validación para una mutación individual."""
    mutation_name: str
    mutation_category: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    detected_changes: int
    expected_changes: int
    total_notes: int


@dataclass
class GlobalValidationResult:
    """Resultado de validación global para todas las mutaciones de un MIDI."""
    midi_name: str
    total_mutations: int
    individual_results: List[ValidationResult]
    overall_precision: float
    overall_recall: float
    overall_f1_score: float
    overall_accuracy: float
    confusion_matrix: np.ndarray
    category_performance: Dict[str, Dict[str, float]]


class MutationValidationAnalyzer:
    """Analizador de validación de mutaciones usando matrices de confusión."""
    
    def __init__(self, results_base_dir: str = "results"):
        """
        Inicializa el analizador de validación.
        
        Args:
            results_base_dir: Directorio base donde están los resultados
        """
        self.results_base_dir = Path(results_base_dir)
        self.tolerance_seconds = 0.1  # Tolerancia para considerar que una nota es la misma
        
    def validate_single_mutation(self, midi_name: str, mutation_category: str, 
                                mutation_name: str) -> Optional[ValidationResult]:
        """
        Valida una mutación individual comparando changes.csv con analysis.csv.
        
        Args:
            midi_name: Nombre del MIDI de referencia
            mutation_category: Categoría de la mutación
            mutation_name: Nombre de la mutación
            
        Returns:
            ValidationResult o None si no se pueden cargar los archivos        """
        try:
            # Rutas de los archivos
            changes_path = self.results_base_dir / f"{midi_name}_Mutaciones" / f"{midi_name}_{mutation_name}" / "changes_detailed.csv"
            analysis_path = self.results_base_dir / f"{midi_name}_Mutaciones" / f"{midi_name}_{mutation_name}" / "analysis.csv"
            
            if not changes_path.exists():
                print(f"⚠️ No se encontró archivo de cambios: {changes_path}")
                return None
                
            if not analysis_path.exists():
                print(f"⚠️ No se encontró archivo de análisis: {analysis_path}")
                return None
            
            # Cargar datos
            changes_df = pd.read_csv(changes_path)
            analysis_df = pd.read_csv(analysis_path)
            
            # Procesar y comparar
            return self._compare_changes_with_analysis(
                changes_df, analysis_df, mutation_name, mutation_category
            )
            
        except Exception as e:
            print(f"❌ Error validando {mutation_category}.{mutation_name}: {e}")
            return None
    
    def _compare_changes_with_analysis(self, changes_df: pd.DataFrame, 
                                     analysis_df: pd.DataFrame,
                                     mutation_name: str, mutation_category: str) -> ValidationResult:
        """
        Compara los cambios esperados con los detectados por el análisis.
        
        Args:
            changes_df: DataFrame con los cambios aplicados por la mutación
            analysis_df: DataFrame con los errores detectados por el análisis
            mutation_name: Nombre de la mutación
            mutation_category: Categoría de la mutación
            
        Returns:
            ValidationResult con las métricas de validación
        """
        # Extraer notas modificadas/añadidas/eliminadas de changes.csv
        expected_changes = self._extract_expected_changes(changes_df)
        
        # Extraer errores detectados de analysis.csv
        detected_errors = self._extract_detected_errors(analysis_df)
        
        # Obtener todas las notas únicas para crear la matriz de confusión
        all_notes = self._get_all_unique_notes(expected_changes, detected_errors)
        
        # Crear vectores de verdad y predicción
        y_true, y_pred = self._create_confusion_vectors(
            all_notes, expected_changes, detected_errors
        )
        
        # Calcular métricas
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        
        return ValidationResult(
            mutation_name=mutation_name,
            mutation_category=mutation_category,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            detected_changes=len(detected_errors),
            expected_changes=len(expected_changes),
            total_notes=len(all_notes)
        )
    
    def _extract_expected_changes(self, changes_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extrae las notas que deberían haberse detectado como errores según changes.csv.
        
        Args:
            changes_df: DataFrame con los cambios de la mutación
            
        Returns:
            Lista de diccionarios con información de notas modificadas
        """
        expected_changes = []
        
        for _, row in changes_df.iterrows():
            change_type = row.get('change_type', '')
            
            if change_type in ['modified', 'added', 'removed']:
                expected_changes.append({
                    'start_time': row.get('start_time', row.get('original_start', 0.0)),
                    'pitch': row.get('pitch', row.get('original_pitch', 60)),
                    'change_type': change_type,
                    'row_data': row.to_dict()
                })
        
        return expected_changes
    
    def _extract_detected_errors(self, analysis_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extrae los errores detectados por el análisis.
        
        Args:
            analysis_df: DataFrame con el análisis de onsets
            
        Returns:
            Lista de diccionarios con errores detectados
        """
        detected_errors = []
        
        for _, row in analysis_df.iterrows():
            onset_type = row.get('onset_type', '')
            
            if onset_type in ['late', 'early', 'extra', 'missing']:
                detected_errors.append({
                    'start_time': row.get('ref_onset_time', row.get('live_onset_time', 0.0)),
                    'pitch': row.get('pitch', 60),  # Si está disponible
                    'error_type': onset_type,
                    'adjustment_ms': row.get('adjustment_ms', 0.0),
                    'row_data': row.to_dict()
                })
        
        return detected_errors
    
    def _get_all_unique_notes(self, expected_changes: List[Dict[str, Any]], 
                            detected_errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Obtiene todas las notas únicas de ambas listas para crear la matriz de confusión.
        
        Args:
            expected_changes: Lista de cambios esperados
            detected_errors: Lista de errores detectados
            
        Returns:
            Lista de notas únicas
        """
        all_notes = []
        seen_notes = set()
        
        # Añadir notas de cambios esperados
        for change in expected_changes:
            note_key = f"{change['start_time']:.3f}_{change['pitch']}"
            if note_key not in seen_notes:
                all_notes.append({
                    'start_time': change['start_time'],
                    'pitch': change['pitch'],
                    'note_key': note_key
                })
                seen_notes.add(note_key)
        
        # Añadir notas de errores detectados
        for error in detected_errors:
            note_key = f"{error['start_time']:.3f}_{error['pitch']}"
            if note_key not in seen_notes:
                all_notes.append({
                    'start_time': error['start_time'],
                    'pitch': error['pitch'],
                    'note_key': note_key
                })
                seen_notes.add(note_key)
        
        return all_notes
    
    def _create_confusion_vectors(self, all_notes: List[Dict[str, Any]], 
                                expected_changes: List[Dict[str, Any]], 
                                detected_errors: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
        """
        Crea vectores de verdad y predicción para la matriz de confusión.
        
        Args:
            all_notes: Lista de todas las notas únicas
            expected_changes: Lista de cambios esperados
            detected_errors: Lista de errores detectados
            
        Returns:
            Tupla con vectores (y_true, y_pred)
        """
        y_true = []
        y_pred = []
        
        for note in all_notes:
            # Verificar si esta nota debería haberse detectado como error
            should_be_detected = self._note_should_be_detected(note, expected_changes)
            
            # Verificar si esta nota fue detectada como error
            was_detected = self._note_was_detected(note, detected_errors)
            
            y_true.append(1 if should_be_detected else 0)
            y_pred.append(1 if was_detected else 0)
        
        return y_true, y_pred
    
    def _note_should_be_detected(self, note: Dict[str, Any], 
                               expected_changes: List[Dict[str, Any]]) -> bool:
        """
        Determina si una nota debería haberse detectado como error.
        
        Args:
            note: Información de la nota
            expected_changes: Lista de cambios esperados
            
        Returns:
            True si la nota debería haberse detectado como error
        """
        for change in expected_changes:
            if (abs(note['start_time'] - change['start_time']) <= self.tolerance_seconds and
                note['pitch'] == change['pitch']):
                return True
        return False
    
    def _note_was_detected(self, note: Dict[str, Any], 
                         detected_errors: List[Dict[str, Any]]) -> bool:
        """
        Determina si una nota fue detectada como error.
        
        Args:
            note: Información de la nota
            detected_errors: Lista de errores detectados
            
        Returns:
            True si la nota fue detectada como error
        """
        for error in detected_errors:
            if (abs(note['start_time'] - error['start_time']) <= self.tolerance_seconds and
                note['pitch'] == error['pitch']):
                return True
        return False
    
    def validate_all_mutations(self, midi_name: str) -> Optional[GlobalValidationResult]:
        """
        Valida todas las mutaciones de un MIDI y genera un resultado global.
        
        Args:
            midi_name: Nombre del MIDI de referencia
            
        Returns:
            GlobalValidationResult con métricas globales
        """        # Buscar todas las mutaciones disponibles en la nueva estructura
        mutations_base_dir = self.results_base_dir / f"{midi_name}_Mutaciones"
        
        if not mutations_base_dir.exists():
            print(f"⚠️ No se encontró directorio de mutaciones: {mutations_base_dir}")
            return None
        
        individual_results = []
        
        # Procesar cada directorio de análisis individual
        for analysis_dir in mutations_base_dir.iterdir():
            if analysis_dir.is_dir() and analysis_dir.name != "validation":
                # Extraer información del nombre del directorio: MIDI_NAME_mutation_name
                dir_name = analysis_dir.name
                
                # Quitar el prefijo del MIDI para obtener el nombre de la mutación
                if dir_name.startswith(f"{midi_name}_"):
                    mutation_name = dir_name[len(f"{midi_name}_"):]
                    
                    # Buscar los archivos de changes y analysis en este directorio
                    changes_file = analysis_dir / "changes_detailed.csv"
                    analysis_file = analysis_dir / "analysis.csv"
                    
                    if changes_file.exists() and analysis_file.exists():
                        # Para la categoría, intentamos extraerla del nombre de la mutación
                        # o usar un valor por defecto si no se puede determinar
                        category = self._extract_category_from_mutation_name(mutation_name)
                        
                        result = self._validate_from_files(
                            str(changes_file), str(analysis_file), 
                            mutation_name, category
                        )
                        if result:
                            individual_results.append(result)
                    else:
                        print(f"⚠️ Archivos faltantes en {analysis_dir.name}: changes_detailed.csv o analysis.csv")
        
        if not individual_results:
            print(f"⚠️ No se encontraron resultados de validación para {midi_name}")
            return None
        
        # Calcular métricas globales
        return self._calculate_global_metrics(midi_name, individual_results)
    
    def _calculate_global_metrics(self, midi_name: str, 
                                individual_results: List[ValidationResult]) -> GlobalValidationResult:
        """
        Calcula métricas globales a partir de los resultados individuales.
        
        Args:
            midi_name: Nombre del MIDI
            individual_results: Lista de resultados individuales
            
        Returns:
            GlobalValidationResult con métricas globales
        """
        # Sumar todas las métricas
        total_tp = sum(r.true_positives for r in individual_results)
        total_fp = sum(r.false_positives for r in individual_results)
        total_tn = sum(r.true_negatives for r in individual_results)
        total_fn = sum(r.false_negatives for r in individual_results)
        
        # Calcular métricas globales
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn) if (total_tp + total_fp + total_tn + total_fn) > 0 else 0.0
        
        # Crear matriz de confusión global
        confusion_mat = np.array([[total_tn, total_fp], [total_fn, total_tp]])
        
        # Calcular performance por categoría
        category_performance = self._calculate_category_performance(individual_results)
        
        return GlobalValidationResult(
            midi_name=midi_name,
            total_mutations=len(individual_results),
            individual_results=individual_results,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1_score=overall_f1_score,
            overall_accuracy=overall_accuracy,
            confusion_matrix=confusion_mat,
            category_performance=category_performance
        )
    
    def _calculate_category_performance(self, individual_results: List[ValidationResult]) -> Dict[str, Dict[str, float]]:
        """
        Calcula el rendimiento por categoría de mutación.
        
        Args:
            individual_results: Lista de resultados individuales
            
        Returns:
            Diccionario con métricas por categoría
        """
        category_stats = {}
        
        for result in individual_results:
            category = result.mutation_category
            if category not in category_stats:
                category_stats[category] = {
                    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'count': 0
                }
            
            stats = category_stats[category]
            stats['tp'] += result.true_positives
            stats['fp'] += result.false_positives
            stats['tn'] += result.true_negatives
            stats['fn'] += result.false_negatives
            stats['count'] += 1
          # Calcular métricas por categoría
        category_performance = {}
        for category, stats in category_stats.items():
            tp, fp, tn, fn = stats['tp'], stats['fp'], stats['tn'], stats['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            
            category_performance[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'mutations_count': stats['count']
            }
        
        return category_performance
    
    def plot_confusion_matrix(self, result: GlobalValidationResult, 
                            save_path: Optional[str] = None) -> None:
        """
        Genera un gráfico de la matriz de confusión.
        
        Args:
            result: Resultado global de validación
            save_path: Ruta donde guardar el gráfico (opcional)
        """
        plt.figure(figsize=(8, 6))
        
        # Crear matriz de confusión con matplotlib
        cm = result.confusion_matrix
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        
        # Añadir valores en las celdas
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), 
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=14, fontweight='bold')
        
        # Configurar etiquetas
        plt.xticks([0, 1], ['No Error', 'Error'])
        plt.yticks([0, 1], ['No Error', 'Error'])
        plt.xlabel('Predicción')
        plt.ylabel('Verdad')
        plt.title(f'Matriz de Confusión - Validación del Analizador\n{result.midi_name}')
        
        # Añadir métricas como texto
        metrics_text = f'Precisión: {result.overall_precision:.3f} | Recall: {result.overall_recall:.3f} | F1: {result.overall_f1_score:.3f}'
        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Matriz de confusión guardada: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_validation_report(self, result: GlobalValidationResult, 
                                 output_path: Optional[str] = None) -> str:
        """
        Genera un reporte detallado de validación.
        
        Args:
            result: Resultado global de validación
            output_path: Ruta donde guardar el reporte (opcional)
            
        Returns:
            Contenido del reporte como string
        """
        report = f"""
REPORTE DE VALIDACIÓN DEL ANALIZADOR
{'=' * 50}
MIDI de Referencia: {result.midi_name}
Fecha de Validación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total de Mutaciones Analizadas: {result.total_mutations}

MÉTRICAS GLOBALES:
{'=' * 30}
Precisión (Precision): {result.overall_precision:.3f}
Recall (Sensibilidad): {result.overall_recall:.3f}
F1-Score: {result.overall_f1_score:.3f}
Exactitud (Accuracy): {result.overall_accuracy:.3f}

MATRIZ DE CONFUSIÓN:
{'=' * 30}
                    Predicción
                No Error    Error
Verdad No Error    {result.confusion_matrix[0,0]:6d}     {result.confusion_matrix[0,1]:6d}
       Error       {result.confusion_matrix[1,0]:6d}     {result.confusion_matrix[1,1]:6d}

Verdaderos Positivos (TP): {result.confusion_matrix[1,1]}
Falsos Positivos (FP): {result.confusion_matrix[0,1]}
Verdaderos Negativos (TN): {result.confusion_matrix[0,0]}
Falsos Negativos (FN): {result.confusion_matrix[1,0]}

RENDIMIENTO POR CATEGORÍA:
{'=' * 30}
"""
        
        for category, metrics in result.category_performance.items():
            report += f"""
{category.upper()}:
  - Mutaciones: {metrics['mutations_count']}
  - Precisión: {metrics['precision']:.3f}
  - Recall: {metrics['recall']:.3f}  
  - F1-Score: {metrics['f1_score']:.3f}
  - Exactitud: {metrics['accuracy']:.3f}
"""
        
        report += f"""

RESULTADOS INDIVIDUALES:
{'=' * 30}
"""
        
        for i, individual in enumerate(result.individual_results, 1):
            report += f"""
{i}. {individual.mutation_category}.{individual.mutation_name}:
   - TP: {individual.true_positives}, FP: {individual.false_positives}
   - TN: {individual.true_negatives}, FN: {individual.false_negatives}
   - Precisión: {individual.precision:.3f}, Recall: {individual.recall:.3f}
   - F1: {individual.f1_score:.3f}, Exactitud: {individual.accuracy:.3f}
   - Cambios esperados: {individual.expected_changes}, Detectados: {individual.detected_changes}
"""
        
        report += f"""

INTERPRETACIÓN:
{'=' * 30}
- Precisión alta indica pocos falsos positivos (el analizador no detecta errores donde no los hay)
- Recall alto indica pocos falsos negativos (el analizador detecta la mayoría de errores reales)
- F1-Score combina precisión y recall en una sola métrica
- Exactitud muestra el porcentaje total de clasificaciones correctas

RECOMENDACIONES:
"""
        
        if result.overall_precision < 0.7:
            report += "- ⚠️ Precisión baja: El analizador puede estar detectando demasiados falsos positivos\n"
        
        if result.overall_recall < 0.7:
            report += "- ⚠️ Recall bajo: El analizador puede estar perdiendo errores reales\n"
        
        if result.overall_f1_score > 0.8:
            report += "- ✅ Excelente rendimiento general del analizador\n"
        elif result.overall_f1_score > 0.6:
            report += "- ✅ Buen rendimiento general del analizador\n"
        else:
            report += "- ⚠️ El analizador necesita mejoras en su precisión o recall\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Reporte de validación guardado: {output_path}")
        
        return report
    
    def save_validation_results_csv(self, result: GlobalValidationResult, 
                                  output_path: str) -> None:
        """
        Guarda los resultados de validación en formato CSV.
        
        Args:
            result: Resultado global de validación
            output_path: Ruta donde guardar el CSV
        """
        data = []
        
        for individual in result.individual_results:
            data.append({
                'midi_name': result.midi_name,
                'mutation_category': individual.mutation_category,
                'mutation_name': individual.mutation_name,
                'true_positives': individual.true_positives,
                'false_positives': individual.false_positives,
                'true_negatives': individual.true_negatives,
                'false_negatives': individual.false_negatives,
                'precision': individual.precision,
                'recall': individual.recall,
                'f1_score': individual.f1_score,
                'accuracy': individual.accuracy,
                'expected_changes': individual.expected_changes,
                'detected_changes': individual.detected_changes,
                'total_notes': individual.total_notes
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Resultados de validación guardados en CSV: {output_path}")
    
    def _extract_category_from_mutation_name(self, mutation_name: str) -> str:
        """
        Extrae la categoría de una mutación basándose en su nombre.
        
        Args:
            mutation_name: Nombre de la mutación
            
        Returns:
            Categoría estimada de la mutación
        """
        # Mapeo de patrones comunes en nombres de mutaciones a categorías
        category_patterns = {
            'ritmo': ['ritmo', 'rhythm', 'beat', 'timing'],
            'velocidad': ['velocidad', 'velocity', 'dynamics', 'vol'],
            'tempo': ['tempo', 'speed', 'bpm'],
            'altura': ['altura', 'pitch', 'note', 'transpose'],
            'ornamentacion': ['ornament', 'grace', 'trill', 'mordent'],
            'articulacion': ['articul', 'staccato', 'legato', 'accent']
        }
        
        mutation_lower = mutation_name.lower()
        
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if pattern in mutation_lower:
                    return category
        
        # Si no se puede determinar, usar "unknown"
        return "unknown"
    
    def _validate_from_files(self, changes_file_path: str, analysis_file_path: str,
                           mutation_name: str, mutation_category: str) -> Optional[ValidationResult]:
        """
        Valida una mutación usando archivos específicos.
        
        Args:
            changes_file_path: Ruta al archivo changes_detailed.csv
            analysis_file_path: Ruta al archivo analysis.csv
            mutation_name: Nombre de la mutación
            mutation_category: Categoría de la mutación
            
        Returns:
            ValidationResult o None si hay error
        """
        try:
            # Cargar datos
            changes_df = pd.read_csv(changes_file_path)
            analysis_df = pd.read_csv(analysis_file_path)
            
            # Validar usando el método existente
            return self._compare_changes_with_analysis(
                changes_df, analysis_df, mutation_name, mutation_category
            )
            
        except Exception as e:
            print(f"❌ Error validando {mutation_category}.{mutation_name}: {e}")
            return None
