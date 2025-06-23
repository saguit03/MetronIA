from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from analyzers.config import VERBOSE_LOGGING
from utils.audio_utils import get_reference_audio_duration


class MutationValidation:
    def __init__(self, mutations_summary_path: str, mutation_dir: str):
        self.mutations_summary = pd.read_csv(mutations_summary_path)
        self.mutation_dir = Path(mutation_dir)
        self.results = []

    def run_validation(self):
        for _, row in self.mutations_summary.iterrows():
            mutation_name = row['mutation_name']
            mutation_category = row['mutation_category']

            csv_log_path = self.mutation_dir / "logs" / f"{mutation_name}.csv"

            if not csv_log_path.exists():
                print(f"Warning: Log file not found for mutation {mutation_name}")
                continue

            ground_truth = pd.read_csv(csv_log_path)

            analysis_results = {
                'correct': row['correct_matches'],
                'late': row['late_matches'],
                'early': row['early_matches'],
                'missing': row['missing_onsets'],
                'extra': row['extra_onsets']
            }

            self.validate_mutation(mutation_name, mutation_category, ground_truth, analysis_results)

        return self.get_overall_metrics()

    def validate_mutation(self, mutation_name, mutation_category, ground_truth, analysis_results):
        # Get the analysis CSV file for this mutation
        midi_name = self.mutation_dir.name.replace('_Mutaciones', '')
        analysis_csv_path = self.mutation_dir / f"{midi_name}_{mutation_name}" / "analysis.csv"

        if not analysis_csv_path.exists():
            print(f"Warning: Analysis CSV not found for mutation {mutation_name}")
            return

        analysis_data = pd.read_csv(analysis_csv_path)

        # Extract onset types from ground truth (expected) and analysis (predicted)
        y_true = self.map_onset_types(ground_truth['onset_type'].tolist())
        y_pred = self.map_onset_types(analysis_data['onset_type'].tolist())

        # Handle length mismatch by truncating to the shorter length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

        self.results.append({
            'mutation_name': mutation_name,
            'category': mutation_category,
            'y_true': y_true,
            'y_pred': y_pred
        })

    def map_onset_types(self, onset_types):
        """Map onset types to standardized labels for comparison"""
        mapping = {
            'no_change': 'correct',
            'correct': 'correct',
            'late': 'late',
            'early': 'early',
            'missing': 'missing',
            'extra': 'extra'
        }
        return [mapping.get(onset_type, 'correct') for onset_type in onset_types]

    def get_overall_metrics(self):
        all_true = []
        all_pred = []
        for res in self.results:
            all_true.extend(res['y_true'])
            all_pred.extend(res['y_pred'])

        if not all_true and not all_pred:
            print("Warning: No validation results to generate metrics.")
            return {
                "confusion_matrix": np.array([]),
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "labels": []
            }

        labels = sorted(list(set(all_true + all_pred)))

        cm = confusion_matrix(all_true, all_pred, labels=labels)
        accuracy = accuracy_score(all_true, all_pred)
        precision = precision_score(all_true, all_pred, average='weighted', zero_division=0)
        recall = recall_score(all_true, all_pred, average='weighted', zero_division=0)
        f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)

        return {
            "confusion_matrix": cm,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "labels": labels
        }

    def plot_confusion_matrix(self, metrics, output_path):
        if metrics['confusion_matrix'].size == 0:
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, "No data to generate confusion matrix.",
                     ha='center', va='center', fontsize=12)
            plt.title('Confusion Matrix')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(output_path)
            plt.close()
            return

        # Calculate total counts for each true label
        all_true = []
        all_pred = []
        for res in self.results:
            all_true.extend(res['y_true'])
            all_pred.extend(res['y_pred'])

        # Filter labels to only include those that actually appear in the data
        present_labels = sorted(list(set(all_true + all_pred)))

        # Count occurrences of each label that actually appears
        label_counts = {}
        for label in present_labels:
            label_counts[label] = all_true.count(label)

        # Filter out labels with zero occurrences in true labels
        filtered_labels = [label for label in present_labels if label_counts[label] > 0]

        if not filtered_labels:
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, "No valid data to generate confusion matrix.",
                     ha='center', va='center', fontsize=12)
            plt.title('Confusion Matrix')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(output_path)
            plt.close()
            return

        # Recalculate confusion matrix with only present labels
        from sklearn.metrics import confusion_matrix
        filtered_cm = confusion_matrix(all_true, all_pred, labels=filtered_labels)

        # Create y-axis labels with counts (only for present labels)
        y_labels_with_counts = [f"{label} (n={label_counts[label]})" for label in filtered_labels]

        plt.figure(figsize=(12, 8))
        sns.heatmap(filtered_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=filtered_labels, yticklabels=y_labels_with_counts)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def generate_validation_report(self, metrics, output_path):
        report = f"""
Validation Report
=================
Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1-Score: {metrics['f1_score']:.4f}
"""
        with open(output_path, 'w') as f:
            f.write(report)

    def save_validation_results_csv(self, output_path):
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)


def run_validation_analysis(midi_name: str, results_dir: Path) -> Dict[str, float]:
    """
    Ejecuta el análisis de validación para todas las mutaciones de un MIDI.
    
    Args:
        midi_name: Nombre del archivo MIDI de referencia
        results_dir: Directorio de resultados
        
    Returns:
        Dict con las métricas de validación
    """
    if VERBOSE_LOGGING:
        print("\n" + "=" * 60)
        print("🔍 ANÁLISIS DE VALIDACIÓN DEL ANALIZADOR")
        print("=" * 60)

    mutations_summary_path = results_dir / f"{midi_name}_Mutaciones" / "mutations_summary.csv"
    mutations_dir = results_dir / f"{midi_name}_Mutaciones"

    if not mutations_summary_path.exists():
        print(f"⚠️ No se encontró el archivo de resumen de mutaciones: {mutations_summary_path}")
        return {}

    validator = MutationValidation(
        mutations_summary_path=str(mutations_summary_path),
        mutation_dir=str(mutations_dir)
    )

    validation_result = validator.run_validation()

    # Crear directorio para los resultados de la validación
    validation_dir = results_dir / f"{midi_name}_Validation"
    validation_dir.mkdir(exist_ok=True)

    confusion_matrix_path = validation_dir / "confusion_matrix.png"
    validator.plot_confusion_matrix(validation_result, str(confusion_matrix_path))

    validation_report_path = validation_dir / "validation_report.txt"
    validator.generate_validation_report(validation_result, str(validation_report_path))

    validation_csv_path = validation_dir / "validation_results.csv"
    validator.save_validation_results_csv(str(validation_csv_path))

    if VERBOSE_LOGGING:
        print(f"\n📊 RESUMEN DE VALIDACIÓN - {midi_name}")
        print(f"   Total mutaciones analizadas: {len(validator.results)}")
        print(f"   Precisión global: {validation_result['precision']:.3f}")
        print(f"   Recall global: {validation_result['recall']:.3f}")
        print(f"   F1-Score global: {validation_result['f1_score']:.3f}")
        print(f"   Exactitud global: {validation_result['accuracy']:.3f}")

        print(f"\n📁 Archivos de validación generados:")
        print(f"   ✅ Matriz de confusión: {confusion_matrix_path}")
        print(f"   ✅ Reporte detallado: {validation_report_path}")
        print(f"   ✅ Resultados CSV: {validation_csv_path}")

    # Retornar métricas para cálculo de promedio
    return {
        'accuracy': validation_result['accuracy'],
        'precision': validation_result['precision'],
        'recall': validation_result['recall'],
        'f1_score': validation_result['f1_score'],
        'total_mutations': len(validator.results)
    }


def generate_average_validation_report(all_validation_metrics: List[Dict[str, float]],
                                       midi_files_processed: List[str],
                                       categories_filter: Optional[List[str]],
                                       output_dir: str) -> None:
    """
    Genera un reporte promedio de validación para múltiples archivos MIDI.
    
    Args:
        all_validation_metrics: Lista de métricas de validación de todos los archivos
        midi_files_processed: Lista de nombres de archivos MIDI procesados
        categories_filter: Lista de categorías filtradas o None si se usaron todas
        output_dir: Directorio donde guardar el reporte promedio
    """
    from datetime import datetime

    if not all_validation_metrics:
        print("⚠️ No hay métricas de validación para calcular promedios")
        return

    # Filtrar métricas válidas (no vacías)
    valid_metrics = [m for m in all_validation_metrics if m]

    if not valid_metrics:
        print("⚠️ No hay métricas válidas para calcular promedios")
        return

    # Calcular promedios
    avg_accuracy = sum(m['accuracy'] for m in valid_metrics) / len(valid_metrics)
    avg_precision = sum(m['precision'] for m in valid_metrics) / len(valid_metrics)
    avg_recall = sum(m['recall'] for m in valid_metrics) / len(valid_metrics)
    avg_f1_score = sum(m['f1_score'] for m in valid_metrics) / len(valid_metrics)
    total_mutations = sum(m['total_mutations'] for m in valid_metrics)

    # Información sobre categorías
    categories_info = "Todas las categorías"
    if categories_filter:
        categories_info = ", ".join(categories_filter)

    # Generar reporte promedio
    report_content = f"""
REPORTE PROMEDIO DE VALIDACIÓN - METRONIA
{'=' * 60}
Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Archivos procesados: {len(valid_metrics)}
Categorías analizadas: {categories_info}
Total de mutaciones analizadas: {total_mutations}

MÉTRICAS PROMEDIO:
- Exactitud promedio (Accuracy): {avg_accuracy:.4f}
- Precisión promedio (Precision): {avg_precision:.4f}
- Recall promedio: {avg_recall:.4f}
- F1-Score promedio: {avg_f1_score:.4f}

ARCHIVOS PROCESADOS:
"""

    for i, midi_file in enumerate(midi_files_processed[:len(valid_metrics)], 1):
        midi_name = Path(midi_file).stem
        report_content += f"  {i}. {midi_name}\n"

    report_content += "\nMÉTRICAS INDIVIDUALES POR ARCHIVO:\n"

    for i, (midi_file, metrics) in enumerate(zip(midi_files_processed[:len(valid_metrics)], valid_metrics), 1):
        midi_name = Path(midi_file).stem
        report_content += f"""
Archivo {i} ({midi_name}):
  - Exactitud: {metrics['accuracy']:.4f}
  - Precisión: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - F1-Score: {metrics['f1_score']:.4f}
  - Mutaciones: {metrics['total_mutations']}
"""
    # Guardar reporte en la ubicación correcta
    report_path = Path(output_dir) / "average_validation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    csv_data = []
    csv_data.append({
        'archivo': 'PROMEDIO',
        'exactitud': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1_score,
        'duración': total_mutations,  # Para PROMEDIO, guardar número total de mutaciones
        'total_onsets_ref': sum(
            get_total_onsets_ref(Path(f).stem, output_dir) for f in midi_files_processed[:len(valid_metrics)])
    })
    for midi_file, metrics in zip(midi_files_processed[:len(valid_metrics)], valid_metrics):
        midi_name = Path(midi_file).stem

        duration = get_reference_audio_duration(midi_name, output_dir)
        total_onsets_ref = get_total_onsets_ref(midi_name, output_dir)

        csv_data.append({
            'archivo': midi_name,
            'exactitud': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'duración': duration,
            'total_onsets_ref': total_onsets_ref
        })

    csv_path = Path(output_dir) / "average_validation_metrics.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')

    if VERBOSE_LOGGING:
        print(f"\n" + "=" * 60)
        print("📊 REPORTE PROMEDIO DE VALIDACIÓN")
        print("=" * 60)
        print(f"Archivos procesados: {len(valid_metrics)}")
        print(f"Categorías analizadas: {categories_info}")
        print(f"Total mutaciones analizadas: {total_mutations}")
        print(f"Exactitud promedio: {avg_accuracy:.4f}")
        print(f"Precisión promedio: {avg_precision:.4f}")
        print(f"Recall promedio: {avg_recall:.4f}")
        print(f"F1-Score promedio: {avg_f1_score:.4f}")


def get_total_onsets_ref(midi_name: str, output_dir: str) -> int:
    """
    Obtiene el total de onsets de referencia desde mutations_summary.csv.
    
    Args:
        midi_name: Nombre del archivo MIDI (sin extensión)
        output_dir: Directorio base de resultados
        
    Returns:
        Número total de onsets de referencia, o 0 si no se puede obtener
    """
    try:
        mutations_summary_path = Path(output_dir) / f"{midi_name}_Mutaciones" / "mutations_summary.csv"

        if not mutations_summary_path.exists():
            print(f"⚠️ No se encontró mutations_summary.csv para {midi_name}")
            return 0

        df = pd.read_csv(mutations_summary_path)

        if 'total_onsets_ref' in df.columns and not df.empty:
            # Tomar el valor de la primera fila (debería ser el mismo para todas las mutaciones del mismo archivo)
            # Obtener el valor más alto de 'total_onsets_ref' si hay múltiples filas
            return int(max(df['total_onsets_ref']))
        else:
            print(f"⚠️ Columna 'total_onsets_ref' no encontrada en mutations_summary.csv para {midi_name}")
            return 0

    except Exception as e:
        print(f"⚠️ Error obteniendo total_onsets_ref para {midi_name}: {e}")
        return 0
