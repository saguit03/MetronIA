from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
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
        self.results_by_category = defaultdict(list)  

    def get_logs_data(self, mutation_name, mutation_category):
        csv_log_path = self.mutation_dir / "logs" / f"{mutation_name}.csv"
        if not csv_log_path.exists():
            print(f"Warning: Log file not found for mutation {mutation_name}")
            return None
        return pd.read_csv(csv_log_path)
    
    def get_analysis_data(self, mutation_name):
        midi_name = self.mutation_dir.name.replace('_Mutaciones', '')
        analysis_csv_path = self.mutation_dir / f"{midi_name}_{mutation_name}" / f"{mutation_name}_analysis.csv"
        if not analysis_csv_path.exists():
            print(f"Warning: Analysis CSV not found for mutation {mutation_name}")
            return None
        return pd.read_csv(analysis_csv_path)
        
    def run_validation(self):
        for _, row in self.mutations_summary.iterrows():
            mutation_name = row['mutation_name']
            mutation_category = row['mutation_category']
            logs_data = self.get_logs_data(mutation_name, mutation_category)
            analysis_data = self.get_analysis_data(mutation_name)
            if logs_data is None or analysis_data is None:
                continue
            self.validate_mutation(mutation_name, mutation_category, logs_data, analysis_data)
        return self.get_overall_metrics()

    def validate_mutation(self, mutation_name, mutation_category, logs_data, analysis_data):
        if logs_data is None or analysis_data is None:
            print(f"Warning: Missing data for mutation {mutation_name}")
            return
        
        if 'onset_type' not in analysis_data.columns:
            print(f"Warning: 'onset_type' column missing in analysis data for {mutation_name}")
            return
            
        if 'onset_type' not in logs_data.columns:
            print(f"Warning: 'onset_type' column missing in logs data for {mutation_name}")
            return
            
        y_pred = self.map_onset_types(analysis_data['onset_type'].tolist())
        
        if len(y_pred) == 0:
            print(f"Warning: No onset data found for mutation {mutation_name}")
            return
        
        if (logs_data['onset_type'] == "tempo").all() or (logs_data['onset_type'] == "articulation").all():
            y_true = ["correct"] * len(y_pred)
        else:
            y_true_raw = self.map_onset_types(logs_data['onset_type'].tolist())
            min_length = min(len(y_true_raw), len(y_pred))
            y_true = y_true_raw[:min_length]
            y_pred = y_pred[:min_length]
            
        result_data = {
            'mutation_name': mutation_name,
            'category': mutation_category,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        self.results.append(result_data)
        self.results_by_category[mutation_category].append(result_data)

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

        if len(all_true) != len(all_pred):
            min_length = min(len(all_true), len(all_pred))
            all_true = all_true[:min_length]
            all_pred = all_pred[:min_length]
            print(f"Truncated to length: {min_length}")

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

    def plot_confusion_matrix(self, cm, labels, output_path):

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def save_validation_results_csv(self, validation_result, output_path):
        data = {
            'accuracy': [validation_result['accuracy']],
            'precision': [validation_result['precision']],
            'recall': [validation_result['recall']],
            'f1_score': [validation_result['f1_score']],
        }
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def get_metrics_by_category(self) -> Dict[str, Dict[str, float]]:
        """Obtiene métricas de validación agrupadas por categoría"""
        category_metrics = {}
        
        for category, results in self.results_by_category.items():
            all_true = []
            all_pred = []
            
            for res in results:
                all_true.extend(res['y_true'])
                all_pred.extend(res['y_pred'])
            
            if not all_true and not all_pred:
                category_metrics[category] = {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "total_mutations": 0
                }
                continue
                
            if category == "tempo_errors":
                all_true = []
                all_pred = []
                for result in results:
                    y_pred = result.get("y_pred", [])
                    all_true.extend(["correct"] * len(y_pred))
                    all_pred.extend(y_pred)
                labels = ["correct", "faster_tempo", "slower_tempo", "a_lot_faster_tempo", "a_lot_slower_tempo"]
            else:
                labels = sorted(list(set(all_true + all_pred)))
                
            if len(all_true) != len(all_pred):
                min_length = min(len(all_true), len(all_pred))
                all_true = all_true[:min_length]
                all_pred = all_pred[:min_length]
            
            cm = confusion_matrix(all_true, all_pred, labels=labels)
            accuracy = accuracy_score(all_true, all_pred)
            precision = precision_score(all_true, all_pred, average='weighted', zero_division=0)
            recall = recall_score(all_true, all_pred, average='weighted', zero_division=0)
            f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)
            
            category_metrics[category] = {
                "confusion_matrix": cm,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "labels": labels,
                "total_mutations": len(results)
            }
        
        return category_metrics

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
    
    mutations_dir = results_dir / f"{midi_name}_Mutaciones"
    mutations_summary_path = mutations_dir / "mutations_summary.csv"
    
    if not mutations_summary_path.exists():
        print(f"⚠️ No se encontró el archivo de resumen de mutaciones: {mutations_summary_path}")
        return {}
    
    validator = MutationValidation(
        mutations_summary_path=str(mutations_summary_path),
        mutation_dir=str(mutations_dir)
    )
    
    validation_result = validator.run_validation()

    validation_results_dir = mutations_dir / "Validation_Results"
    validation_results_dir.mkdir(exist_ok=True)

    confusion_matrix_path = validation_results_dir / "confusion_matrix.png"
    validator.plot_confusion_matrix(validation_result["confusion_matrix"], validation_result["labels"], str(confusion_matrix_path))

    category_metrics = validator.get_metrics_by_category()
    
    validation_by_category_data = []
    
    for category, metrics in category_metrics.items():
        validation_by_category_data.append({
            'archivo': midi_name,
            'categoria': category,
            'exactitud': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'total_mutaciones': metrics['total_mutations']
        })
    
    validation_by_category_data.append({
        'archivo': midi_name,
        'categoria': 'PROMEDIO',
        'exactitud': validation_result['accuracy'],
        'precision': validation_result['precision'],
        'recall': validation_result['recall'],
        'f1_score': validation_result['f1_score'],
        'total_mutaciones': len(validator.results)
    })
    
    df_validation_results = pd.DataFrame(validation_by_category_data)
    validation_results_path = validation_results_dir / "validation_results_by_category.csv"
    df_validation_results.to_csv(validation_results_path, index=False)

    for category, metrics in category_metrics.items():
        category_data = []
        
        category_results = validator.results_by_category[category]
        for result in category_results:
            category_data.append({
                'mutation_name': result['mutation_name'],
                'category': result['category'],
                'y_true': result['y_true'],
                'y_pred': result['y_pred']
            })
            
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

    valid_metrics = [m for m in all_validation_metrics if m]

    if not valid_metrics:
        print("⚠️ No hay métricas válidas para calcular promedios")
        return
    
    avg_accuracy = round(sum(m['accuracy'] for m in valid_metrics) / len(valid_metrics), 4)
    avg_precision = round(sum(m['precision'] for m in valid_metrics) / len(valid_metrics), 4)
    avg_recall = round(sum(m['recall'] for m in valid_metrics) / len(valid_metrics), 4)
    avg_f1_score = round(sum(m['f1_score'] for m in valid_metrics) / len(valid_metrics), 4)
    total_mutations = sum(m['total_mutations'] for m in valid_metrics)

    categories_info = "Todas las categorías"
    if categories_filter:
        categories_info = ", ".join(categories_filter)

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
    validation_results_dir = Path(output_dir) / "Validation_Results"
    validation_results_dir.mkdir(exist_ok=True)
    
    report_path = validation_results_dir / "average_validation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    csv_data = []
    csv_data.append({
        'archivo': 'PROMEDIO',
        'exactitud': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1_score,
        'duración': total_mutations,
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
    
    df = pd.DataFrame(csv_data)
    validation_csv_path = validation_results_dir / "average_validation_by_file.csv"
    df.to_csv(validation_csv_path, index=False, encoding='utf-8')


def generate_category_validation_reports(all_validation_metrics: List[Dict[str, float]],
                                        midi_files_processed: List[str],
                                        categories_filter: Optional[List[str]],
                                        output_dir: str) -> None:
    """
    Genera reportes de validación específicos por categoría de mutación.
    
    Args:
        all_validation_metrics: Lista de métricas de validación de todos los archivos
        midi_files_processed: Lista de nombres de archivos MIDI procesados
        categories_filter: Lista de categorías filtradas o None si se usaron todas
        output_dir: Directorio donde guardar los reportes
    """
    from collections import defaultdict
    from datetime import datetime
    
    category_data = defaultdict(list)
    
    for midi_file in midi_files_processed:
        midi_name = Path(midi_file).stem
        
        validation_results_file = Path(output_dir) / f"{midi_name}_Mutaciones" / "Validation_Results" / "validation_results_by_category.csv"
        
        if not validation_results_file.exists():
            if VERBOSE_LOGGING:
                print(f"⚠️ No se encontró {validation_results_file}")
            continue
            
        try:
            df = pd.read_csv(validation_results_file)
            if not df.empty:
                df_categories = df[df['categoria'] != 'PROMEDIO']
                
                for _, row in df_categories.iterrows():
                    category = row['categoria']
                    category_data[category].append({
                        'midi_name': midi_name,
                        'exactitud': row['exactitud'],
                        'precision': row['precision'],
                        'recall': row['recall'],
                        'f1_score': row['f1_score'],
                        'total_mutaciones': row['total_mutaciones']
                    })
        except Exception as e:
            print(f"⚠️ Error leyendo {validation_results_file}: {e}")
    
    main_validation_dir = Path(output_dir) / "Validation_Results"
    main_validation_dir.mkdir(exist_ok=True)
    
    for category, category_results in category_data.items():
        if not category_results:
            continue
            
        category_csv_path = main_validation_dir / f"{category}_validation.csv"
        
        category_metrics_data = []
        
        for result in category_results:
            category_metrics_data.append({
                'archivo': result['midi_name'],
                'categoria': category,
                'exactitud': result['exactitud'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'total_mutaciones': result['total_mutaciones']
            })
        
        if category_metrics_data:
            avg_exactitud = round(sum(r['exactitud'] for r in category_results) / len(category_results), 4)
            avg_precision = round(sum(r['precision'] for r in category_results) / len(category_results), 4)
            avg_recall = round(sum(r['recall'] for r in category_results) / len(category_results), 4)
            avg_f1_score = round(sum(r['f1_score'] for r in category_results) / len(category_results), 4)
            total_mutaciones_sum = sum(r['total_mutaciones'] for r in category_results)
            
            category_metrics_data.append({
                'archivo': 'PROMEDIO',
                'categoria': category,
                'exactitud': avg_exactitud,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1_score,
                'total_mutaciones': total_mutaciones_sum
            })
            
            df_category_metrics = pd.DataFrame(category_metrics_data)
            df_category_metrics.to_csv(category_csv_path, index=False, encoding='utf-8')
            
            if VERBOSE_LOGGING:
                print(f"📊 Categoría {category}:")
                print(f"   Exactitud promedio: {avg_exactitud:.4f}")
                print(f"   Precisión promedio: {avg_precision:.4f}")
                print(f"   Recall promedio: {avg_recall:.4f}")
                print(f"   F1-Score promedio: {avg_f1_score:.4f}")
                print(f"   Total mutaciones: {total_mutaciones_sum}")
                print(f"   Archivo guardado: {category_csv_path}")
    
    if category_data:
        category_summary_data = []
        
        for category, category_results in category_data.items():
            if not category_results:
                continue
            
            avg_exactitud = round(sum(r['exactitud'] for r in category_results) / len(category_results), 4)
            avg_precision = round(sum(r['precision'] for r in category_results) / len(category_results), 4)
            avg_recall = round(sum(r['recall'] for r in category_results) / len(category_results), 4)
            avg_f1_score = round(sum(r['f1_score'] for r in category_results) / len(category_results), 4)
            total_mutaciones_sum = sum(r['total_mutaciones'] for r in category_results)
            
            category_summary_data.append({
                'categoria': category,
                'exactitud': avg_exactitud,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1_score,
                'total_mutaciones': total_mutaciones_sum,
                'archivos_procesados': len(category_results)
            })
        
        if category_summary_data:
            summary_csv_path = main_validation_dir / "average_validation_by_category.csv"
            df_summary = pd.DataFrame(category_summary_data)
            df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')            
            if VERBOSE_LOGGING:
                print(f"📊 Resumen por categorías guardado en: {summary_csv_path}")
    
    if VERBOSE_LOGGING:
        print(f"\n✅ Reportes de validación por categoría completados")
        print(f"📁 Directorio: {main_validation_dir}")


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
            return int(max(df['total_onsets_ref']))
        else:
            print(f"⚠️ Columna 'total_onsets_ref' no encontrada en mutations_summary.csv para {midi_name}")
            return 0

    except Exception as e:
        print(f"⚠️ Error obteniendo total_onsets_ref para {midi_name}: {e}")
        return 0
