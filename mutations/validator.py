import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from collections import defaultdict

from utils.config import VERBOSE_LOGGING


class MutationValidation:
    def __init__(self, mutations_summary_path: str, mutation_dir: str):
        self.mutations_summary = pd.read_csv(mutations_summary_path)
        self.mutation_dir = Path(mutation_dir)
        self.results = []
        self.results_by_category = defaultdict(list)

    def get_logs_data(self, mutation_name):
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
            logs_data = self.get_logs_data(mutation_name)
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
        return self.get_metrics(self.results)

    def plot_confusion_matrix(self, cm, labels, output_path):
        plt.figure(figsize=(12, 8))
        plt.title('Confusion Matrix')
        if cm is None:
            plt.text(0.5, 0.5, 'No Confusion Matrix: Perfect prediction', horizontalalignment='center',
                     verticalalignment='center', fontsize=20)
            plt.axis('off')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def get_metrics(self, results_list):
        all_true = []
        all_pred = []

        for res in results_list:
            all_true.extend(res['y_true'])
            all_pred.extend(res['y_pred'])

        labels = sorted(list(set(all_true + all_pred)))

        if len(all_true) != len(all_pred):
            min_length = min(len(all_true), len(all_pred))
            all_true = all_true[:min_length]
            all_pred = all_pred[:min_length]

        if len(labels) > 1:
            cm = confusion_matrix(all_true, all_pred, labels=labels)
        else:
            cm = None

        accuracy = accuracy_score(all_true, all_pred)
        precision = precision_score(all_true, all_pred, average='weighted', zero_division=0)
        recall = recall_score(all_true, all_pred, average='weighted', zero_division=0)
        f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)

        return {
            "confusion_matrix": cm,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "labels": labels,
            "total_mutations": len(results_list),
            "total_onsets_ref": len(all_true),
            "total_onsets_live": len(all_pred),
        }

    def get_metrics_by_category(self) -> Dict[str, Dict[str, float]]:
        category_metrics = {}

        for category, results in self.results_by_category.items():
            category_metrics[category] = self.get_metrics(results)

        return category_metrics


def run_validation_analysis(midi_name: str, results_dir: Path) -> Dict[str, float]:
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
    validator.plot_confusion_matrix(validation_result["confusion_matrix"], validation_result["labels"],
                                    str(confusion_matrix_path))

    category_metrics = validator.get_metrics_by_category()

    validation_by_category_data = []

    for category, metrics in category_metrics.items():
        category_confusion_matrix_path = validation_results_dir / f"confusion_matrix_{category}.png"
        validator.plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"],
                                        str(category_confusion_matrix_path))

        validation_by_category_data.append({
            'archivo': midi_name,
            'categoria': category,
            'exactitud': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'total_onsets_ref': metrics['total_onsets_ref'],
            'total_onsets_live': metrics['total_onsets_live'],
        })

    validation_by_category_data.append({
        'archivo': midi_name,
        'categoria': 'PROMEDIO',
        'exactitud': validation_result['accuracy'],
        'precision': validation_result['precision'],
        'recall': validation_result['recall'],
        'f1_score': validation_result['f1_score'],
        'total_onsets_ref': validation_result['total_onsets_ref'],
        'total_onsets_live': validation_result['total_onsets_live'],
    })

    df_validation_results = pd.DataFrame(validation_by_category_data)
    validation_results_path = validation_results_dir / "validation_results_by_category.csv"
    df_validation_results.to_csv(validation_results_path, index=False)
    return {
        'accuracy': validation_result['accuracy'],
        'precision': validation_result['precision'],
        'recall': validation_result['recall'],
        'f1_score': validation_result['f1_score'],
        'total_mutations': len(validator.results),
        'total_onsets_ref': validation_result['total_onsets_ref'],
        'total_onsets_live': validation_result['total_onsets_live']
    }
