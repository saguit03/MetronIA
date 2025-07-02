import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
            if mutation_category == 'timing':
                self.validate_timing(mutation_name, mutation_category, logs_data, analysis_data)
            elif mutation_category == 'duration':
                self.validate_duration(mutation_name, mutation_category, logs_data, analysis_data)
            else:
                self.validate_mutation(mutation_name, mutation_category, logs_data, analysis_data)
        return self.get_overall_metrics()
    
    def validate_timing(self, mutation_name, mutation_category, logs_data, analysis_data):
        logs_data['onset_time'] = logs_data['onset_time'].round(2)
        analysis_data['onset_ref_time'] = analysis_data['onset_ref_time'].round(2)
        analysis_data['onset_live_time'] = analysis_data['onset_live_time'].round(2)

        filtered_logs = logs_data[logs_data['onset_type'] != "no_change"].copy()

        y_true = []
        y_pred = []

        ref_times = analysis_data['onset_ref_time'].values

        for _, row in filtered_logs.iterrows():
            onset_time = row['onset_time']
            true_type = row['onset_type']

            idx_closest = np.argmin(np.abs(ref_times - onset_time))
            pred_type = analysis_data.iloc[idx_closest]['onset_type']

            y_true.append(true_type)
            y_pred.append(pred_type)

        result_data = {
            'mutation_name': mutation_name,
            'category': mutation_category,
            'y_true': self.map_onset_types(y_true),
            'y_pred': self.map_onset_types(y_pred),
        }

        self.results.append(result_data)
        self.results_by_category[mutation_category].append(result_data)

    def validate_duration(self, mutation_name, mutation_category, logs_data, analysis_data):
        filtered_logs = logs_data[logs_data['onset_type'] != "no_change"].copy()

        y_true = []
        y_pred = []

        ref_times = analysis_data['onset_ref_time'].values

        for _, row in filtered_logs.iterrows():
            onset_time = row['onset_time']
            idx_closest = np.argmin(np.abs(ref_times - onset_time))
            pred_type = analysis_data.iloc[idx_closest]['onset_type']
            y_true.append('correct')
            y_pred.append(pred_type)

        result_data = {
            'mutation_name': mutation_name,
            'category': mutation_category,
            'y_true': self.map_onset_types(y_true),
            'y_pred': self.map_onset_types(y_pred),
        }

        self.results.append(result_data)
        self.results_by_category[mutation_category].append(result_data)

    def validate_mutation(self, mutation_name, mutation_category, logs_data, analysis_data):
        y_pred_raw = analysis_data['onset_type'].tolist()
        onset_refs = analysis_data['onset_ref_time'].tolist()

        if len(y_pred_raw) == 0:
            print(f"Warning: No onset data found for mutation {mutation_name}")
            return

        is_tempo_or_articulation = False
        has_onset_time = 'onset_time' in logs_data.columns
        
        if 'onset_type' in logs_data.columns:
            is_tempo_or_articulation = (logs_data['onset_type'] == "tempo").any() or (logs_data['onset_type'] == "articulation").any()
        
        if is_tempo_or_articulation or not has_onset_time:
            y_true = ["correct"] * len(y_pred_raw)
            y_pred = self.map_onset_types(y_pred_raw)
        else:
            logs_times = logs_data['onset_time'].tolist()
            logs_types = logs_data['onset_type'].tolist()

            y_true_raw = []
            for ref_time in onset_refs:
                closest_idx = np.argmin(np.abs(np.array(logs_times) - ref_time))
                y_true_raw.append(logs_types[closest_idx])

            y_true = self.map_onset_types(y_true_raw)
            y_pred = self.map_onset_types(y_pred_raw)

            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
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
        if cm is not None:
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
    validator.plot_confusion_matrix(validation_result["confusion_matrix"], validation_result["labels"], str(confusion_matrix_path))

    category_metrics = validator.get_metrics_by_category()

    validation_by_category_data = []

    for category, metrics in category_metrics.items():
        category_confusion_matrix_path = validation_results_dir / f"confusion_matrix_{category}.png"
        validator.plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"],
                                        str(category_confusion_matrix_path))

        validation_by_category_data.append({
            'category': category,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'total_onsets_ref': metrics['total_onsets_ref'],
            'total_onsets_live': metrics['total_onsets_live'],
        })

    validation_by_category_data.append({
        'category': 'PROMEDIO',
        'accuracy': validation_result['accuracy'],
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

def calculate_global_validation_results(output_dir: Path, processed_files: list):
    calculate_global_validation_results_by_category(output_dir, processed_files)
    calculate_global_validation_results_by_file(output_dir, processed_files)

def calculate_global_validation_results_by_category(output_dir: Path, processed_files: list):
    all_dataframes = []
    
    for midi_file_path in processed_files:
        midi_name = Path(midi_file_path).stem
        validation_file = output_dir / f"{midi_name}_Mutaciones" / "Validation_Results" / "validation_results_by_category.csv"
        
        if validation_file.exists():
            df = pd.read_csv(validation_file)
            if not df.empty:
                all_dataframes.append(df)

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df = combined_df.drop(columns=['total_onsets_ref', 'total_onsets_live'], errors='ignore')
    
    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
    
    if 'category' in combined_df.columns:
        global_results = combined_df.groupby('category')[numeric_columns].mean().reset_index()
        global_results[numeric_columns] = global_results[numeric_columns].round(4)

    global_file = output_dir / "global_validation_results_by_category.csv"
    global_results.to_csv(global_file, index=False, encoding='utf-8')
    
    print(f"📈 Categorías procesadas: {len(global_results) - 1}")  # -1 para excluir el promedio general
    return global_file


def calculate_global_validation_results_by_file(output_dir: Path, processed_files: list):
    all_file_results = []
    
    for midi_file_path in processed_files:
        midi_name = Path(midi_file_path).stem
        validation_file = output_dir / f"{midi_name}_Mutaciones" / "Validation_Results" / "validation_results_by_category.csv"
        
        if validation_file.exists():
                df = pd.read_csv(validation_file)
                if not df.empty:
                    promedio_row = df[df['category'] == 'PROMEDIO']
                    
                    if not promedio_row.empty:
                        file_result = promedio_row.iloc[0].copy()
                        file_result['category'] = midi_name
                        file_result_dict = file_result.to_dict()
                        file_result_dict['midi'] = file_result_dict.pop('category')
                        
                        all_file_results.append(file_result_dict)
    
    global_results_df = pd.DataFrame(all_file_results)
    if not global_results_df.empty:
        cols = list(global_results_df.columns)
        if 'midi' in cols:
            cols.insert(0, cols.pop(cols.index('midi')))
            global_results_df = global_results_df[cols]
    global_file = output_dir / "global_validation_results_by_file.csv"
    global_results_df.to_csv(global_file, index=False, encoding='utf-8')
    
    return global_file