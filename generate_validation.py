from pathlib import Path
from tqdm import tqdm

import matplotlib
from collections import defaultdict
from pathlib import Path

matplotlib.use('Agg')
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

global_robustness = []
global_error_validation = []

class ValidationMetrics:
    def __init__(self, mutations_summary_path: str, mutation_dir: str):
        self.mutations_summary = pd.read_csv(mutations_summary_path)
        self.mutation_dir = Path(mutation_dir)
        self.robustness_results = []
        self.error_validation_results = []

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
            if mutation_category == 'timing' or mutation_category == 'note':
                self.validate_detection(mutation_name, mutation_category, logs_data, analysis_data)
            else:
                self.validate_robustness(mutation_name, mutation_category, logs_data, analysis_data)
        return self.get_overall_metrics()

    def validate_detection(self, mutation_name: str, mutation_category: str, logs_data, analysis_data):
        filtered_logs = logs_data[logs_data['onset_type'] != "no_change"].copy()
        if mutation_name == 'note_not_expected':
            dummy_me = pd.DataFrame([{'onset_type': 'extra'}])
            filtered_logs = pd.concat([filtered_logs, dummy_me], ignore_index=True)

        filtered_analysis = analysis_data[analysis_data['onset_type'] != "correct"].copy()

        detected_errors = False
        located_errors = False
        total_errors_introduced = len(filtered_logs)
        total_errors_analyzed = len(filtered_analysis)

        for _, row_log in filtered_logs.iterrows():
            for _, row_analysis in filtered_analysis.iterrows():
                if row_analysis['onset_type'] == row_log['onset_type']:
                    detected_errors = True
                    if row_analysis.name == row_log.name:
                        located_errors = True

        result_data = {
            'mutation_name': mutation_name,
            'category': mutation_category,
            'detected_errors': detected_errors,
            'located_errors': located_errors,
            'total_errors_introduced': total_errors_introduced,
            'total_errors_analyzed': total_errors_analyzed,
            'midi_id': self.mutation_dir.name.replace('_Mutaciones', '')
        }
        self.error_validation_results.append(result_data)


    def validate_robustness(self, mutation_name: str, mutation_category: str, logs_data, analysis_data):
        incorrect_data = analysis_data[analysis_data['onset_type'] != "correct"].copy()
        total_onsets = len(analysis_data)
        total_error = len(incorrect_data)
        error_rate = total_error / total_onsets if total_onsets > 0 else 0
        error_rate = round(error_rate, 4)

        result_data = {
            'mutation_name': mutation_name,
            'category': mutation_category,
            'total_onsets': total_onsets,
            'total_errors': total_error,
            'error_rate': error_rate,
            'midi_id': self.mutation_dir.name.replace('_Mutaciones', '')
        }
        self.robustness_results.append(result_data)


    def get_overall_metrics(self):
        all_robustness_results = pd.DataFrame(self.robustness_results)
        all_error_validation_results = pd.DataFrame(self.error_validation_results)
        all_robustness_results.to_excel(self.mutation_dir / "validation_robustness_results.xlsx", index=False)
        all_error_validation_results.to_excel(self.mutation_dir / "validation_error_results.xlsx", index=False)
        
        all_robustness_results.to_csv(self.mutation_dir / "validation_robustness_results.csv", index=False)
        all_error_validation_results.to_csv(self.mutation_dir / "validation_error_results.csv", index=False)

        global_robustness.extend(self.robustness_results)
        global_error_validation.extend(self.error_validation_results)
  

def create_validation_files(mutations_dir):
    mutations_summary_path = Path(mutations_dir) / "mutations_summary.csv"

    if not mutations_summary_path.exists():
        print(f"⚠️ No se encontró el archivo de resumen de mutaciones: {mutations_summary_path}")
        return {}

    validator = ValidationMetrics(
        mutations_summary_path=str(mutations_summary_path),
        mutation_dir=str(mutations_dir)
    )

    validator.run_validation()
    
def validate_pipeline(output_dir: Path):
    
    for subdir in tqdm(output_dir.iterdir(), desc="Creando ficheros de validación", unit="subdirectorio", dynamic_ncols=True):
        if subdir.is_dir() and subdir.name.endswith("_Mutaciones"):
            create_validation_files(subdir)

    validate_path = output_dir / "__Validation_Results"
    Path(validate_path).mkdir(parents=True, exist_ok=True)  

    category_path = validate_path / "Category"
    Path(category_path).mkdir(parents=True, exist_ok=True)

    if global_robustness:
        df_robustness = pd.DataFrame(global_robustness)
        df_robustness.to_excel(validate_path / "global_robustness.xlsx", index=False)
        df_robustness.to_csv(validate_path / "global_robustness.csv", index=False)

        robustness_path = validate_path / "Robustness"
        Path(robustness_path).mkdir(parents=True, exist_ok=True)

        # Agrupar por mutation_name
        grouped_by_mutation = defaultdict(list)
        for row in global_robustness:
            grouped_by_mutation[row['mutation_name']].append(row)
        for mutation_name, rows in grouped_by_mutation.items():
            df_group = pd.DataFrame(rows)
            df_group.to_excel(robustness_path / f"robustness_{mutation_name}.xlsx", index=False)
            df_group.to_csv(robustness_path / f"robustness_{mutation_name}.csv", index=False)

        # Agrupar por category
        grouped_by_category = defaultdict(list)
        for row in global_robustness:
            grouped_by_category[row['category']].append(row)
        for category, rows in grouped_by_category.items():
            df_group = pd.DataFrame(rows)
            df_group.to_excel(category_path / f"robustness_category_{category}.xlsx", index=False)
            df_group.to_csv(category_path / f"robustness_category_{category}.csv", index=False)

    if global_error_validation:
        df_error_validation = pd.DataFrame(global_error_validation)
        df_error_validation.to_excel(validate_path / "global_error_validation.xlsx", index=False)
        df_error_validation.to_csv(validate_path / "global_error_validation.csv", index=False)
        
        errors_path = validate_path / "Errors"
        Path(errors_path).mkdir(parents=True, exist_ok=True)

        # Agrupar por mutation_name
        grouped_by_mutation = defaultdict(list)
        for row in global_error_validation:
            grouped_by_mutation[row['mutation_name']].append(row)
        for mutation_name, rows in grouped_by_mutation.items():
            df_group = pd.DataFrame(rows)
            df_group.to_excel(errors_path / f"errors_{mutation_name}.xlsx", index=False)
            df_group.to_csv(errors_path / f"errors_{mutation_name}.csv", index=False)

        # Agrupar por category
        grouped_by_category = defaultdict(list)
        for row in global_error_validation:
            grouped_by_category[row['category']].append(row)
        for category, rows in grouped_by_category.items():
            df_group = pd.DataFrame(rows)
            df_group.to_excel(category_path / f"errors_category_{category}.xlsx", index=False)
            df_group.to_csv(category_path / f"errors_category_{category}.csv", index=False)


if __name__ == "__main__":
    
    base_dir = "results"
    mutation_dir = "validation_gaps"
    output_dir = Path(base_dir) / Path(mutation_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    validate_pipeline(output_dir)

