from typing import Dict, Any, List

import pandas as pd
import traceback
from tqdm import tqdm

from analyzers import MetronIA
from .config import VERBOSE_LOGGING
from .midi_utils import save_mutation_complete

def aplicar_mutaciones(mutation_manager, original_excerpt, base_tempo, midi_name, results_dir):
    successful_mutations = []
    failed_mutations = []
    
    mutations_base_dir = results_dir / f"{midi_name}_Mutaciones"
    mutations_base_dir.mkdir(exist_ok=True)

    for category_name, category in mutation_manager.categories.items():
        for mutation_name, mutation in category.mutations.items():
            try:
                success = mutation.apply(
                    original_excerpt,
                    tempo=base_tempo,
                    output_dir=str(mutations_base_dir)
                )

                if success and mutation.excerpt is not None:
                    file_name = f"{midi_name}_{mutation_name}"
                    audio_path, midi_path, calculated_tempo = save_mutation_complete(
                        mutation_result=mutation,
                        mutation_name=midi_name,
                        save_name=file_name,
                        base_tempo=base_tempo
                    )

                    mutation.set_audio_path(audio_path)
                    mutation.set_midi_path(midi_path)

                    successful_mutations.append((category_name, mutation_name, mutation, audio_path, original_excerpt))
                else:
                    failed_mutations.append((category_name, mutation_name, mutation.error or "Unknown error"))

            except Exception as e:
                failed_mutations.append((category_name, mutation_name, str(e)))

    if failed_mutations:
        print(f"\n⚠️ Mutaciones fallidas:")
        for category, name, error in failed_mutations:
            print(f"  - {category}.{name}: {error}")
            
    return successful_mutations



def analizar_mutaciones(analyzer, successful_mutations, reference_audio_path, base_tempo, midi_name, results_dir):
    csv_data = []
    mutations_base_dir = results_dir / f"{midi_name}_Mutaciones"
    progress_bar = tqdm(successful_mutations, desc=f"Iniciando análisis de {midi_name}", unit="mutación",
                        dynamic_ncols=True, ascii=True)
    for category_name, mutation_name, mutation, audio_path, original_excerpt in progress_bar:
        try:
            progress_bar.set_description(f"{midi_name}: Analizando {mutation_name}")
            analysis_name = f"{midi_name}_{mutation_name}"

            analysis_dir = mutations_base_dir / analysis_name
            analysis_dir.mkdir(parents=True, exist_ok=True)            
            analysis_result = analyzer.comprehensive_analysis(
                reference_path=reference_audio_path,
                live_path=audio_path,
                save_name=analysis_name,
                save_dir=str(analysis_dir),
                reference_tempo=base_tempo,
                mutation_name=mutation_name
            )
            dtw_onset_result = analysis_result.get('dtw_onsets')
            if dtw_onset_result:
                csv_data.append(get_mutation_result_row(category_name=category_name,
                                             mutation_name=mutation_name,
                                             analysis_result=analysis_result,
                                             dtw_onset_result=dtw_onset_result))
                progress_bar.set_postfix_str(f"✅ {category_name}")
            else:
                progress_bar.set_postfix_str(f"⚠️ Sin resultados DTW")
        except Exception as e:
            traceback.print_exc()
            progress_bar.set_postfix_str(f"❌ Error: {str(e)}")
    progress_bar.set_description("Análisis completado")
    progress_bar.close()

    if csv_data:
        mutations_summary_dir = results_dir / f"{midi_name}_Mutaciones"
        mutations_summary_dir.mkdir(parents=True, exist_ok=True)
        csv_file = mutations_summary_dir / "mutations_summary.csv"
        save_analysis_results_to_csv(csv_data, csv_file)

def save_analysis_results_to_csv(analysis_data: List[Dict[str, Any]], output_file: str):
    if not analysis_data:
        print("⚠️ No hay datos de análisis para guardar en CSV")
        return

    df = pd.DataFrame(analysis_data)

    df.to_csv(output_file, index=False, encoding='utf-8')
    if VERBOSE_LOGGING: print(f"✅ Resultados guardados en CSV: {output_file}")

    return df

def get_mutation_result_row(category_name, mutation_name, analysis_result, dtw_onset_result):
    return {
                    'mutation_category': category_name,
                    'mutation_name': mutation_name,
                    'total_onsets_ref': len(dtw_onset_result.matches) + len(dtw_onset_result.missing_onsets),
                    'total_onsets_live': len(dtw_onset_result.matches) + len(dtw_onset_result.extra_onsets),
                    'correct_matches': len(
                        [m for m in dtw_onset_result.matches if m.classification.value == 'correct']),
                    'late_matches': len([m for m in dtw_onset_result.matches if m.classification.value == 'late']),
                    'early_matches': len([m for m in dtw_onset_result.matches if m.classification.value == 'early']),
                    'missing_onsets': len(dtw_onset_result.missing_onsets),
                    'extra_onsets': len(dtw_onset_result.extra_onsets),
                    'corret_notes': dtw_onset_result.correct_notes,
                    'beat_spectrum_max_difference': f"{analysis_result['beat_spectrum'].max_difference:.3f}",
                    'tempo_reference_bpm': f"{analysis_result['tempo'].tempo_ref:.2f}",
                    'tempo_live_bpm': f"{analysis_result['tempo'].tempo_live:.2f}",
                    'tempo_difference_bpm': f"{analysis_result['tempo'].difference:.2f}",
                }