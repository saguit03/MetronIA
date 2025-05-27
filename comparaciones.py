
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.spatial.distance import cdist

# Comparación 1
def compare_beat_spectrums(beat_ref, beat_aligned, threshold=0.2):
    diff = np.abs(beat_ref - beat_aligned)
    if np.max(diff) > threshold:
        print("⚠️ Diferencias significativas en el beat spectrum.")
    else:
        print("✅ Beat spectrum similar.")
    return diff

def compare_onsets(audio_ref, audio_live, sr, margin=0.05):
    onsets_ref = librosa.onset.onset_detect(y=audio_ref, sr=sr, units='time')
    onsets_live = librosa.onset.onset_detect(y=audio_live, sr=sr, units='time')

    matched = []
    unmatched_ref = []
    unmatched_live = list(onsets_live)  # convertir a lista para poder usar .remove()

    for onset in onsets_ref:
        diffs = np.abs(np.array(unmatched_live) - onset)
        if len(diffs) > 0 and np.min(diffs) < margin:
            idx = np.argmin(diffs)
            matched.append((onset, unmatched_live[idx]))
            unmatched_live.pop(idx)  # eliminar el onset emparejado
        else:
            unmatched_ref.append(onset)

    print(f"✅ Onsets emparejados: {len(matched)}")
    print(f"❌ Notas faltantes (en vivo): {len(unmatched_ref)}")
    print(f"❌ Notas extras (en vivo): {len(unmatched_live)}")

    return matched, unmatched_ref, unmatched_live

def compare_local_tempo(audio_ref, audio_live, sr, threshold=5):
    tempo_ref, _ = librosa.beat.beat_track(y=audio_ref, sr=sr)
    tempo_live, _ = librosa.beat.beat_track(y=audio_live, sr=sr)

    tempo_ref = float(tempo_ref)
    tempo_live = float(tempo_live)

    print(f"🎼 Tempo referencia: {tempo_ref:.2f} BPM")
    print(f"🎼 Tempo en vivo: {tempo_live:.2f} BPM")

    if abs(tempo_ref - tempo_live) > threshold:
        print("⚠️ Diferencia significativa de tempo.")
    else:
        print("✅ Tempo similar.")

def evaluate_dtw_path(wp, tolerance=0.3):
    wp = np.array(wp)
    ref_idxs, live_idxs = wp[:, 0], wp[:, 1]
    deltas = live_idxs - ref_idxs
    deviations = np.abs(deltas - np.mean(deltas))

    if np.max(deviations) > tolerance * len(ref_idxs):
        print("⚠️ Camino DTW con desviaciones anómalas.")
    else:
        print("✅ Camino DTW razonablemente regular.")

    return deviations

def segment_and_validate(audio_ref, audio_live, sr, compas_dur=2.0, tolerance=0.2):
    duration_ref = librosa.get_duration(y=audio_ref, sr=sr)
    duration_live = librosa.get_duration(y=audio_live, sr=sr)

    n_compases_ref = int(duration_ref // compas_dur)
    n_compases_live = int(duration_live // compas_dur)

    print(f"🎵 Compases en referencia: {n_compases_ref}")
    print(f"🎵 Compases en vivo: {n_compases_live}")

    if abs(n_compases_ref - n_compases_live) > 1:
        print("⚠️ Desajuste en el número de compases.")
    elif abs(duration_ref - duration_live) > tolerance * duration_ref:
        print("⚠️ Diferencias en la duración de los compases.")
    else:
        print("✅ Estructura de compases compatible.")

# Comparación 2

def plot_onsets_errors(onsets_ref, onsets_live, matched, unmatched_ref, unmatched_live):
    plt.figure(figsize=(12, 3))

    # Onsets referencia (azul)
    plt.vlines(onsets_ref, 0.8, 1.0, color='blue', label='Onsets referencia')

    # Onsets emparejados (verde)
    matched_live = [live for _, live in matched]
    plt.vlines(matched_live, 0.6, 0.8, color='green', label='Onsets emparejados')

    # Onsets faltantes (negro)
    plt.vlines(unmatched_ref, 0.4, 0.6, color='black', label='Notas faltantes')

    # Onsets extras (rojo)
    plt.vlines(unmatched_live, 0.2, 0.4, color='red', label='Notas extras')

    plt.ylim(0, 1.1)
    plt.yticks([])
    plt.xlabel('Tiempo (segundos)')
    plt.title('Detección de errores rítmicos nota por nota')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def extract_features(audio, sr, hop_length=512, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
    return librosa.util.normalize(mfcc, axis=1).T  # (frames, features)

def compute_self_similarity_matrix(features):
    D = cdist(features, features, metric='cosine')
    S = 1 - D
    return S

def compute_beat_spectrum(S):
    n = S.shape[0]
    return np.array([np.mean(np.diag(S, k=lag)) for lag in range(1, n)])

def grafico_comparativo_1(reference_path, live_path, hop_length):
    reference_audio, sampling_rate = librosa.load(reference_path)
    live_audio, sampling_rate = librosa.load(live_path)
    ref_feat = extract_features(reference_audio, sampling_rate, hop_length)
    live_feat = extract_features(live_audio, sampling_rate, hop_length)
    # === DTW para alinear ===
    D, wp = librosa.sequence.dtw(X=ref_feat.T, Y=live_feat.T, metric='cosine')
    wp = np.array(wp[::-1])  # Asegura que va de principio a fin

    # === Aplicar alineamiento a live_feat ===
    aligned_live_feat = np.zeros_like(ref_feat)
    for i, (ref_idx, live_idx) in enumerate(wp):
        if ref_idx < len(aligned_live_feat) and live_idx < len(live_feat):
            aligned_live_feat[ref_idx] = live_feat[live_idx]

    # === Calcular matrices de autosemejanza ===
    S_ref = compute_self_similarity_matrix(ref_feat)
    S_aligned = compute_self_similarity_matrix(aligned_live_feat)

    # === Beat spectrums ===
    beat_ref = compute_beat_spectrum(S_ref)
    beat_aligned = compute_beat_spectrum(S_aligned)
    # Llamadas a las funciones de evaluación
    compare_beat_spectrums(beat_ref, beat_aligned)
    compare_onsets(reference_audio, live_audio, sampling_rate)
    compare_local_tempo(reference_audio, live_audio, sampling_rate)
    evaluate_dtw_path(wp)
    segment_and_validate(reference_audio, live_audio, sampling_rate, compas_dur=2.0)
