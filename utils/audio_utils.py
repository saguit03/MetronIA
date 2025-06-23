from pathlib import Path
from typing import Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrubberband.pyrb as pyrb
import scipy.io.wavfile as wavfile

N_ARROWS = 10  # Número de flechas para la visualización

def check_extension(file_path: str, midi_name) -> str:
    if Path(file_path).suffix.lower() == '.mid':
        result = obtener_audio_de_midi(file_path, midi_name)
        if result is None:
            raise ValueError(f"Error obtaining audio from MIDI file: {file_path}")
        reference_audio, tempo, audio_path = result
    else:
        audio_path = file_path
    return str(audio_path)


def load_audio_files(reference_path: str, live_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Carga archivos de audio."""
    reference_audio, sr = librosa.load(reference_path)
    sr = int(sr)  # Ensure sr is always an int
    live_audio, _ = librosa.load(live_path, sr=sr)  # Usar mismo sr

    return reference_audio, live_audio, sr


def calculate_warping_path(reference_audio: np.ndarray, live_audio: np.ndarray, fs: int, hop_length: int):
    """Extrae las características de cromograma de dos audios."""
    reference_chroma = librosa.feature.chroma_cqt(y=reference_audio, sr=fs, hop_length=hop_length)
    live_chroma = librosa.feature.chroma_cqt(y=live_audio, sr=fs, hop_length=hop_length)
    D, wp = librosa.sequence.dtw(X=reference_chroma, Y=live_chroma, metric='cosine')
    wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)
    return D, wp, wp_s


def sinc(x, wp_s, sample_rate, out_len, n_arrows):
    time_map = [(int(x * sample_rate), int(y * sample_rate)) for (x, y) in wp_s[::len(wp_s) // n_arrows]]
    time_map.append((len(x), out_len))
    return pyrb.timemap_stretch(x, sample_rate, time_map)


def sinc_creciente(x, wp_s, sample_rate, out_len, n_arrows):
    #raw_map = [(int(t2 * sample_rate), int(t1 * sample_rate)) for t1, t2 in wp_s[::len(wp_s) // n_arrows]]
    raw_map = [(int(t2 * sample_rate), int(t1 * sample_rate)) for t1, t2 in wp_s[::n_arrows]]

    raw_map.sort(key=lambda pair: pair[0])
    raw_map.append((len(x), out_len))

    time_map = []
    last_in, last_out = -1, -1
    for t_in, t_out in raw_map:
        if t_in > last_in and t_out > last_out:
            time_map.append((t_in, t_out))
            last_in, last_out = t_in, t_out
        elif t_in == last_in or t_out == last_out:
            time_map[-1] = (t_in, t_out)

    return pyrb.timemap_stretch(x, sample_rate, time_map)


def save_comparative_plot(x_audio: np.ndarray, y_audio: np.ndarray, fs: int, save_name, save_dir):
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 4))
    librosa.display.waveshow(x_audio, sr=fs, ax=ax2)
    ax2.set(title='Referencia')
    librosa.display.waveshow(y_audio, sr=fs, ax=ax1)
    ax1.set(title='Alineado')
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_filename = Path(save_dir) / f"{save_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_audio(audio, save_name, save_dir, sample_rate):
    """Guarda el audio en un fichero WAV."""
    audio_normalized = np.int16(audio / np.max(np.abs(audio)) * 32767)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_filename = Path(save_dir) / f"{save_name}.wav"
    wavfile.write(output_filename, sample_rate, audio_normalized)
    return output_filename


def stretch_audio(x_audio: np.ndarray, y_audio: np.ndarray, wp_s, fs: int, hop_length: int, n_arrows=N_ARROWS,
                  save_name="aligned", save_dir: Optional[str] = "aligned"):
    if save_dir is None:
        save_dir = "aligned"
    aligned = sinc_creciente(y_audio, wp_s, fs, len(x_audio), n_arrows=n_arrows)
    save_comparative_plot(x_audio, aligned, fs, save_name, save_dir)
    save_audio(aligned, save_name, save_dir, fs)
    return aligned


def obtener_audio_de_midi(midi_file_path: str, midi_name, verbose: Optional[bool] = False):
    from mutations.midi_utils import load_midi_with_pretty_midi, load_midi_with_mido, save_excerpt_in_audio, \
        extract_tempo_from_midi
    try:
        original_excerpt = load_midi_with_pretty_midi(midi_file_path)
        if verbose: print("✅ Archivo MIDI cargado exitosamente con pretty_midi")
    except Exception as e:
        if verbose: print(f"⚠️ Error con pretty_midi: {e}")
        try:
            original_excerpt = load_midi_with_mido(midi_file_path)
            if verbose: print("✅ Archivo MIDI cargado exitosamente con mido")
        except Exception as e2:
            print(f"❌ Error cargando MIDI: {e2}")
            return None

    base_tempo = extract_tempo_from_midi(midi_file_path)
    if verbose: print(f"✅ Tempo detectado: {base_tempo} BPM")

    try:
        reference_audio_path = save_excerpt_in_audio(
            dir_name=midi_name,
            excerpt=original_excerpt,
            save_name=f"{midi_name}"
        )
        if verbose: print(f"✅ Audio de referencia guardado: {reference_audio_path}")
    except Exception as e:
        print(f"❌ Error generando audio de referencia: {e}")
        return None

    return original_excerpt, base_tempo, reference_audio_path


def get_reference_audio_duration(midi_name: str, output_dir: str) -> float:
    """
    Obtiene la duración en segundos del audio de referencia.
    
    Args:
        midi_name: Nombre del archivo MIDI (sin extensión)
        output_dir: Directorio base de resultados
        
    Returns:
        Duración en segundos del audio, o 0.0 si no se puede obtener
    """
    try:
        # Buscar el archivo de audio de referencia en diferentes ubicaciones posibles
        audio_path = Path("mutts") / f"{midi_name}" / "audios" / f"{midi_name}.wav"

        if audio_path.exists():
            duration = librosa.get_duration(path=str(audio_path))
            return round(duration, 4)

    except Exception as e:
        print(f"⚠️ Error obteniendo duración para {midi_name}: {e}")
        return 0.0


def ejemplo():
    hop_length = 1024
    x_audio, fs = librosa.load('mutts/audios/Acordai-100_reference.wav')
    y_audio, fs = librosa.load('mutts/audios/Acordai-100_faster_tempo.wav')
    fs = int(fs)  # Asegurarse de que fs sea un entero
    D, wp, wp_s = calculate_warping_path(x_audio, y_audio, fs, hop_length)
    print(len(x_audio), len(y_audio), len(wp), len(wp_s))
    n_arrows = 50
    aligned = sinc_creciente(y_audio, wp_s, fs, len(x_audio), n_arrows=n_arrows)
    save_name = 'aligned_audios'
    save_dir = 'z/aligned'
    save_comparative_plot(x_audio, aligned, fs, save_name, save_dir)
    save_audio(aligned, save_name, save_dir, fs)


if __name__ == "__main__":
    ejemplo()
    print("Ejemplo ejecutado correctamente.")
