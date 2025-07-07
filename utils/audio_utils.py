import librosa
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pyrubberband.pyrb as pyrb
import scipy.io.wavfile as wavfile

N_SYNC_POINTS = 5  # Número de puntos de sincronización para la alineación


def check_extension(file_path: str, midi_name) -> str:
    if Path(file_path).suffix.lower() == '.mid':
        result = obtener_audio_de_midi(file_path, midi_name)
        if result is None:
            raise ValueError(f"Error obtaining audio from MIDI file: {file_path}")
        reference_audio, tempo, audio_path = result
    else:
        audio_path = file_path
    return str(audio_path)


def load_audio(path: str, sr: int = None):
    path = Path(path)
    if path.suffix == '.mp3':
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(path), "-ar", str(sr or 22050), tmp_wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode != 0:
            raise RuntimeError(f"❌ Falló la conversión de {path} a WAV con ffmpeg.")
        audio, sr_loaded = librosa.load(tmp_wav_path, sr=sr)
        Path(tmp_wav_path).unlink(missing_ok=True)
        return audio, sr_loaded

    return librosa.load(str(path), sr=sr)


def load_audio_files(reference_path: str, live_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    reference_audio, sr = load_audio(reference_path)
    sr = int(sr)
    live_audio, _ = load_audio(live_path, sr=sr)
    return reference_audio, live_audio, sr


def calculate_warping_path(reference_audio: np.ndarray, live_audio: np.ndarray, sr: int, hop_length: int):
    reference_chroma = librosa.feature.chroma_cqt(y=reference_audio, sr=sr, hop_length=hop_length)
    live_chroma = librosa.feature.chroma_cqt(y=live_audio, sr=sr, hop_length=hop_length)
    D, wp = librosa.sequence.dtw(X=reference_chroma, Y=live_chroma, metric='cosine')
    wp_s = librosa.frames_to_time(wp, sr=sr, hop_length=hop_length)
    return D, wp, wp_s


def sinc(x, wp_s, sample_rate, out_len, n_sync_points):
    time_map = [(int(x * sample_rate), int(y * sample_rate)) for (x, y) in wp_s[::len(wp_s) // n_sync_points]]
    time_map.append((len(x), out_len))
    return pyrb.timemap_stretch(x, sample_rate, time_map)


def sinc_creciente(x, wp_s, sample_rate, out_len, n_sync_points):
    raw_map = [(int(t2 * sample_rate), int(t1 * sample_rate)) for t1, t2 in wp_s[::n_sync_points]]

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

def sinc_2_points(x, sample_rate, out_len):
    raw_map = [(0,0), (len(x), out_len)]
    return pyrb.timemap_stretch(x, sample_rate, raw_map)


def save_comparative_plot(reference_audio: np.ndarray, live_audio: np.ndarray, sr: int, save_name, save_dir):
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 4))
    librosa.display.waveshow(reference_audio, sr=sr, ax=ax2)
    ax2.set(title='Referencia')
    librosa.display.waveshow(live_audio, sr=sr, ax=ax1)
    ax1.set(title='Alineado')
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_filename = Path(save_dir) / f"{save_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_audio(audio, save_name, save_dir, sample_rate):
    audio_normalized = np.int16(audio / np.max(np.abs(audio)) * 32767)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_filename = Path(save_dir) / f"{save_name}.wav"
    wavfile.write(output_filename, sample_rate, audio_normalized)
    return output_filename


def stretch_audio(reference_audio: np.ndarray, live_audio: np.ndarray, sr: int, hop_length: int,
                  n_sync_points=N_SYNC_POINTS,
                  save_name="aligned", save_dir: Optional[str] = "aligned"):
    #distance, wp, wp_s = calculate_warping_path(reference_audio, live_audio, sr, hop_length)
    #aligned = sinc_creciente(live_audio, wp_s, sr, len(reference_audio), n_sync_points=n_sync_points)
    aligned = sinc_2_points(live_audio, sr, len(reference_audio))
    save_audio(aligned, save_name, save_dir, sr)
    return aligned


def obtener_audio_de_midi(midi_file_path: str, midi_name, verbose: Optional[bool] = False, cut_excerpt: Optional[bool] = False) -> Optional[Tuple]:
    from utils.midi_utils import load_midi_with_pretty_midi, load_midi_with_mido, save_excerpt_in_audio, extract_tempo_from_midi
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

    if cut_excerpt:original_excerpt = original_excerpt[:100]

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


def ejemplo():
    hop_length = 1024
    reference_audio, sr = librosa.load('mutts/audios/Acordai-100_reference.wav')
    live_audio, sr = librosa.load('mutts/audios/Acordai-100_faster_tempo.wav')
    sr = int(sr)
    D, wp, wp_s = calculate_warping_path(reference_audio, live_audio, sr, hop_length)
    print(len(reference_audio), len(live_audio), len(wp), len(wp_s))
    n_sync_points = 50
    aligned = sinc_creciente(live_audio, wp_s, sr, len(reference_audio), n_sync_points=n_sync_points)
    save_name = 'aligned_audios'
    save_dir = 'z/aligned'
    save_comparative_plot(reference_audio, aligned, sr, save_name, save_dir)
    save_audio(aligned, save_name, save_dir, sr)


if __name__ == "__main__":
    ejemplo()
    print("Ejemplo ejecutado correctamente.")
