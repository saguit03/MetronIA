from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pyrubberband.pyrb as pyrb
import scipy.io.wavfile as wavfile
from pathlib import Path

hop_length = 1024

def load_audio_files(reference_path: str, live_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Carga archivos de audio."""
    reference_audio, sr = librosa.load(reference_path)
    live_audio, _ = librosa.load(live_path, sr=sr)  # Usar mismo sr
    
    return reference_audio, live_audio, sr

def calculate_warping_path(reference_audio: np.ndarray, live_audio: np.ndarray, fs: int, hop_length: int):
    """Extrae las características de cromograma de dos audios."""
    reference_chroma = librosa.feature.chroma_cqt(y=reference_audio, sr=fs, hop_length=hop_length)
    live_chroma = librosa.feature.chroma_cqt(y=live_audio, sr=fs, hop_length=hop_length)
    D, wp = librosa.sequence.dtw(X=reference_chroma, Y=live_chroma, metric='cosine')
    wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)
    return wp, wp_s


def sinc(x, wp_s, sample_rate, out_len, n_arrows):
  time_map = [(int(x*sample_rate), int(y*sample_rate)) for (x, y) in wp_s[::len(wp_s)//n_arrows]]
  time_map.append((len(x), out_len))
  return pyrb.timemap_stretch(x, sample_rate, time_map)

def sinc_creciente(x, wp_s, sample_rate, out_len, n_arrows):
    # Convertimos wp_s a muestras
    raw_map = [(int(t1 * sample_rate), int(t2 * sample_rate)) for t1, t2 in wp_s//n_arrows]

    # Ordenamos por tiempo de entrada
    raw_map.sort(key=lambda pair: pair[0])

    time_map = []
    last_in, last_out = -1, -1
    for t_in, t_out in raw_map:
        # Aseguramos que ambos tiempos crezcan estrictamente
        if t_in > last_in and t_out > last_out:
            time_map.append((t_in, t_out))
            last_in, last_out = t_in, t_out

    final_point = (len(x), out_len)
    if time_map and time_map[-1][0] >= final_point[0]:
        # Reemplazar último punto si ya pasó el final
        time_map[-1] = final_point
    else:
        time_map.append(final_point)

    for i in range(1, len(time_map)):
        if time_map[i][0] <= time_map[i-1][0] or time_map[i][1] <= time_map[i-1][1]:
            print(f"Error at {i}: {time_map[i-1]} -> {time_map[i]}")
    return pyrb.timemap_stretch(x, sample_rate, time_map)


def save_comparative_plot(x_audio: np.ndarray, y_audio: np.ndarray, fs: int, wp_s: np.ndarray, save_name, save_dir):
    # Use default directory if save_dir is None
    if save_dir is None:
        save_dir = "aligned"
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8,4))
    librosa.display.waveshow(x_audio, sr=fs, ax=ax2)
    ax2.set(title='Referencia $x_audio$')
    librosa.display.waveshow(y_audio, sr=fs, ax=ax1)
    ax1.set(title='Vivo $y_audio$')
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_filename = Path(save_dir) / f"{save_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_audio(audio, save_name, save_dir, sample_rate):
    """Guarda el audio en un fichero WAV."""
    # Use default directory if save_dir is None
    if save_dir is None:
        save_dir = "aligned"
    
    audio_normalized = np.int16(audio / np.max(np.abs(audio)) * 32767)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_filename = Path(save_dir) / f"{save_name}.wav"
    wavfile.write(output_filename, sample_rate, audio_normalized)
    return output_filename

def stretch_audio(x_audio: np.ndarray, y_audio: np.ndarray, fs: int, hop_length: int, n_arrows = 50, save_name = "aligned", save_dir: Optional[str] = "aligned"):
    # Use default directory if save_dir is None
    if save_dir is None:
        save_dir = "aligned"
    
    wp, wp_s = calculate_warping_path(x_audio, y_audio, fs, hop_length)
    aligned = sinc_creciente(y_audio, wp_s, fs, len(x_audio), n_arrows=n_arrows)
    save_comparative_plot(x_audio, aligned, fs, wp_s, save_name, save_dir)
    save_audio(y_audio, save_name, save_dir, fs)
    return aligned, wp, wp_s

def ejemplo():
    x_audio, fs = librosa.load('mutts/audios/Acordai-100_reference.wav')
    y_audio, fs = librosa.load('mutts/audios/Acordai-100_faster_tempo.wav')
    wp, wp_s = calculate_warping_path(x_audio, y_audio, fs, hop_length)
    print(len(x_audio), len(y_audio), len(wp), len(wp_s))
    n_arrows = 50
    aligned = sinc_creciente(y_audio, wp_s, fs, len(x_audio), n_arrows=n_arrows)
    save_name = 'aligned_audios'
    save_dir = 'z/aligned'
    save_comparative_plot(x_audio, aligned, fs, wp_s, save_name, save_dir)
    save_audio(y_audio, save_name, save_dir, fs)

if __name__ == "__main__":
    ejemplo()
    print("Ejemplo ejecutado correctamente.")