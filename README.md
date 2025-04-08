# Librosa Examples

## How to load audio files

Load the audio file using `librosa.load()`.

## Beat tracking

Example at `plot_beat_tracking.ipynb`. In this example, we demonstrate how to use the `librosa` library to perform beat tracking on an audio file with a time-varying tempo.

### Static tempo

```
tempo, beats_static = librosa.beat.beat_track(y=y, sr=sr, units='time', trim=True)
click_track = librosa.clicks(times=beats_static, sr=sr, click_freq=660, click_duration=0.25, length=len(y))
print(f"Tempo estimate: {tempo[0]:.2f} BPM")
Audio(data=y+click_track, rate=sr)
```

### Dynamic tempo

```
std_bpm = 4
# tempo_dynamic = librosa.feature.tempo(y=y, sr=sr, aggregate=None, std_bpm=std_bpm)
tightness = 10
start_bpm = 125

tempo, beats_dynamic = librosa.beat.beat_track(y=y, sr=sr, units='time',
                                               tightness=tightness,
                                               start_bpm=start_bpm,
                                              #  bpm=tempo_dynamic,
                                               trim=False)

click_dynamic = librosa.clicks(times=beats_dynamic, sr=sr, click_freq=660,
                               click_duration=0.25, length=len(y))

Audio(data=y+click_dynamic, rate=sr)
```


## Music Synchronization with Dynamic Time Warping
Example at `plot_music_sync.ipynb`. In this short tutorial, we demonstrate the use of dynamic time warping (DTW) for music synchronization which is implemented in `librosa`.

### Steps
1. Load the audio files using `librosa.load()`.
2. Extract the chroma features using `librosa.feature.chroma_cqt()`.
3. Align Chroma Sequences:
   1. `librosa.sequence.dtw()`.
   2. `librosa.frames_to_time()`.