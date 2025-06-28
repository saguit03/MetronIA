# MetronIA ― Sistema de Análisis de Sincronía de Ritmos Musicales en Audios

MetronIA se ha desarrollado como Trabajo de Fin de Grado (TFG) en la Universidad de Extremadura, España.  

El objetivo principal de este proyecto es analizar la sincronía de ritmos musicales en ficheros de audio, proporcionando una herramienta útil para estudiantes de música.  

Se divide en dos partes principales:
- Analizador MetronIA, que analiza la sincronía de dos audios.
- Validador con mutantes, que genera variaciones de un MIDI para la validación del analizador.

## Analizador MetronIA (analizador.py)

Uso:
```bash
    python3 analizador.py [nombre_analisis] <ruta_referencia> <live_paths> 
```

Args:
- ruta_referencia: Ruta al archivo de audio de referencia
- ruta_en_vivo: Ruta al archivo de audio en vivo
- nombre_analisis: Nombre para el análisis

Ejemplos:
```bash
    python3 analizador.py mi_analisis audio/reference.wav audio/live.wav
    python3 analizador.py mi_analisis audio/reference.wav audio/live-1.wav audio/live-2.wav audio/live-3.wav
```

## Validador con mutantes (mutar_y_analizar.py)

```bash
# Aplicar múltiples categorías específicas
python mutar_y_analizar.py --categories timing tempo

# Usar un archivo MIDI específico
python mutar_y_analizar.py --midi path/to/your/file.mid

# Analizar una carpeta con varios archivos MIDI
python mutar_y_analizar.py --all_midi path/to/your/midi/directory

# Establecer un límite de ficheros a procesar
python mutar_y_analizar.py --all_midi path/to/your/midi/directory --files_limit 10
```
  
Categorías disponibles:
  - pitch: Errores de altura de las notas
  - tempo: Errores relacionados con el tempo
  - timing: Errores de timing de las notas
  - duration: Errores de duración de las notas
  - note: Errores de presencia de notas
  - articulation: Errores de articulación

