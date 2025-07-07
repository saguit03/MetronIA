# Analizador MetronIA

Aquí se incluyen los métodos y funciones necesarias para el análisis de los audios:  

- `MetronIA`. Clase principal del analizador MetronIA, que llama a todas las funciones necesarias para el análisis de los audios.
- `Beat Spectrum Analyzer`. Genera el espectro del pulso a partir de las características cromáticas de los audios.
- `Onset DTW Analyzer`. Realiza el análisis de los *onsets* de los audios, utilizando DTW para el emparejamiento de los *onsets*.
- `Onset Results`. Define las clases que almacenan los resultados del análisis de los *onsets*.
- `Tempo Analyzer`. Realiza el análisis del tempo de los audios.