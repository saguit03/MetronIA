# OnsetDTWAnalyzer - Documentación

## Descripción General

La clase `OnsetDTWAnalyzer` implementa un sistema avanzado de emparejamiento de onsets que combina:

1. **Análisis de altura (pitch)** en cada onset
2. **Dynamic Time Warping (DTW)** para alineamiento temporal
3. **Emparejamiento basado en similitud** tanto temporal como de altura

## Características Principales

### 1. Detección de Onsets con Altura
```python
def detect_onsets_with_pitch(self, audio, sr) -> Tuple[np.ndarray, np.ndarray]
```
- Detecta onsets usando `librosa.onset.onset_detect`
- Extrae la altura en cada onset using `librosa.piptrack`
- Retorna tiempos de onset y alturas correspondientes

### 2. Alineamiento DTW
```python
def match_onsets_with_dtw(self, audio_ref, audio_live, sr) -> OnsetDTWAnalysisResult
```
- Crea características combinando tiempo normalizado y altura MIDI
- Implementa DTW personalizado para encontrar el camino óptimo de alineamiento
- Empareja cada onset de referencia con el más similar del audio en vivo

### 3. Cálculo de Ajustes Temporales
Para cada emparejamiento, calcula:
- **time_adjustment**: Diferencia temporal en milisegundos (ref_onset - live_onset)
- **pitch_similarity**: Similitud de altura (0-1, donde 1 = idéntica)

## Estructuras de Datos

### OnsetMatch
```python
class OnsetMatch(NamedTuple):
    ref_onset: float          # Tiempo del onset de referencia
    live_onset: float         # Tiempo del onset en vivo
    ref_pitch: float          # Altura de referencia (Hz)
    live_pitch: float         # Altura en vivo (Hz)
    time_adjustment: float    # Ajuste temporal necesario (ms)
    pitch_similarity: float   # Similitud de altura (0-1)
```

### OnsetDTWAnalysisResult
```python
class OnsetDTWAnalysisResult(NamedTuple):
    matches: List[OnsetMatch]                    # Emparejamientos exitosos
    unmatched_ref: List[Tuple[float, float]]     # Onsets de referencia no emparejados
    unmatched_live: List[Tuple[float, float]]    # Onsets en vivo no emparejados
    dtw_path: np.ndarray                         # Camino DTW
    alignment_cost: float                        # Costo del alineamiento
```

## Algoritmo de Emparejamiento

### Paso 1: Extracción de Características
1. Detecta onsets en ambos audios
2. Extrae pitch en cada onset
3. Normaliza tiempos al rango [0,1]
4. Convierte pitches a escala MIDI
5. Combina en matriz de características [tiempo_norm, pitch_norm]

### Paso 2: Alineamiento DTW
1. Calcula matriz de distancias euclidiana entre características
2. Aplica algoritmo DTW para encontrar camino óptimo
3. El camino DTW determina qué onsets se emparejan

### Paso 3: Análisis de Ajustes
1. Para cada emparejamiento DTW:
   - Calcula ajuste temporal: `(ref_onset - live_onset) * 1000` ms
   - Calcula similitud de altura usando distancia en semitonos
   - Almacena métricas del emparejamiento

## Parámetros Configurables

```python
self.pitch_weight = 0.7      # Peso de similitud de altura en DTW
self.time_weight = 0.3       # Peso de proximidad temporal en DTW
self.max_pitch_diff = 2.0    # Diferencia máxima de semitonos para considerar match
```

## Métodos de Análisis

### get_alignment_adjustments()
Extrae estadísticas de los ajustes temporales:
- Lista de todos los ajustes en ms
- Media y desviación estándar
- Valores máximo y mínimo

### analyze_pitch_accuracy()
Analiza precisión de altura:
- Similitud promedio de altura
- Tasa de emparejamientos "perfectos" (similitud > 0.95)
- Número de emparejamientos perfectos

## Interpretación de Resultados

### Ajustes Temporales Positivos
- **time_adjustment > 0**: El onset en vivo ocurrió ANTES que el de referencia
- **time_adjustment < 0**: El onset en vivo ocurrió DESPUÉS que el de referencia

### Similitud de Altura
- **1.0**: Alturas idénticas
- **0.8-0.99**: Alturas muy similares (diferencia < 0.5 semitonos)
- **0.5-0.79**: Alturas moderadamente similares
- **< 0.5**: Alturas diferentes

## Uso Típico

```python
# Configuración
config = AudioAnalysisConfig()
analyzer = OnsetDTWAnalyzer(config)

# Análisis
result = analyzer.match_onsets_with_dtw(audio_ref, audio_live, sr)

# Revisar emparejamientos
for match in result.matches:
    print(f"Ajuste: {match.time_adjustment:.1f}ms, "
          f"Similitud: {match.pitch_similarity:.3f}")

# Estadísticas generales
adjustments = analyzer.get_alignment_adjustments(result)
pitch_stats = analyzer.analyze_pitch_accuracy(result)
```

## Ventajas sobre Métodos Tradicionales

1. **No usa ventanas fijas**: DTW encuentra automáticamente los mejores emparejamientos
2. **Considera altura musical**: Evita emparejamientos incorrectos entre notas diferentes
3. **Maneja cambios de tempo**: DTW se adapta a variaciones de velocidad
4. **Cuantifica ajustes precisos**: Reporta exactamente qué corrección temporal se necesita
5. **Robusto a desplazamientos globales**: Puede manejar secciones completas desplazadas en tiempo

## Limitaciones

1. **Requiere detección de pitch**: Funciona mejor con audio monofónico o con pitch dominante claro
2. **Computacionalmente intensivo**: DTW es O(n*m) donde n,m son número de onsets
3. **Sensible a errores de detección**: Onsets mal detectados pueden afectar todo el alineamiento
4. **Asume correspondencia 1:1**: Cada onset de referencia se empareja con máximo un onset en vivo
