# Funcionalidad de Proporción de Tempo

## Descripción

Se ha implementado una nueva funcionalidad que calcula la proporción de tempo entre los audios de referencia y en vivo (`live_tempo/reference_tempo`) y re-samplea automáticamente el audio de referencia cuando es necesario para mejorar el alineamiento DTW.

## Funcionamiento

### 1. Cálculo de Proporción de Tempo

```python
tempo_proportion = actual_live_tempo / actual_ref_tempo
```

- Si `tempo_proportion > 1.0`: El audio en vivo es más rápido que la referencia
- Si `tempo_proportion < 1.0`: El audio en vivo es más lento que la referencia
- Si `tempo_proportion ≈ 1.0`: Los tempos son similares

### 2. Re-sampling Automático

El sistema aplica re-sampling del audio de referencia cuando:
- `tempo_proportion < 0.95` (audio en vivo más de 5% más lento)
- `tempo_proportion > 1.05` (audio en vivo más de 5% más rápido)

### 3. Mejora del Alineamiento DTW

Cuando se aplica re-sampling:
- El audio de referencia se ajusta para coincidir con el tempo del audio en vivo
- Esto mejora significativamente la calidad del alineamiento DTW
- Los análisis de onsets y beat spectrum se realizan con mayor precisión

## Implementación Técnica

### En `MusicAnalyzer.calculate_tempo_proportion_and_resample()`

```python
def calculate_tempo_proportion_and_resample(self, audio_ref, audio_live, sr, reference_tempo=None):
    # 1. Detectar tempos
    # 2. Calcular proporción
    # 3. Aplicar re-sampling si es necesario usando librosa.effects.time_stretch()
    # 4. Retornar audio ajustado y metadatos
```

### Modificaciones en `comprehensive_analysis()`

- Se llama a `calculate_tempo_proportion_and_resample()` antes del análisis DTW
- Todos los análisis posteriores usan `audio_ref_resampled` en lugar de `audio_ref`
- Se preserva el `audio_ref` original para referencia

## Nuevos Campos en Resultados

### TempoAnalysisResult
- `tempo_proportion`: Proporción calculada
- `original_ref_tempo`: Tempo original del audio de referencia
- `original_live_tempo`: Tempo original del audio en vivo  
- `resampling_applied`: Boolean indicando si se aplicó re-sampling

### CSV de Resultados
- `tempo_proportion`: Proporción de tempo
- `original_ref_tempo_bpm`: Tempo original de referencia
- `original_live_tempo_bpm`: Tempo original en vivo
- `resampling_applied`: Si se aplicó re-sampling

### Reporte de Resumen
- Estadísticas de proporción de tempo (promedio, mínimo, máximo)
- Conteo de análisis con re-sampling aplicado

## Beneficios

1. **Mejor Alineamiento DTW**: Los audios con tempos similares se alinean más precisamente
2. **Análisis Más Precisos**: Los análisis de onsets y beat spectrum son más confiables
3. **Transparencia**: Se registra claramente cuándo y por qué se aplica re-sampling
4. **Preservación de Datos**: Se mantienen tanto los datos originales como los procesados

## Ejemplo de Uso

```python
analyzer = MusicAnalyzer()
result = analyzer.comprehensive_analysis(
    reference_path="reference.wav",
    live_path="live.wav", 
    reference_tempo=120.0  # BPM conocido del MIDI
)

# Acceder a información de tempo
print(f"Proporción: {result['tempo_proportion']:.3f}")
print(f"Re-sampling aplicado: {result['resampling_applied']}")
```

## Salida en Consola

```
🎵 Tempos detectados:
   Referencia: 120.00 BPM
   En vivo: 135.50 BPM
   Proporción (live/ref): 1.129
⚡ Aplicando re-sampling del audio de referencia (factor: 1.129)
   Duración original ref: 30.25s
   Duración re-sampled ref: 26.80s
   Duración live: 26.75s
```
