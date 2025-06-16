# Sistema de Validación del Analizador de Mutaciones

## Descripción

El sistema de validación utiliza matrices de confusión para evaluar qué tan efectivamente el analizador detecta los cambios introducidos por las mutaciones. Compara los cambios esperados (registrados en `changes.csv`) con los errores detectados (registrados en `analysis.csv`).

## Conceptos Clave

### Matriz de Confusión

```
                    PREDICCIÓN
                No Error    Error
VERDAD No Error    TN        FP
       Error       FN        TP
```

- **TP (True Positives)**: Errores correctamente detectados
- **FP (False Positives)**: Falsos errores detectados
- **TN (True Negatives)**: Notas correctas correctamente identificadas
- **FN (False Negatives)**: Errores reales no detectados

### Métricas de Evaluación

- **Precisión**: `TP / (TP + FP)` - Proporción de detecciones correctas
- **Recall**: `TP / (TP + FN)` - Proporción de errores reales detectados
- **F1-Score**: Media armónica de precisión y recall
- **Exactitud**: `(TP + TN) / Total` - Proporción total de clasificaciones correctas

## Uso del Sistema

### 1. Validación Automática (Integrada)

La validación se ejecuta automáticamente al final del pipeline principal:

```bash
python mutar_y_analizar.py --midi midi/Tarrega_Gran_Vals.mid
```

Esto generará automáticamente:
- `results/Tarrega_Gran_Vals/validation/validation_report.txt`
- `results/Tarrega_Gran_Vals/validation/validation_results.csv`
- `results/Tarrega_Gran_Vals/validation/confusion_matrix.png`

### 2. Validación Manual (Script Independiente)

Para análisis más detallados o validaciones específicas:

```bash
# Validar todas las mutaciones de un MIDI
python validate_analyzer.py --midi Tarrega_Gran_Vals

# Validar una mutación específica
python validate_analyzer.py --midi Tarrega_Gran_Vals --category timing_errors --mutation note_too_late

# Personalizar directorio de salida
python validate_analyzer.py --midi Acordai-100 --output-dir custom_validation/
```

## Archivos Generados

### 1. `validation_report.txt`
Reporte detallado con:
- Métricas globales
- Matriz de confusión numérica
- Rendimiento por categoría de mutación
- Resultados individuales de cada mutación
- Interpretación automática y recomendaciones

### 2. `validation_results.csv`
Datos estructurados con columnas:
- `midi_name`, `mutation_category`, `mutation_name`
- `true_positives`, `false_positives`, `true_negatives`, `false_negatives`
- `precision`, `recall`, `f1_score`, `accuracy`
- `expected_changes`, `detected_changes`, `total_notes`

### 3. `confusion_matrix.png`
Visualización gráfica de la matriz de confusión con métricas resumidas.

## Interpretación de Resultados

### Valores Ideales
- **Precisión > 0.8**: Pocos falsos positivos
- **Recall > 0.8**: Pocos falsos negativos
- **F1-Score > 0.8**: Excelente balance general

### Problemas Comunes

#### Precisión Baja (< 0.7)
- **Problema**: Muchos falsos positivos
- **Causa**: El analizador detecta errores donde no los hay
- **Solución**: Ajustar tolerancias o umbrales de detección

#### Recall Bajo (< 0.7)
- **Problema**: Muchos falsos negativos
- **Causa**: El analizador pierde errores reales
- **Solución**: Mejorar sensibilidad de detección

#### F1-Score Bajo (< 0.6)
- **Problema**: Rendimiento general deficiente
- **Causa**: Problemas en el algoritmo de análisis
- **Solución**: Revisar algoritmos de DTW y detección de onsets

## Estructura del Análisis

### Comparación de Archivos

1. **`changes.csv`** (Cambios Esperados):
   - Contiene las modificaciones reales aplicadas por la mutación
   - Tipos: `modified`, `added`, `removed`
   - Campos: `start_time`, `pitch`, `change_type`

2. **`analysis.csv`** (Errores Detectados):
   - Contiene los errores detectados por el analizador
   - Tipos: `late`, `early`, `extra`, `missing`
   - Campos: `ref_onset_time`, `live_onset_time`, `onset_type`

### Algoritmo de Comparación

1. **Extracción de Cambios**: Identifica notas que deberían detectarse como errores
2. **Extracción de Errores**: Identifica notas detectadas como errores
3. **Emparejamiento**: Usa tolerancia temporal para emparejar notas
4. **Clasificación**: Determina TP, FP, TN, FN para cada nota
5. **Cálculo de Métricas**: Genera estadísticas de rendimiento

## Validación por Categorías

El sistema analiza el rendimiento por tipo de mutación:

- **`timing_errors`**: Errores de timing (tarde/temprano)
- **`pitch_errors`**: Errores de altura
- **`tempo_errors`**: Errores de tempo
- **`duration_errors`**: Errores de duración
- **`note_errors`**: Notas añadidas/eliminadas
- **`articulation_errors`**: Errores de articulación

## Tolerancias y Configuración

### Tolerancia Temporal
- **Valor por defecto**: 0.1 segundos
- **Propósito**: Considerar que dos notas son "la misma" si están dentro de esta tolerancia
- **Configurable en**: `MutationValidationAnalyzer.__init__()`

### Umbral de Pitch
- **Valor por defecto**: Exacto (mismo número MIDI)
- **Propósito**: Emparejar notas con la misma altura
- **Modificable**: En método `_note_should_be_detected()`

## Ejemplos de Interpretación

### Resultado Excelente
```
Precisión: 0.95, Recall: 0.92, F1: 0.93
Interpretación: El analizador funciona excelentemente
```

### Problema de Falsos Positivos
```
Precisión: 0.60, Recall: 0.90, F1: 0.72
Interpretación: Detecta bien pero con muchos falsos positivos
```

### Problema de Falsos Negativos
```
Precisión: 0.90, Recall: 0.55, F1: 0.68
Interpretación: Muy preciso pero pierde muchos errores reales
```

## Extensiones Futuras

### Validación Temporal Avanzada
- Análizar precisión del timing de detección
- Evaluar desviaciones temporales en los emparejamientos

### Validación por Complejidad
- Agrupar resultados por complejidad de las mutaciones
- Identificar qué tipos de errores son más difíciles de detectar

### Validación Contextual
- Considerar el contexto musical en la evaluación
- Ponderar errores según su importancia musical

## Troubleshooting

### Error: "No se encontró archivo de cambios"
- **Causa**: El archivo `changes.csv` no existe
- **Solución**: Verificar que la mutación se aplicó correctamente

### Error: "No se encontró archivo de análisis"
- **Causa**: El archivo `analysis.csv` no existe
- **Solución**: Verificar que el análisis se ejecutó completamente

### Métricas todas en 0
- **Causa**: No hay emparejamientos entre cambios esperados y detectados
- **Solución**: Revisar tolerancias o formato de los archivos CSV

### Gráfico no se genera
- **Causa**: Problemas con matplotlib/seaborn
- **Solución**: Usar `--no-plots` o instalar dependencias gráficas
