# Sistema de Validación de MetronIA

Este sistema permite validar la efectividad del analizador de mutaciones usando matrices de confusión.

## Uso Rápido

### 1. Análisis Completo con Validación Automática

```bash
python mutar_y_analizar.py --midi midi/Tarrega_Gran_Vals.mid
```

Esto ejecutará:
1. Generación de mutaciones
2. Análisis de cada mutación
3. **Validación automática** (nuevo)

### 2. Validación Manual de un MIDI

```bash
python validate_analyzer.py --midi Tarrega_Gran_Vals
```

### 3. Validación de una Mutación Específica

```bash
python validate_analyzer.py --midi Tarrega_Gran_Vals --category timing_errors --mutation note_too_late
```

## Resultados Generados

### Estructura de Directorios
```
results/
├── Tarrega_Gran_Vals/
│   ├── mutations_summary.csv
│   ├── summary_report.txt
│   ├── mutation_changes/
│   │   ├── timing_errors_note_too_late/
│   │   │   ├── note_too_late_changes.csv
│   │   │   └── note_too_late_summary.txt
│   └── validation/                           # ← NUEVO
│       ├── confusion_matrix.png              # ← Matriz de confusión visual
│       ├── validation_report.txt             # ← Reporte detallado
│       └── validation_results.csv            # ← Datos estructurados
└── Tarrega_Gran_Vals_timing_errors_note_too_late/
    ├── onset_errors_detailed.png
    ├── analysis.csv
    ├── beat_spectrum.png
    └── timeline.png
```

### Interpretación de Métricas

- **Precisión > 0.8**: 🟢 Pocos falsos positivos
- **Recall > 0.8**: 🟢 Detecta la mayoría de errores reales
- **F1-Score > 0.8**: 🟢 Excelente rendimiento general
- **F1-Score < 0.6**: 🔴 Necesita mejoras

## Flujo de Validación

1. **Compara** `changes.csv` (cambios reales) con `analysis.csv` (errores detectados)
2. **Clasifica** cada nota como TP, FP, TN, o FN
3. **Calcula** métricas de rendimiento
4. **Genera** matriz de confusión visual
5. **Crea** reporte con interpretación automática

## Ejemplos de Interpretación

### ✅ Analizador Funcionando Bien
```
Precisión: 0.92, Recall: 0.88, F1: 0.90
→ El analizador detecta correctamente los errores con pocos falsos positivos
```

### ⚠️ Demasiados Falsos Positivos
```
Precisión: 0.55, Recall: 0.95, F1: 0.70
→ Detecta bien pero marca errores donde no los hay
```

### ⚠️ Pierde Errores Reales
```
Precisión: 0.90, Recall: 0.50, F1: 0.64
→ Muy preciso pero no detecta muchos errores reales
```

## Archivos Clave

- `analyzers/validation_analyzer.py`: Sistema de validación principal
- `validate_analyzer.py`: Script independiente para validación
- `docs/VALIDATION_SYSTEM.md`: Documentación detallada

## Troubleshooting

### "No se encontró archivo de cambios"
- Verificar que las mutaciones se aplicaron correctamente
- El archivo `changes.csv` debe existir en `mutation_changes/`

### "No se encontró archivo de análisis"  
- Verificar que el análisis se completó
- El archivo `analysis.csv` debe existir en el directorio de análisis

### Métricas todas en 0
- Revisar tolerancias en `MutationValidationAnalyzer`
- Verificar formato de los archivos CSV

¡El sistema de validación te ayudará a evaluar y mejorar la precisión del analizador de MetronIA!
