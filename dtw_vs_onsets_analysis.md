# Análisis detallado de DTW vs Onsets: Por qué pueden dar resultados aparentemente contradictorios.

DIFERENCIAS FUNDAMENTALES:

1. DTW (Dynamic Time Warping):
   - Analiza características espectrales (MFCC) frame por frame
   - Busca similitud de contenido armónico/timbral
   - Es tolerante a diferencias de timing si el contenido musical es similar
   - Suaviza diferencias locales de tiempo
   - Ventana temporal: hop_length (normalmente 512 samples ≈ 23ms)

2. Detección de Onsets:
   - Detecta eventos de inicio de nota específicos
   - Sensible a cambios abruptos de energía
   - Precisión temporal muy alta (milisegundos)
   - No considera contenido armónico, solo timing de eventos

## CASOS DONDE PUEDEN DIVERGIR:

A) Micro-desplazamientos temporales:
   - Onsets: Detecta que una nota está 50ms tarde
   - DTW: Ve que el contenido musical general está alineado

B) Notas desplazadas pero contenido preservado:
   - Una sección completa está ligeramente adelantada/atrasada
   - DTW puede alinear globalmente y mostrar poca desviación
   - Onsets individuales pueden estar significativamente desplazados

C) Diferencias en resolución temporal:
   - DTW trabaja con frames de ~23ms
   - Onsets pueden detectar diferencias de ~5-10ms

## EJEMPLO PRÁCTICO:
Si tienes una secuencia musical donde:
- El contenido armónico es correcto (mismas notas, mismas alturas)
- Pero todas las notas están 30ms tarde de forma consistente

Resultado:
- DTW: "Camino regular, diferencias mínimas" (porque mapea correctamente el contenido)
- Onsets: "Múltiples notas atrasadas" (porque detecta el desplazamiento temporal)
