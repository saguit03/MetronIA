# MetronIA - Musical Performance Analysis System

Sistema de análisis de interpretaciones musicales que utiliza técnicas de procesamiento de audio y aprendizaje automático para comparar grabaciones de referencia con interpretaciones en vivo.

## Estructura del Proyecto

### 📁 `analyzers/`
Módulo modularizado para análisis de audio musical.
- **Análisis de beat spectrum** con alineamiento DTW
- **Detección de onsets** y errores de timing
- **Comparación de tempo** y estructura musical
- **Visualizaciones** especializadas para análisis

### 📁 `audio/`
Archivos de audio de prueba y referencia.
- Grabaciones de referencia y en vivo
- Archivos MIDI y MP3 para testing
- Diferentes tempos y variaciones musicales

### 📁 `mdtk/`
Music Degradation Toolkit - Herramientas para degradación musical.
- **Mutaciones** y degradaciones de audio
- **Utilidades** para procesamiento MIDI
- **Evaluación** de calidad musical
- **Datasets** y modelos de PyTorch

### 📁 `midi/`
Archivos MIDI para análisis y procesamiento.
- Partituras en formato MIDI
- Material musical de referencia

### 📁 `mutations/`
Sistema modular de mutaciones musicales.
- **Generación** de errores musicales controlados
- **Categorización** de tipos de errores
- **Gestión** centralizada de mutaciones
- **Visualización** de comparaciones

### 📁 `mutts/`
Archivos generados por el sistema de mutaciones.
- Audio con diferentes tipos de errores aplicados
- Resultados de mutaciones específicas

### 📁 `plots/`
Gráficos y visualizaciones generadas.
- Comparaciones de beat spectrum
- Análisis de onsets y errores
- Visualizaciones de resultados

### 📁 `references/`
Material de referencia y documentación.

### 📁 `old/`
Versiones anteriores y código legacy.

## Archivos Principales

TODO

## Uso Básico

TODO

## Características
TODO
- ✅ **Análisis DTW** para alineamiento temporal
- ✅ **Detección de onsets** con clasificación de errores
- ✅ **Comparación de tempo** y estructura musical
- ✅ **Sistema de mutaciones** para generación de errores
- ✅ **Visualizaciones** interactivas y exportables
- ✅ **Arquitectura modular** para fácil extensión
