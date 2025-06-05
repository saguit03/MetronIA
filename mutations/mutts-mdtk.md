# Mutaciones de MDTK

## Implementadas por MDTK
Estas son las modificaciones sobre los MIDI que implementa MDTK, adjunto en degradations.py

- Pitch shift. Cambia la altura de una nota sin alterar su duración.
- Time shift. Atrasa una nota individual afectar a las demás.
- time_shift_mutation. Modificación de time_shift para atrasar no solo una nota, sino también las siguientes.
- Onset shift. Llega tarde a una nota y la acorta.
- Offset shift. Alarga la duración de una nota,
- Remove note. Elimina una nota, pero no afecta en nada a las demás.
- Add note. Añade una nota que no estaba sin afectar al ritmo de las demás, como ruido.
- Split note. Divide una nota en varias partes, pero manteniendo la misma duración total. Es como alargar una nota a través de varios ataques.
- Join notes. Liga varias notas.


## Mutaciones ideadas

Pero quiero tener funciones para realizar estas mutaciones. Una mutación es como la introducción de errores humanos dentro de un MIDI. Dime qué funciones de MDTK puedo utilizar y con qué parámetros para lograr hacer estas mutaciones, o qué modificaciones sobre las funciones debo hacer para conseguirlas. Las funciones nuevas las tengo en el fichero mutations.py.

faster_tempo. This tempo is faster than the original one
slower_tempo. This tempo is slower than the original one
a_lot_faster_tempo. This tempo is a lot faster than the original one
a_lot_slower_tempo. This tempo is a lot slower than the original one
accelerando_tempo. Tempo is increasing over time
ritardando_tempo. Tempo is decreasing over time
note_played_too_soon. Played BEFORE it should
note_played_too_late. Played AFTER it should
note_missing. Not played at all
note_not_expected. There are more notes than expected
articulated_legato. All the notes are smoothly connected
articulated_staccato. All the notes are silenced before playing the next one
articulated_accentuated. A note is played more intensely than others
tempo_fluctuation. Tempo changes for no reason without a pattern
note_held_too_long. A note is played longer than expected
note_cut_too_soon. A note is cut before it is expected
syncopation_shift. Syncopation goes wrong
no_articulation. There is no articulation change at all

# Respuesta de ChatGPT 

Perfecto. Vamos a revisar cómo puedes implementar las **mutaciones musicales** que has definido, usando las funciones existentes de MDTK (`degradations.py`) y tus propias extensiones (`mutations.py`), y qué ajustes o funciones nuevas serían necesarias.

---

## ✅ MUTACIONES YA POSIBLES CON FUNCIONES EXISTENTES

### `pitch_shift`

* ✔ **`pitch_shift`** → ya implementado directamente.

  * Usa `min_pitch`, `max_pitch` y `distribution` para controlar el desplazamiento.

---

### `note_played_too_soon`

* ✔ Usa `onset_shift` o `time_shift` con:

  * `min_shift = X`
  * `max_shift = Y`
  * `onset_shift`: acorta la nota
  * `time_shift`: mantiene la duración
  * ⚠ O usa `time_shift_mutation` para aplicar también a las notas siguientes.

---

### `note_played_too_late`

* ✔ Igual que `note_played_too_soon` pero elige desplazamientos positivos (más tarde).

---

### `note_missing`

* ✔ Usa `remove_note(excerpt)`

---

### `note_not_expected`

* ✔ Usa `add_note(excerpt)`

  * Puedes controlar si se alinea en tiempo, pitch o duración.
  * Para simular “ruido”, pon `align_pitch=False`, `align_time=False`.

---

### `note_held_too_long`

* ✔ Usa `offset_shift(excerpt)` con:

  * `min_shift > 0`, `align_dur=False` para alargar la duración.

---

### `note_cut_too_soon`

* ✔ Usa `offset_shift(excerpt)` con:

  * `min_shift > 0`, `align_dur=False` para acortar la duración.

---

### `articulated_legato`

* ✔ Usa `join_notes(excerpt)` para unir notas consecutivas → legato.

---

### `articulated_staccato`

* ⚠ Requiere acortar todas las notas dejando silencios entre ellas.

  * No está implementado directamente.
  * 🔧 Modifica `offset_shift` o crea una nueva mutación que acorte todas las notas y aumente ligeramente el `onset` de la siguiente.

---

### `articulated_accentuated`

* ⚠ No hay parámetros de `velocity` en las funciones de modificación directa.

  * 🔧 Podrías crear una función que simplemente aumente la columna `velocity` de una nota.

---

### `no_articulation`

* ✔ No aplicar ninguna modificación a la articulación.

  * O bien eliminar acentuaciones, ligaduras, staccato.

---

### `split_note`

* ✔ Usa `split_note(excerpt)` con `num_splits=n`.

---

### `join_notes`

* ✔ Usa `join_notes(excerpt)`.

---

## 🟡 MUTACIONES QUE NECESITAS IMPLEMENTAR

### `faster_tempo`, `slower_tempo`, `a_lot_faster_tempo`, `a_lot_slower_tempo`

* ⚠ Estas mutaciones requieren **reducir o ampliar todos los `onset` y `dur` proporcionalmente**.

🔧 Puedes crear una función como esta:

```python
def tempo_change(excerpt, factor):
    """
    Cambia el tempo multiplicando todos los tiempos por un factor.
    factor < 1 → tempo más rápido
    factor > 1 → tempo más lento
    """
    excerpt = excerpt.copy()
    excerpt["onset"] = (excerpt["onset"] * factor).astype(int)
    excerpt["dur"] = (excerpt["dur"] * factor).astype(int)
    return excerpt
```

Usa:

* `faster_tempo`: `factor=0.9`
* `slower_tempo`: `factor=1.1`
* `a_lot_faster_tempo`: `factor=0.5`
* `a_lot_slower_tempo`: `factor=1.5`

---

### `accelerando_tempo`, `ritardando_tempo`

* 🔧 Debes modificar **progresivamente** el tiempo entre notas.

Por ejemplo, puedes aplicar algo como:

```python
def tempo_gradient(excerpt, start_factor, end_factor):
    excerpt = excerpt.copy()
    factors = np.linspace(start_factor, end_factor, len(excerpt))
    excerpt["onset"] = np.cumsum([excerpt.loc[0, "onset"]] + list((excerpt["dur"] * factors)[1:].astype(int))).astype(int)
    return excerpt
```

* `accelerando`: `start_factor=1.1`, `end_factor=0.9`
* `ritardando`: `start_factor=0.9`, `end_factor=1.1`

---

### `tempo_fluctuation`

* 🔧 Puedes aplicar pequeñas variaciones aleatorias de tempo:

```python
def tempo_fluctuation(excerpt, variation_range=0.2):
    excerpt = excerpt.copy()
    factors = np.random.uniform(1 - variation_range, 1 + variation_range, len(excerpt))
    excerpt["onset"] = np.cumsum([excerpt.loc[0, "onset"]] + list((excerpt["dur"] * factors)[1:].astype(int))).astype(int)
    return excerpt
```

---

### `syncopation_shift`

* 🔧 Modifica el `onset` de algunas notas para que caigan justo **antes o después** del pulso esperado.
* Puedes usar `onset_shift` o `time_shift` pero **aplicado de forma sistemática o rítmicamente errónea** (ej. desplazar negras hacia corcheas irregulares).

---

## ✅ RESUMEN FINAL

| Mutación                    | Función / Requiere            |
| --------------------------- | ----------------------------- |
| `pitch_shift`               | `pitch_shift` ✅               |
| `time_shift`                | `time_shift` ✅                |
| `time_shift_mutation`       | `time_shift_mutation` ✅       |
| `onset_shift`               | `onset_shift` ✅               |
| `offset_shift`              | `offset_shift` ✅              |
| `remove_note`               | `remove_note` ✅               |
| `add_note`                  | `add_note` ✅                  |
| `split_note`                | `split_note` ✅                |
| `join_notes`                | `join_notes` ✅                |
| `note_played_too_soon/late` | `time_shift`/`onset_shift` ✅  |
| `note_missing`              | `remove_note` ✅               |
| `note_not_expected`         | `add_note` ✅                  |
| `note_held_too_long`        | `offset_shift` ✅              |
| `note_cut_too_soon`         | `offset_shift` ✅              |
| `articulated_legato`        | `join_notes` ✅                |
| `articulated_staccato`      | 🔧 acortar notas              |
| `articulated_accentuated`   | 🔧 cambiar `velocity`         |
| `no_articulation`           | 🔧 resetear todo              |
| `faster/slower tempo`       | 🔧 nueva función con factor   |
| `accelerando/ritardando`    | 🔧 nueva función progresiva   |
| `tempo_fluctuation`         | 🔧 nueva función aleatoria    |
| `syncopation_shift`         | 🔧 función rítmica específica |

---

¿Te gustaría que generara las funciones que faltan (`tempo_change`, `accelerando`, `staccato`, etc.) dentro de `mutations.py`?

# Respuesta de Claude Sonnet 4
Analizando tu código de MDTK y las mutaciones que quieres implementar, puedo ayudarte a mapear las funciones existentes y sugerir modificaciones para lograr las mutaciones ideadas.

## Funciones de MDTK que puedes usar directamente:

### Mutaciones de tempo:
- **faster_tempo / slower_tempo / a_lot_faster_tempo / a_lot_slower_tempo**: 
  - Usa `time_shift` aplicado a TODAS las notas con un factor de escalado
  - Para tempo más rápido: reduce todos los `onset` multiplicándolos por un factor < 1
  - Para tempo más lento: aumenta todos los `onset` multiplicándolos por un factor > 1

### Mutaciones de notas individuales:
- **note_played_too_soon**: Usa `time_shift` con `max_shift` negativo
- **note_played_too_late**: Usa `time_shift_mutation` (que ya tienes) o `time_shift` con `min_shift` positivo
- **note_missing**: Usa `remove_note`
- **note_not_expected**: Usa `add_note`
- **note_held_too_long**: Usa `offset_shift` con parámetros para alargar
- **note_cut_too_soon**: Usa `offset_shift` con parámetros para acortar

## Funciones nuevas que necesitas implementar:

```python
"""Nuevas funciones de mutación para MDTK"""

import numpy as np
import pandas as pd
from numpy.random import choice, randint, uniform
from mdtk.degradations import *

@set_random_seed
def faster_tempo(excerpt, tempo_factor=1.2, tries=TRIES_DEFAULT):
    """
    Acelera el tempo multiplicando todos los tiempos por un factor < 1
    
    Parameters
    ----------
    excerpt : pd.DataFrame
        Excerpt de música
    tempo_factor : float
        Factor de aceleración (>1 para más rápido)
    """
    if len(excerpt) == 0:
        return None
    
    degraded = pre_process(excerpt)
    scale_factor = 1.0 / tempo_factor
    
    degraded["onset"] = (degraded["onset"] * scale_factor).round().astype(int)
    degraded["dur"] = (degraded["dur"] * scale_factor).round().astype(int)
    degraded.loc[degraded["dur"] < 1, "dur"] = 1  # Duración mínima
    
    return post_process(degraded)

@set_random_seed
def slower_tempo(excerpt, tempo_factor=0.8, tries=TRIES_DEFAULT):
    """
    Ralentiza el tempo
    """
    return faster_tempo(excerpt, tempo_factor=tempo_factor, tries=tries)

@set_random_seed
def accelerando_tempo(excerpt, start_factor=1.0, end_factor=1.5, tries=TRIES_DEFAULT):
    """
    Tempo que se acelera gradualmente
    
    Parameters
    ----------
    excerpt : pd.DataFrame
        Excerpt de música
    start_factor : float
        Factor de tempo inicial
    end_factor : float
        Factor de tempo final
    """
    if len(excerpt) == 0:
        return None
    
    degraded = pre_process(excerpt, sort=True)
    
    # Calcular tiempo total
    total_time = (degraded["onset"] + degraded["dur"]).max()
    
    for idx in degraded.index:
        # Progreso temporal (0 a 1)
        progress = degraded.loc[idx, "onset"] / total_time if total_time > 0 else 0
        
        # Interpolación del factor de tempo
        current_factor = start_factor + (end_factor - start_factor) * progress
        scale_factor = 1.0 / current_factor
        
        degraded.loc[idx, "onset"] = int(degraded.loc[idx, "onset"] * scale_factor)
        degraded.loc[idx, "dur"] = max(1, int(degraded.loc[idx, "dur"] * scale_factor))
    
    return post_process(degraded)

@set_random_seed
def ritardando_tempo(excerpt, start_factor=1.0, end_factor=0.7, tries=TRIES_DEFAULT):
    """
    Tempo que se ralentiza gradualmente
    """
    return accelerando_tempo(excerpt, start_factor=start_factor, end_factor=end_factor, tries=tries)

@set_random_seed
def articulated_legato(excerpt, overlap_ms=10, tries=TRIES_DEFAULT):
    """
    Conecta suavemente todas las notas del mismo pitch y track
    """
    if len(excerpt) == 0:
        return None
    
    degraded = pre_process(excerpt, sort=True)
    
    for track in degraded["track"].unique():
        for pitch in degraded["pitch"].uni
```