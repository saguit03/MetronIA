# Sistema de validación con mutantes para MetronIA

Aquí se incluyen los métodos y funciones necesarias para la creación y validación de los mutantes de los ficheros MIDI.  

- `Catalog`. Se definen las categorías y los operadores de mutación específicos que pertenecen a cada categoría.
- `Category`. Se define la clase que representa una categoría de mutación.
- `Controller`. Sirve para llamar a las funciones de mutación y degradación de los MIDI.
- `Globals`. Incluye las constantes globales para la creación de mutantes.
- `Logs`. Contiene las clases y métodos necesarios para la creación de logs o registros de las mutaciones.
- `Results`. Definición de la clase que almacena los resultados de las mutaciones.
- `Validator`. Contiene las funciones necesarias para validar los mutantes generados, incluyendo su recuperación de los ficheros generados durante el análisis y el cálculo de las métricas de validación.