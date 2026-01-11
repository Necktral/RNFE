#### Manejo y descripción de errores en DED–D6

El motor y generador DED–D6 pueden lanzar errores en los siguientes casos:

- **ValueError: Regla no segura**
   - Ocurre si alguna regla tiene variables en la cabeza que no aparecen en literales positivos del cuerpo (viola safety).
- **ValueError: Negación no estratificable**
   - Ocurre si hay ciclos negativos o la negación apunta a un predicado del mismo o mayor estrato (viola estratificación).
- **RuntimeError: No se pudo generar una tarea válida**
   - Ocurre si tras muchos intentos no se logra construir una tarea que cumpla todos los invariantes (negación, profundidad, unicidad, etc).
- **Errores de importación o dependencias**
   - Si faltan módulos requeridos o hay problemas de entorno.

**Interpretación y solución:**
- Los errores de safety y estratificación indican problemas en la definición de reglas: revisa que todas las variables estén ancladas y que la negación solo apunte a estratos inferiores.
- El error de generación suele indicar que los parámetros son demasiado restrictivos (por ejemplo, profundidad mínima muy alta o dominio muy pequeño). Prueba aumentando el dominio, relajando restricciones o incrementando el número de intentos.
- Los errores de importación se resuelven asegurando que el entorno esté correctamente configurado y las dependencias instaladas.
### DED–D6: Negación estratificada, safety y explicabilidad exhaustiva
Motor y generador para tareas de razonamiento simbólico con negación por defecto, estratificación, safety y explicabilidad exhaustiva.

**Objetivos cubiertos:**
- Negación por defecto (ausencia como evidencia negativa, no-monótona)
- Estratificación: reglas y predicados organizados en estratos, sin ciclos negativos
- Safety: todas las variables de la cabeza aparecen en literales positivos del cuerpo
- Recursión permitida solo positiva; negación solo sobre estratos inferiores
- Unicidad y determinismo del cierre
- Explicabilidad exhaustiva: para cada hecho deducible, se muestra la prueba; para cada hecho no deducible, se listan todos los caminos fallidos y las razones (incluyendo pasos [NEG] detallados)
- El generador de tareas fuerza que cada tarea tenga razonamiento negativo profundo y opciones únicas

**Garantías DED–D6:**
- Todas las tareas generadas incluyen al menos una deducción con negación y profundidad mínima.
- La traza de prueba (`proof_trace`) incluye pasos `[NEG]` que explican la ausencia de hechos requeridos.
- Distractores garantizados: solo una opción es deducible, las demás no.
- Safety y estratificación validadas en cada tarea.
- Explicabilidad exhaustiva para positivos y negativos.

**Ejemplo de uso:**
```python
from rnfe.pmv.reasoning.ded.ded_d6_stratified_negation import generate_ded_d6_tasks, DED_D6Config
cfg = DED_D6Config(n_tasks=10, seed=42, min_proof_depth=6)
tasks = list(generate_ded_d6_tasks(cfg))
for t in tasks:
   print(t["prompt"])
   print("Prueba:")
   for line in t["proof_trace"]:
      print(line)
```

**Formato de los datasets DED–D6:**
Cada línea contiene:
- `prompt`: descripción de la KB y la pregunta.
- `choices`: opciones en lenguaje natural.
- `choices_symbolic`: opciones simbólicas.
- `answer_index`: índice de la opción correcta.
- `answer_symbolic`: opción correcta simbólica.
- `meta`: metadatos (semilla, profundidad, n_facts, n_rules).
- `proof_trace`: explicación exhaustiva de la deducción (incluye pasos [NEG]).

**Notas:**
- El generador puede requerir muchos intentos para cumplir todos los invariantes (negación, profundidad, unicidad).
- La explicación negativa es exhaustiva: para cada hecho no deducible, se listan todos los caminos fallidos y las razones.
## Cobertura de tests y garantías

El proyecto cuenta con una batería exhaustiva de 71 tests automáticos que validan todos los motores deductivos, módulos de simulación, infraestructura y utilidades. A continuación se detallan los aspectos cubiertos:

### Motores deductivos

- **DED–D1 (Proposicional, Horn clauses):**
   - Saturación y forward chaining determinista.
   - Queries positivas y negativas, trazas de prueba.
   - Manejo de ciclos y casos límite.
   - Garantía de terminación y explicabilidad.

- **DED–D2 (Variables y ProbLog):**
   - Razonamiento con variables y reglas generales.
   - Backend ProbLog: queries de caminos, etiquetas True/False.
   - Consistencia entre programa y etiquetas.

- **DED–D2.1 (Negación estratificada y safety):**
   - Validación de seguridad, ciclos negativos y estratificación.
   - Ejecución de programas con negación, splits ID/OOD/trap.
   - Detección de errores y robustez ante teorías no estratificables.

- **DED–D3 (IDL/Z3):**
   - Entailment en Integer Difference Logic con Z3.
   - Validación de cadenas de restricciones, core de Z3.
   - Generación y validación de datasets, consistencia entre generador y Z3.

- **DED–D4 (CostReach/Horn/Z3):**
   - Alcance de costo mínimo en grafos (DAG) con Z3 Fixedpoint.
   - Validación de caminos, costos, consistencia entre generador y Z3.
   - Datasets balanceados, manifest y diagnóstico formal.

- **DED–D5 (Deducción con profundidad mínima):**
   - Generación de tareas con prueba mínima garantizada.
   - Determinismo, unicidad de respuesta, escritura y lectura de datasets.
   - Validación de invariantes y robustez ante distractores.

### Simulación y mundo artificial

- **MiniWorld Boxes:**
   - Validación de formas, determinismo, avance temporal y observaciones.
   - Pruebas de errores, casos extremos y consistencia de estados.

- **MFM (Memory Fractal Model):**
   - Validación de configuración, escritura/lectura, TTL y ring buffer.
   - Lectura multiresolución, estadísticas de uso, manejo de errores.

### Infraestructura y telemetría

- **TelemetryBus:**
   - Registro y recuperación de métricas escalares y vectoriales.
   - Consistencia de esquema, congelación/descongelación, errores de tipo y dimensión.
   - Exportación a diccionario NumPy, manejo de errores y casos límite.

### Fases y experimentos PMV

- **Phase0 Calibration:**
   - Ejecución completa, registro de métricas, determinismo y consistencia.
   - Validación de dimensiones y series de telemetría.

- **Phase1 Inductive:**
   - Entrenamiento y predicción de modelos lineales.
   - Métricas de error, comparación con baseline, estimación de MDL y costo.
   - Experimentos end-to-end, manejo de errores y casos límite.

- **Phase1 Unimodal:**
   - Normalización de métricas, clipping, cascada de métricas globales.
   - Validación de rangos, casos patológicos y robustez de S_F1.

### Escenarios y proveedores

- **F1 Scenarios:**
   - Generación de secuencias fractales y no fractales.
   - Construcción de datasets, métricas agregadas, diferencias entre modos.
   - Validación de formas, evolución temporal y consistencia de resultados.

- **GenericWorldSequenceProvider:**
   - Proveedor de secuencias de mundo artificial, evolución monótona y embeddings.

### Utilidades y wrappers

- **progress_utils.py:**
   - Wrapper robusto y reutilizable para barras de progreso con tqdm.
   - Pruebas de integración en generadores de datasets y scripts.

---

**Garantías globales:**
- Determinismo y reproducibilidad en todos los motores y simulaciones.
- Robustez ante errores, inputs inválidos y casos límite.
- Consistencia de formatos, metadatos y exportación de datasets.
- Cobertura de splits ID/OOD/trap y validación formal con Z3/ProbLog.
- Explicabilidad y trazabilidad en todos los motores deductivos.
# Proyecto RNFE

## Descripción

Este proyecto implementa una arquitectura de simulación y experimentación para escenarios de mundos artificiales, aprendizaje y razonamiento, con módulos para agentes, control, métricas, experimentos y más. El código está organizado en submódulos bajo `src/rnfe` y cuenta con scripts, tests y documentación adicional.

facts = [Atom("p", ("a",)), Atom("q", ("a",))]
rules = [Rule(head=Atom("r", ("a",)), body=(facts[0], facts[1]))]
ch = ForwardChainer(facts, rules)
ch.saturate()
print(ch.entails(Atom("r", ("a",))))  # True
print(ch.get_proof_tree(Atom("r", ("a",))))  # Trazas de prueba

## Motores deductivos y benchmarks

El proyecto incluye cuatro motores deductivos principales y generadores de datasets para benchmarking de razonamiento simbólico:

### DED–D1: Proposicional (Horn clauses)
Motor forward chaining clásico, determinista y verificable. Genera datasets con splits ID/OOD y queries etiquetadas.

### DED–D2: Variables y unificación (ProbLog)
Extiende DED–D1 con variables, reglas generales y backend ProbLog. Permite tareas más expresivas y queries con variables.

### DED–D2.1: Negación estratificada y safety
Agrega negación estratificada (\+), chequeos de seguridad y detección de ciclos negativos. Genera splits ID, OOD y "trap" (teorías no estratificables).

### DED–D4: Alcance de costo mínimo en grafos (Horn Clauses + Z3)
Motor para tareas de alcance de costo mínimo en grafos dirigidos acíclicos (DAGs) con pesos positivos.
Cada tarea pregunta si existe un camino desde un nodo origen `s` a un nodo destino `t` con costo total menor o igual a un presupuesto `B`.
El etiquetado es exacto y verificable: True si existe tal camino, False si no.
Utiliza Dijkstra para el costo mínimo y Z3 Fixedpoint (muZ, Horn Clauses) para validación lógica y consistencia.
Exporta datasets balanceados IID/OOD y manifiestos con estadísticas.
Incluye explicaciones (camino mínimo) y diagnóstico formal (sat/unsat/unknown) vía Z3.

#### Uso rápido

```bash
PYTHONPATH=src python scripts/run_ded_d4.py --n 7200 --out artifacts/ded_d4.jsonl
```

#### Formato de los datasets DED–D4
Cada línea en el archivo JSONL contiene:
- `edges`: lista de aristas (u, v, w) del grafo DAG.
- `query`: diccionario con `source`, `target`, `budget`.
- `label`: True/False si existe camino s→t con costo ≤ budget.
- `proof_edge_indices`: (opcional) camino mínimo como lista de índices de aristas.
- `shortest_cost`: (opcional) costo mínimo calculado.
- `z3_status`: "sat", "unsat" o "unknown" según Z3.
- `z3_answer`: (opcional) respuesta formal de Z3.
- `difficulty`: "iid" u "ood".
- `meta`: metadatos de generación.

#### Garantías DED–D4
- Etiquetado robusto y verificable (Dijkstra + Z3).
- Diagnóstico formal (sat/unsat/unknown).
- Explicación por construcción (camino mínimo).
- Datasets balanceados y manifest exportado.

#### Componentes principales
- **Motor deductivo**: Deducción de hechos y queries sobre KBs generadas.
- **Generador de tareas**: Control de dificultad, tamaño, ruido y splits (ID/OOD/trap).
- **Evaluador**: Métricas de exactitud, detección de errores y robustez.
- **Exportador**: Datasets en formato JSONL para entrenamiento y evaluación.

#### Uso rápido

**DED–D1:**
```bash
PYTHONPATH=src python scripts/run_deductive_d1.py --n_kb 200 --queries_per_kb 6 --out artifacts/ded_d1.jsonl
```

**DED–D2:**
```bash
PYTHONPATH=src python scripts/run_deductive_d2.py --n_kb 200 --queries_per_kb 6 --out artifacts/ded_d2.jsonl
```

**DED–D2.1:**
```bash
PYTHONPATH=src python scripts/run_deductive_d2_1.py --n_kb 200 --queries_per_kb 6 --trap_n 500 --out artifacts/ded_d2_1.jsonl
```

**Ejecutar tests:**
```bash
PYTHONPATH=src pytest -q
```

#### Formato de los datasets
Cada línea en los archivos JSONL (por ejemplo, artifacts/ded_d2_1.jsonl) es un ejemplo con:
- `program`: KB en formato Prolog/ProbLog.
- `query`: Query ground (ej. path(c0,c7)).
- `label`: Etiqueta booleana (True/False) si la query es válida.
- `difficulty`: "easy", "mid", "hard" o "trap".
- `split`: "id", "ood" o "trap".
- `meta`: Metadatos (profundidad, n_consts, n_edges, n_noise, etc).

Ejemplo:
```json
{
   "program": "edge(c0,c1).\nedge(c1,c2).\npath(X,Y) :- edge(X,Y).\npath(X,Z) :- edge(X,Y), path(Y,Z).\n",
   "query": "path(c0,c2)",
   "label": true,
   "difficulty": "easy",
   "split": "id",
   "meta": {"depth": 2, "n_consts": 5, "n_edges": 2, "n_rules": 2, "n_noise": 0}
}
```

#### Splits
- **id**: In-distribution (estructura y dificultad estándar).
- **ood**: Out-of-distribution (más profundidad, ruido, nodos, queries difíciles).
- **trap**: Teorías no estratificables o con errores de seguridad (solo en DED–D2.1).

#### Ejemplo de API
```python
from rnfe.pmv.reasoning.deductive_d2_1_stratneg import generate_ded_d2_1_tasks, evaluate_d2_1
tasks = generate_ded_d2_1_tasks(seed=42, n_kb=10, queries_per_kb=4, split="id", difficulty="easy")
metrics = evaluate_d2_1(tasks)
print(metrics)
```

#### Criterios de aceptación
- Determinismo y terminación garantizada.
- Etiquetado consistente y robusto.
- Control de dificultad, ruido y splits.
- Tests unitarios completos y verificados.

#### Referencias
- RuleTaker, ProofWriter, IA simbólica clásica.
- ProbLog, razonamiento con negación estratificada.



## Estructura del proyecto

- `src/rnfe/` — Código fuente principal, organizado en:
   - `agents/` — Agentes y estrategias.
   - `core/` — Núcleo: control, geometría, métricas, etc.
   - `fmse/` — Modelos de evolución, semántica y mundo.
   - `infra/` — Infraestructura: logging, almacenamiento, utilidades.
   - `pmv/` — Módulos de experimentos, fases y razonamiento.
   - `pmv/reasoning/` — Motores deductivos y generadores de datasets.
- `scripts/` — Scripts de entrenamiento, pruebas y utilidades.
- `tests/` — Pruebas unitarias y de integración.
- `configs/` — Configuraciones y presets de experimentos.
- `data/` — Datos y logs generados.
- `docs/` — Documentación, diagramas y especificaciones.
- `experiments/` — Resultados y configuraciones de experimentos.


## Instalación

1. Clona el repositorio y entra al directorio del proyecto.
2. Crea y activa un entorno virtual (recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   O usa tu entorno gestionado por pyenv:
   ```bash
   pyenv activate rnfe-lab
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```


## Ejecución de tests

Para ejecutar los tests, asegúrate de tener el entorno virtual activo y ejecuta:
```bash
PYTHONPATH=src pytest -q
```

## Ejemplo de API DED–D4
```python
from rnfe.pmv.reasoning.ded.ded_d4_z3_fixedpoint_horn import generate_ded_d4_tasks, build_manifest
tasks = generate_ded_d4_tasks(num_tasks=100, seed=42)
manifest = build_manifest(tasks)
print(manifest)
```


## Notas
- Si tienes problemas de importación, revisa que `PYTHONPATH` incluya la carpeta `src`.
- Los datasets generados se guardan en `artifacts/` y pueden analizarse con cualquier herramienta que lea JSONL.
- El archivo `requirements.txt` se puede regenerar con `pip freeze > requirements.txt` tras instalar nuevos paquetes.


## Contacto
Para dudas, sugerencias o contribuciones, contacta al responsable del repositorio.
