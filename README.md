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
