"""
DED–D2 (ProbLog backend):
- Deducción Horn con variables (sin negación por ahora).
- Motor de inferencia: ProbLog (no reinventamos unificación/SLD/backtracking).

Qué resuelve esta versión:
- Entailment booleano (True/False) para consultas ground (sin variables en la query).
- Generación de tareas controladas por profundidad y fan-out (ID/OOD).
- Métricas de costo: tiempo de evaluación + tamaños de KB.

Referencias de implementación:
- API Python ProbLog: PrologString + get_evaluatable().create_from(...).evaluate()
  https://problog.readthedocs.io/en/latest/python.html
  https://dtai.cs.kuleuven.be/problog/tutorial/advanced/01_python_interface.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import random
import time

try:
    from problog.program import PrologString
    from problog import get_evaluatable
except Exception as _e:  # pragma: no cover
    PrologString = None
    get_evaluatable = None


# -----------------------------
# Tipos y utilidades
# -----------------------------

@dataclass(frozen=True, slots=True)
class Ded2Task:
    program: str
    query: str           # por ejemplo: "path(c0,c6)"
    label: bool
    difficulty: str      # "easy" | "mid" | "hard"
    split: str           # "id" | "ood"
    meta: Dict[str, int] # tamaños, profundidad, distractores, constantes


def _require_problog() -> None:
    if PrologString is None or get_evaluatable is None:
        raise RuntimeError(
            "ProbLog no está disponible. Instala con: pip install problog"
        )


def _mk_const(i: int) -> str:
    return f"c{i}"


def _edge(a: str, b: str) -> str:
    return f"edge({a},{b})."


def _fact(pred: str, args: Sequence[str]) -> str:
    inner = ",".join(args)
    return f"{pred}({inner})."


def _query(atom: str) -> str:
    return f"query({atom})."


def _ded_rules_transitive_closure() -> List[str]:
    # path(X,Y) :- edge(X,Y).
    # path(X,Z) :- edge(X,Y), path(Y,Z).
    return [
        "path(X,Y) :- edge(X,Y).",
        "path(X,Z) :- edge(X,Y), path(Y,Z).",
    ]


def problog_entails_many(program_without_queries: str, queries: Sequence[str]) -> Tuple[Dict[str, bool], float]:
    """
    Evalúa múltiples queries en una sola compilación/evaluación (más eficiente que compilar por query).

    Devuelve:
      - dict { "path(c0,c4)" : True/False }
      - wall_seconds de la evaluación total
    """
    _require_problog()

    # Inyectamos todas las queries dentro del programa
    q_lines = "\n".join(_query(q) for q in queries)
    full_program = program_without_queries.rstrip() + "\n" + q_lines + "\n"

    t0 = time.perf_counter()
    result = get_evaluatable().create_from(PrologString(full_program)).evaluate()
    dt = time.perf_counter() - t0

    # result: dict(Term -> prob). Mapeamos por string estable.
    prob_by_str = {str(term): float(prob) for term, prob in result.items()}
    out: Dict[str, bool] = {}
    for q in queries:
        # ProbLog devuelve key == str(atom) (sin "query(...)")
        p = prob_by_str.get(q, 0.0)
        out[q] = (p >= 1.0 - 1e-12)

    return out, dt


# -----------------------------
# Generación de KBs y tareas D2
# -----------------------------

def _gen_dag_edges(
    rng: random.Random,
    nodes: Sequence[str],
    *,
    chain_depth: int,
    extra_edges: int,
    forbid_cross_to: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Genera edges acíclicos (i < j) para evitar ciclos:
    - Cadena base de longitud chain_depth usando los primeros chain_depth+1 nodos.
    - extra_edges adicionales para fan-out controlado (también acíclicos).
    - forbid_cross_to: si se pasa, evita edges hacia ese conjunto (útil para asegurar negativos).
    """
    if chain_depth < 1:
        raise ValueError("chain_depth debe ser >= 1")

    edges: List[Tuple[str, str]] = []
    n_chain = chain_depth + 1
    if n_chain > len(nodes):
        raise ValueError("No hay suficientes nodos para la cadena")

    chain_nodes = list(nodes[:n_chain])
    # cadena: c0->c1->...->cD
    for i in range(chain_depth):
        edges.append((chain_nodes[i], chain_nodes[i + 1]))

    forbid_set = set(forbid_cross_to) if forbid_cross_to else set()
    used = set(edges)

    # Extra edges: i<j (DAG), solo dentro del prefijo chain_nodes por defecto
    # para subir fan-out sin tocar la componente negativa.
    tries = 0
    while len(used) < len(edges) + extra_edges and tries < 200_000:
        tries += 1
        i = rng.randrange(0, n_chain - 1)
        j = rng.randrange(i + 1, n_chain)
        a, b = chain_nodes[i], chain_nodes[j]
        if b in forbid_set:
            continue
        if (a, b) not in used:
            used.add((a, b))

    return [_edge(a, b) for a, b in used]


def _build_program(
    *,
    edges: Sequence[str],
    extra_noise_facts: Sequence[str],
    rules: Sequence[str],
) -> str:
    blocks = []
    blocks.extend(edges)
    blocks.extend(extra_noise_facts)
    blocks.extend(rules)
    return "\n".join(blocks) + "\n"


def generate_ded_d2_tasks(
    *,
    seed: int,
    n_kb: int,
    queries_per_kb: int,
    split: str,        # "id" | "ood"
    difficulty: str,   # "easy" | "mid" | "hard"
) -> List[Ded2Task]:
    """
    DED–D2: path/2 sobre edge/2 con reglas recursivas (transitive closure).
    - Variables y unificación aparecen en las reglas.
    - Las queries son ground: path(c0,cK)

    ID vs OOD:
    - ID: profundidad y fan-out moderados
    - OOD: mayor profundidad y fan-out + más ruido

    Importante: NO usamos negación aún (eso sería D2.1 con estratificación).
    """
    _require_problog()

    if split not in {"id", "ood"}:
        raise ValueError("split debe ser 'id' o 'ood'")
    if difficulty not in {"easy", "mid", "hard"}:
        raise ValueError("difficulty debe ser 'easy', 'mid' o 'hard'")

    rng = random.Random(seed)
    rules = _ded_rules_transitive_closure()

    # Parámetros base (ID)
    if difficulty == "easy":
        depth_id = (2, 3)
        extra_edges_id = (0, 3)
        noise_facts_id = (0, 3)
        n_consts = 12
    elif difficulty == "mid":
        depth_id = (4, 6)
        extra_edges_id = (3, 12)
        noise_facts_id = (3, 10)
        n_consts = 18
    else:
        depth_id = (7, 10)
        extra_edges_id = (12, 35)
        noise_facts_id = (10, 25)
        n_consts = 26

    # OOD amplifica
    if split == "ood":
        depth = (depth_id[1] + 3, depth_id[1] + 6)
        extra_edges = (extra_edges_id[1] + 10, extra_edges_id[1] + 40)
        noise_facts = (noise_facts_id[1] + 10, noise_facts_id[1] + 40)
        n_consts = n_consts + 10
    else:
        depth = depth_id
        extra_edges = extra_edges_id
        noise_facts = noise_facts_id

    preds_noise = ["foo", "bar", "baz", "rel", "tag"]  # ruido sintáctico, no afecta path/edge

    tasks: List[Ded2Task] = []

    for _kb in range(n_kb):
        consts = [_mk_const(i) for i in range(n_consts)]
        d = rng.randint(depth[0], depth[1])

        # Reservamos un bloque de constantes "negativas" que NO conectamos desde la componente del root
        # para que haya negativas reales sin depender de suerte.
        # root_component = consts[0 : d+1]
        # neg_pool = consts[d+1 : ]  (no recibe edges desde root_component)
        root_component = consts[: d + 1]
        neg_pool = consts[d + 1 :]

        xedges = rng.randint(extra_edges[0], extra_edges[1])
        nnoise = rng.randint(noise_facts[0], noise_facts[1])

        edges = _gen_dag_edges(
            rng,
            root_component + neg_pool,
            chain_depth=d,
            extra_edges=xedges,
            forbid_cross_to=neg_pool,  # asegura que no haya edges hacia el pool negativo
        )

        # ruido: hechos de predicados irrelevantes con constantes al azar
        noise_lines: List[str] = []
        for _ in range(nnoise):
            pred = rng.choice(preds_noise)
            arity = rng.choice([1, 2])
            args = [rng.choice(consts) for _ in range(arity)]
            noise_lines.append(_fact(pred, args))

        program_wo_queries = _build_program(edges=edges, extra_noise_facts=noise_lines, rules=rules)

        # Construir queries: alternamos positivas y negativas
        # Positiva garantizada: path(c0, c_d) usando la cadena base
        pos_target = root_component[-1]
        pos_q = f"path({root_component[0]},{pos_target})"

        queries: List[str] = []
        for i in range(queries_per_kb):
            if i % 2 == 0:
                queries.append(pos_q)
            else:
                # negativa: destino en neg_pool
                if neg_pool:
                    neg_target = rng.choice(neg_pool)
                else:
                    # fallback: si no hay neg_pool (pasa en tamaños raros), elegimos un destino fuera de alcance por construcción
                    neg_target = _mk_const(n_consts + 999)
                queries.append(f"path({root_component[0]},{neg_target})")

        labels, _dt = problog_entails_many(program_wo_queries, queries)

        for q in queries:
            tasks.append(
                Ded2Task(
                    program=program_wo_queries,
                    query=q,
                    label=bool(labels.get(q, False)),
                    difficulty=difficulty,
                    split=split,
                    meta={
                        "depth": d,
                        "n_consts": n_consts,
                        "n_edges": program_wo_queries.count("edge("),
                        "n_rules": len(rules),
                        "n_noise": len(noise_lines),
                    },
                )
            )

    return tasks


# -----------------------------
# Evaluación y export
# -----------------------------

def evaluate_tasks(tasks: Sequence[Ded2Task]) -> dict:
    """
    Evalúa reusando ProbLog como oráculo (consistencia):
    - accuracy esperada ~1.0 (es el mismo motor),
      útil como sanity-check y para medir costo promedio real.
    """
    _require_problog()

    t0 = time.perf_counter()
    n = len(tasks)
    correct = 0
    positives = 0
    wall_eval = 0.0

    # Agrupar por programa para evaluar múltiples queries por KB (eficiente)
    by_prog: Dict[str, List[Ded2Task]] = {}
    for t in tasks:
        by_prog.setdefault(t.program, []).append(t)

    for prog, group in by_prog.items():
        qs = [g.query for g in group]
        labels, dt = problog_entails_many(prog, qs)
        wall_eval += dt

        for g in group:
            pred = labels.get(g.query, False)
            if pred == g.label:
                correct += 1
            if g.label:
                positives += 1

    dt_total = time.perf_counter() - t0
    return {
        "n": n,
        "accuracy": (correct / n) if n else 0.0,
        "positive_rate": (positives / n) if n else 0.0,
        "wall_seconds_total": dt_total,
        "wall_seconds_problog_eval": wall_eval,
        "avg_eval_seconds_per_task": (wall_eval / n) if n else 0.0,
        "kbs": len(by_prog),
    }


def export_jsonl(tasks: Sequence[Ded2Task], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            obj = {
                "program": t.program,
                "query": t.query,
                "label": bool(t.label),
                "difficulty": t.difficulty,
                "split": t.split,
                "meta": t.meta,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
