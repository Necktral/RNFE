"""
DED–D1: Motor deductivo proposicional (Horn/definite clauses) + generador de tareas + evaluación.

Base teórica:
- KB como conjunto de cláusulas de Horn (definite clauses).
- Inferencia por encadenamiento hacia adelante (forward chaining / bottom-up).
- Modus Ponens: si (a ∧ b ∧ ... ∧ n) y (a ∧ b ∧ ... ∧ n → h) entonces h.

Diseño:
- "Ground" (sin variables). D2 puede extender a unificación/variables.
- Trazas de prueba: para cada hecho derivado guardamos (regla usada + premisas).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json
import random
import time


# -----------------------------
# Representación: Átomos y Reglas
# -----------------------------

@dataclass(frozen=True, slots=True)
class Atom:
    pred: str
    args: Tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.pred
        inner = ",".join(self.args)
        return f"{self.pred}({inner})"


@dataclass(frozen=True, slots=True)
class Rule:
    head: Atom
    body: Tuple[Atom, ...]  # conjunción de átomos

    def __str__(self) -> str:
        if not self.body:
            return f"{self.head}."
        left = " & ".join(str(a) for a in self.body)
        return f"{left} -> {self.head}."


@dataclass(frozen=True, slots=True)
class ProofStep:
    rule: Optional[Rule]          # None si es hecho base
    premises: Tuple[Atom, ...]    # premisas usadas (vacío si es hecho base)


class ForwardChainer:
    """
    Encadenamiento hacia adelante para KB Horn proposicional:
    - indexa reglas por premisas
    - usa "agenda" de hechos recién conocidos
    - dispara reglas cuando todas sus premisas están satisfechas
    """

    def __init__(self, facts: Iterable[Atom], rules: Iterable[Rule]) -> None:
        self.facts: Set[Atom] = set(facts)
        self.rules: List[Rule] = list(rules)

        # Index: premisa -> [rule_id...]
        self._premise_index: Dict[Atom, List[int]] = defaultdict(list)

        # Contadores por regla: cuántas premisas faltan
        self._missing_count: List[int] = []
        self._body_sets: List[Set[Atom]] = []

        for i, r in enumerate(self.rules):
            body_set = set(r.body)
            self._body_sets.append(body_set)
            self._missing_count.append(len(body_set))
            for prem in body_set:
                self._premise_index[prem].append(i)

        # Pruebas: Atom -> ProofStep
        self.proof: Dict[Atom, ProofStep] = {}
        for f in self.facts:
            self.proof[f] = ProofStep(rule=None, premises=())

        # Derivados (incluye hechos)
        self.derived: Set[Atom] = set(self.facts)

        # Estadísticas
        self.fired_rules: int = 0
        self.derived_new: int = 0

    def saturate(self, *, max_steps: int = 10_000_000) -> None:
        """
        Satura la KB hasta punto fijo o hasta max_steps pops del agenda.
        max_steps existe para blindar contra casos patológicos.
        """
        agenda = deque(self.facts)
        steps = 0

        # Copia local de missing_count para mutar (por performance y seguridad)
        missing = list(self._missing_count)

        while agenda:
            steps += 1
            if steps > max_steps:
                raise RuntimeError(f"ForwardChainer: excedió max_steps={max_steps}")

            p = agenda.popleft()

            # Disparar todas las reglas donde p aparece como premisa
            for rid in self._premise_index.get(p, []):
                if missing[rid] <= 0:
                    continue

                # Decrementa exactamente una vez por premisa (como trabajamos con body_set, está ok)
                missing[rid] -= 1

                # Cuando llega a 0, la regla está lista para disparar
                if missing[rid] == 0:
                    r = self.rules[rid]
                    h = r.head
                    self.fired_rules += 1

                    if h not in self.derived:
                        self.derived.add(h)
                        self.derived_new += 1
                        agenda.append(h)
                        # Guardar prueba: la regla + premisas completas
                        self.proof[h] = ProofStep(rule=r, premises=r.body)

        # No retornamos nada: estado queda en self.derived y self.proof

    def entails(self, query: Atom) -> bool:
        return query in self.derived

    def get_proof_tree(self, query: Atom) -> Optional[dict]:
        """
        Retorna una prueba como árbol serializable (dict) o None si no se deduce.
        """
        if query not in self.derived:
            return None

        visited: Set[Atom] = set()

        def build(a: Atom) -> dict:
            if a in visited:
                # En Horn proposicional, la derivación aquí no debería necesitar ciclos si saturación fue sana,
                # pero igual blindamos para serializar.
                return {"atom": str(a), "cycle": True}

            visited.add(a)
            step = self.proof.get(a)
            if step is None or step.rule is None:
                return {"atom": str(a), "type": "fact"}

            return {
                "atom": str(a),
                "type": "derived",
                "rule": str(step.rule),
                "premises": [build(p) for p in step.premises],
            }

        return build(query)


# -----------------------------
# Generación de tareas (ID/OOD) para DED-D1
# -----------------------------

@dataclass(frozen=True, slots=True)
class DedTask:
    facts: Tuple[Atom, ...]
    rules: Tuple[Rule, ...]
    query: Atom
    label: bool                 # True si KB |= query
    difficulty: str             # "easy" | "mid" | "hard"
    split: str                  # "id" | "ood"
    proof: Optional[dict]       # árbol de prueba si label=True


def _mk_atom(rng: random.Random, preds: Sequence[str], consts: Sequence[str], *, max_arity: int) -> Atom:
    pred = rng.choice(preds)
    arity = rng.randint(0, max_arity)
    args = tuple(rng.choice(consts) for _ in range(arity))
    return Atom(pred=pred, args=args)


def _make_chain_kb(
    rng: random.Random,
    *,
    depth: int,
    preds: Sequence[str],
    consts: Sequence[str],
    max_arity: int,
    distractor_rules: int,
    extra_facts: int,
) -> Tuple[List[Atom], List[Rule], Atom]:
    """
    Construye una KB con una cadena garantizada de longitud 'depth':
        f0
        f0 -> f1
        f1 -> f2
        ...
        f(depth-1) -> f(depth)
    El query será f(depth) (entailed por construcción).
    """
    # Hecho base inicial
    f0 = _mk_atom(rng, preds, consts, max_arity=max_arity)
    chain = [f0]

    # Construir átomos intermedios distintos
    used = {f0}
    while len(chain) < depth + 1:
        a = _mk_atom(rng, preds, consts, max_arity=max_arity)
        if a not in used:
            used.add(a)
            chain.append(a)

    rules: List[Rule] = []
    for i in range(depth):
        rules.append(Rule(head=chain[i + 1], body=(chain[i],)))

    facts: List[Atom] = [chain[0]]

    # Hechos extra (ruido)
    while len(facts) < 1 + extra_facts:
        a = _mk_atom(rng, preds, consts, max_arity=max_arity)
        if a not in used:
            used.add(a)
            facts.append(a)

    # Reglas distractoras (que no conectan con la cadena)
    for _ in range(distractor_rules):
        # cuerpo de 1 a 3 premisas
        k = rng.randint(1, 3)
        body = []
        for _j in range(k):
            body.append(_mk_atom(rng, preds, consts, max_arity=max_arity))
        head = _mk_atom(rng, preds, consts, max_arity=max_arity)
        rules.append(Rule(head=head, body=tuple(body)))

    query = chain[-1]
    return facts, rules, query


def _make_negative_query(
    rng: random.Random,
    preds: Sequence[str],
    consts: Sequence[str],
    max_arity: int,
    forbidden: Set[Atom],
) -> Atom:
    """
    Genera un atom que intentamos que NO sea deducible (al menos no trivialmente).
    """
    for _ in range(10_000):
        a = _mk_atom(rng, preds, consts, max_arity=max_arity)
        if a not in forbidden:
            return a
    # fallback (altamente improbable)
    return Atom(pred="never", args=("x",))


def generate_ded_d1_tasks(
    *,
    seed: int,
    n_kb: int,
    queries_per_kb: int,
    split: str,          # "id" | "ood"
    difficulty: str,     # "easy" | "mid" | "hard"
) -> List[DedTask]:
    """
    Genera un conjunto de tareas DED-D1.

    ID vs OOD:
    - ID: profundidades pequeñas y ruido moderado
    - OOD: profundidades más altas y/o ruido agresivo (más distractores)
    """
    if split not in {"id", "ood"}:
        raise ValueError("split debe ser 'id' o 'ood'")
    if difficulty not in {"easy", "mid", "hard"}:
        raise ValueError("difficulty debe ser 'easy', 'mid' o 'hard'")

    rng = random.Random(seed)

    # Parámetros base por dificultad (ID)
    if difficulty == "easy":
        depth_id = (1, 2)
        distract_id = (2, 6)
        extra_facts = (0, 2)
    elif difficulty == "mid":
        depth_id = (3, 4)
        distract_id = (6, 14)
        extra_facts = (2, 5)
    else:
        depth_id = (5, 6)
        distract_id = (14, 30)
        extra_facts = (4, 10)

    # OOD amplifica dificultad: sobre todo profundidad + distractores
    if split == "ood":
        depth = (depth_id[1] + 2, depth_id[1] + 4)
        distract = (distract_id[1] + 10, distract_id[1] + 30)
    else:
        depth = depth_id
        distract = distract_id

    # Vocabulario controlado (para que el espacio no explote)
    preds = [f"p{i}" for i in range(12)]
    consts = [f"c{i}" for i in range(12)]
    max_arity = 2

    tasks: List[DedTask] = []

    for kb_i in range(n_kb):
        d = rng.randint(depth[0], depth[1])
        dr = rng.randint(distract[0], distract[1])
        ef = rng.randint(extra_facts[0], extra_facts[1])

        facts, rules, pos_query = _make_chain_kb(
            rng,
            depth=d,
            preds=preds,
            consts=consts,
            max_arity=max_arity,
            distractor_rules=dr,
            extra_facts=ef,
        )

        ch = ForwardChainer(facts=facts, rules=rules)
        ch.saturate()

        # Precomputar universo "derivable" para diseñar negativos
        derivable = set(ch.derived)

        # Crear queries: 50% positivas, 50% negativas
        for q_i in range(queries_per_kb):
            is_pos = (q_i % 2 == 0)

            if is_pos:
                q = pos_query
                lbl = True
                proof = ch.get_proof_tree(q)
            else:
                q = _make_negative_query(rng, preds, consts, max_arity=max_arity, forbidden=derivable)
                lbl = ch.entails(q)  # normalmente False; si sale True, queda registrado (dataset honesto)
                proof = ch.get_proof_tree(q) if lbl else None

            tasks.append(
                DedTask(
                    facts=tuple(facts),
                    rules=tuple(rules),
                    query=q,
                    label=lbl,
                    difficulty=difficulty,
                    split=split,
                    proof=proof,
                )
            )

    return tasks


# -----------------------------
# Evaluación básica (sanity + métricas)
# -----------------------------

def evaluate_tasks(tasks: Sequence[DedTask]) -> dict:
    """
    Evalúa consistencia de etiquetas y métricas rápidas:
    - accuracy del motor contra sí mismo (debe ser 1.0 salvo negativos que accidentalmente sean entailed)
    - tasa de positivos reales
    - stats de tamaño y costo (fired_rules)
    """
    t0 = time.perf_counter()
    n = len(tasks)
    correct = 0
    positives = 0
    fired_sum = 0
    derived_sum = 0

    for task in tasks:
        ch = ForwardChainer(task.facts, task.rules)
        ch.saturate()

        pred = ch.entails(task.query)
        if pred == task.label:
            correct += 1
        if task.label:
            positives += 1

        fired_sum += ch.fired_rules
        derived_sum += len(ch.derived)

    dt = time.perf_counter() - t0
    acc = correct / n if n else 0.0
    pos_rate = positives / n if n else 0.0

    return {
        "n": n,
        "accuracy": acc,
        "positive_rate": pos_rate,
        "avg_fired_rules": fired_sum / n if n else 0.0,
        "avg_derived_atoms": derived_sum / n if n else 0.0,
        "wall_seconds": dt,
    }


def export_jsonl(tasks: Sequence[DedTask], path: str) -> None:
    """
    Exporta tareas a JSONL para entrenar/evaluar modelos (LLM u otros).
    """
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            obj = {
                "facts": [str(a) for a in t.facts],
                "rules": [str(r) for r in t.rules],
                "query": str(t.query),
                "label": bool(t.label),
                "difficulty": t.difficulty,
                "split": t.split,
                "proof": t.proof,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
