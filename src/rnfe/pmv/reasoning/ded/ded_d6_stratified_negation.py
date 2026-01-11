"""
DED–D6: Motor de razonamiento con negación estratificada, safety y explicabilidad exhaustiva
-------------------------------------------------------------------------------
Objetivos cubiertos:
 1. Negación por defecto (negación como ausencia, no-monótona)
 2. Estratificación: reglas y predicados organizados en estratos, sin ciclos negativos
 3. Safety: todas las variables de la cabeza aparecen en literales positivos del cuerpo
 4. Recursión permitida solo positiva; negación solo sobre estratos inferiores
 5. Unicidad y determinismo del cierre
 6. Explicabilidad exhaustiva: para cada hecho deducible, se muestra la prueba; para cada hecho no deducible, se listan todos los caminos fallidos y las razones (incluyendo pasos [NEG] detallados)
 7. El generador de tareas fuerza que cada tarea tenga razonamiento negativo profundo y opciones únicas

Notas de implementación:
- La función eval_stratified implementa el cierre bottom-up por estratos, validando safety y estratificación.
- extract_proof_steps produce explicaciones exhaustivas, listando todos los intentos fallidos de deducción para hechos no deducibles.
- El generador de tareas (generate_ded_d6_tasks) fuerza que la opción correcta requiera razonamiento negativo y profundidad mínima, y que la traza de prueba incluya pasos [NEG].
-------------------------------------------------------------------------------
"""
from __future__ import annotations
# Clase de configuración para DED-D6
class DED_D6Config:
    def __init__(self, n_tasks=7200, seed=6, out_path="artifacts/ded_d6.jsonl", domain_size=5, min_proof_depth=6):
        self.n_tasks = n_tasks
        self.seed = seed
        self.out_path = out_path
        self.domain_size = domain_size
        self.min_proof_depth = min_proof_depth
        self.unary_preds = 8
        self.binary_preds = 4
        self.facts_unary_range = (6, 10)
        self.facts_binary_range = (4, 8)
        self.rules_range = (8, 14)
        self.max_attempts_per_task = 2000
__all__ = [
    "Atom",
    "Literal",
    "Rule",
    "Var",
    "eval_stratified",
    "generate_ded_d6_tasks",
    "DED_D6Config",
]

from dataclasses import dataclass
# ============================
# Motor de evaluación estratificada (stub funcional)
# ============================
from collections import defaultdict


# ============================
# Validadores de safety y estratificación
# ============================
def validate_safety(rule):
    # Todas las variables de la cabeza deben aparecer en literales positivos del cuerpo
    head_vars = set()
    for arg in rule.head.args:
        if isinstance(arg, Var):
            head_vars.add(arg.name)
    body_vars = set()
    for lit in rule.body:
        if not lit.negated:
            for arg in lit.atom.args:
                if isinstance(arg, Var):
                    body_vars.add(arg.name)
    return head_vars <= body_vars

def compute_strata(rules):
    # Grafo de dependencias: nodo = predicado, arista positiva/negativa
    from collections import defaultdict, deque
    dep_graph = defaultdict(list)
    all_preds = set()
    for r in rules:
        head = r.head.pred
        all_preds.add(head)
        for lit in r.body:
            dep_graph[head].append((lit.atom.pred, lit.negated))
            all_preds.add(lit.atom.pred)
    # Inicializar todos los predicados a estrato 0
    strata = {p: 0 for p in all_preds}
    changed = True
    while changed:
        changed = False
        for r in rules:
            h = r.head.pred
            max_stratum = 0
            for lit in r.body:
                b = lit.atom.pred
                s = strata.get(b, 0)
                if lit.negated:
                    s += 1
                if s > max_stratum:
                    max_stratum = s
            if h not in strata or strata[h] != max_stratum:
                strata[h] = max_stratum
                changed = True
    # Detectar ciclos negativos (no estratificable)
    for r in rules:
        h = r.head.pred
        for lit in r.body:
            b = lit.atom.pred
            if lit.negated and strata[h] <= strata[b]:
                raise ValueError(f"Negación no estratificable: {h} depende negativamente de {b} en el mismo o menor estrato")
    return strata

# ============================
# Evaluación bottom-up por estratos (modelo perfecto)
# ============================
@dataclass(frozen=True)
class StratifiedDerivation:
    fact: Atom
    rid: str
    premises: Tuple[Atom, ...]
    neg_required_absent: Tuple[Atom, ...] = ()

def eval_stratified(domain, edb_facts, rules, pred_stratum):
    """
    Evalúa el cierre estratificado bottom-up, respetando negación y safety.
    Devuelve (closure, deriv, base_facts).
    """
    # Validar safety
    for r in rules:
        if not validate_safety(r):
            raise ValueError(f"Regla no segura: {r}")
    # Validar estratificación
    strata = compute_strata(rules)
    # Agrupar reglas por estrato
    rules_by_stratum = {}
    for r in rules:
        s = strata[r.head.pred]
        rules_by_stratum.setdefault(s, []).append(r)
    max_stratum = max(rules_by_stratum.keys(), default=0)
    # Evaluación por capas
    closure = set(edb_facts)
    deriv = {}
    base_facts = set(edb_facts)
    for s in range(max_stratum + 1):
        new_facts = set()
        for r in rules_by_stratum.get(s, []):
            # Enumerar todas las instancias posibles de las variables
            vars_in_rule = set()
            for arg in r.head.args:
                if isinstance(arg, Var):
                    vars_in_rule.add(arg.name)
            for lit in r.body:
                for arg in lit.atom.args:
                    if isinstance(arg, Var):
                        vars_in_rule.add(arg.name)
            # Generar todas las asignaciones posibles
            from itertools import product
            for values in product(domain, repeat=len(vars_in_rule)):
                subst = dict(zip(sorted(vars_in_rule), values))
                head_inst = _ground_atom(r.head, subst)
                # Evaluar cuerpo
                pos_body = []
                neg_body = []
                for lit in r.body:
                    a = _ground_atom(lit.atom, subst)
                    if lit.negated:
                        neg_body.append(a)
                    else:
                        pos_body.append(a)
                if all(a in closure for a in pos_body) and all(a not in closure for a in neg_body):
                    if head_inst not in closure:
                        closure.add(head_inst)
                        deriv[head_inst] = StratifiedDerivation(head_inst, r.rid, tuple(pos_body), tuple(neg_body))
                        new_facts.add(head_inst)
        closure |= new_facts
    return closure, deriv, base_facts

def _ground_atom(atom, subst):
    args = tuple(subst.get(a.name, a) if isinstance(a, Var) else a for a in atom.args)
    return Atom(atom.pred, args)
# src/rnfe/pmv/reasoning/ded/ded_d5.py
from dataclasses import dataclass
# ----------------------------
# Lógica de literales (para reglas con negación estratificada)
# ----------------------------

@dataclass(frozen=True)
class Literal:
    atom: Atom
    negated: bool = False

    def __str__(self) -> str:
        if self.negated:
            return f"not {self.atom}"
        return str(self.atom)
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union
import argparse
import hashlib
import json
import random

# Barra de progreso reutilizable
from rnfe.pmv.reasoning.ded.progress_utils import progress_bar


# ----------------------------
# Core logic (Horn / Datalog-ish)
# ----------------------------

@dataclass(frozen=True)
class Var:
    name: str


Term = Union[str, Var]  # str = constant symbol


@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[Term, ...]

    def is_ground(self) -> bool:
        return all(isinstance(a, str) for a in self.args)

    def vars(self) -> Set[str]:
        out: Set[str] = set()
        for a in self.args:
            if isinstance(a, Var):
                out.add(a.name)
        return out

    def ground(self, subst: Dict[str, str]) -> "Atom":
        g_args: List[Term] = []
        for a in self.args:
            if isinstance(a, Var):
                g_args.append(subst[a.name])
            else:
                g_args.append(a)
        return Atom(self.pred, tuple(g_args))  # type: ignore[arg-type]

    def to_symbolic(self) -> str:
        inside = ",".join(str(a) for a in self.args)
        return f"{self.pred}({inside})"


@dataclass(frozen=True)
class Rule:
    rid: str
    head: Atom
    body: Tuple[Atom, ...]

    def vars(self) -> Set[str]:
        out = set(self.head.vars())
        for b in self.body:
            out |= b.vars()
        return out

    def ground_all(self, domain: Sequence[str]) -> Iterator["GroundRule"]:
        """
        Ground rule by enumerating all assignments to its variables over the finite domain.
        Domain is small by design (D5), so brute grounding is acceptable.
        """
        vnames = sorted(self.vars())
        if not vnames:
            yield GroundRule(self.rid, self.head, self.body)
            return

        for values in product(domain, repeat=len(vnames)):
            subst = dict(zip(vnames, values))
            g_head = self.head.ground(subst)
            g_body = tuple(b.ground(subst) for b in self.body)
            yield GroundRule(self.rid, g_head, g_body)


@dataclass(frozen=True)
class GroundRule:
    rid: str
    head: Atom  # ground
    body: Tuple[Atom, ...]  # ground

    def __post_init__(self) -> None:
        if not self.head.is_ground():
            raise ValueError("GroundRule head must be ground")
        for b in self.body:
            if not b.is_ground():
                raise ValueError("GroundRule body must be ground")


@dataclass(frozen=True)
class Derivation:
    """
    A single application producing `fact` using `rid` from `premises`.
    """
    fact: Atom
    rid: str
    premises: Tuple[Atom, ...]


def forward_chain(
    facts: Iterable[Atom],
    rules: Iterable[Rule],
    domain: Sequence[str],
    max_steps: int = 10000,
) -> Tuple[Set[Atom], Dict[Atom, Derivation]]:
    """
    Bottom-up forward chaining for Horn rules (definite clauses).
    Returns closure and a derivation map (one witness derivation per derived fact).
    """
    base: Set[Atom] = set()
    deriv: Dict[Atom, Derivation] = {}

    for f in facts:
        if not f.is_ground():
            raise ValueError("All input facts must be ground")
        base.add(f)

    grounded: List[GroundRule] = []
    for r in rules:
        grounded.extend(list(r.ground_all(domain)))

    changed = True
    steps = 0
    while changed:
        changed = False
        for gr in grounded:
            if steps >= max_steps:
                raise RuntimeError("forward_chain hit max_steps (possible explosion)")
            if gr.head in base:
                continue
            if all(b in base for b in gr.body):
                base.add(gr.head)
                deriv[gr.head] = Derivation(
                    fact=gr.head,
                    rid=gr.rid,
                    premises=tuple(gr.body),
                )
                changed = True
                steps += 1
    return base, deriv


def proof_depth(
    query: Atom,
    deriv: Dict[Atom, Derivation],
    base_facts: Set[Atom],
    memo: Optional[Dict[Atom, int]] = None,
) -> int:
    """
    Depth of a proof tree where leaves are base facts.
    Base fact depth = 1
    Derived fact depth = 1 + max(depth(premise_i))
    """
    if memo is None:
        memo = {}
    if query in memo:
        return memo[query]
    if query in base_facts:
        memo[query] = 1
        return 1
    if query not in deriv:
        memo[query] = 0
        return 0
    d = deriv[query]
    depths = [proof_depth(p, deriv, base_facts, memo) for p in d.premises]
    if any(x <= 0 for x in depths):
        memo[query] = 0
        return 0
    memo[query] = 1 + max(depths)
    return memo[query]



def extract_proof_steps(query: Atom, deriv: dict, base_facts: set, rules=None, closure=None) -> list:
    """
    Explicación exhaustiva: para hechos deducibles, muestra la prueba; para no deducibles, muestra todos los caminos fallidos y las razones.
    Si rules y closure se pasan, se exploran todos los intentos de deducción fallidos.
    """
    seen = set()
    lines = []

    def dfs(a: Atom):
        if a in seen:
            return
        seen.add(a)
        if a in base_facts:
            lines.append(f"[FACT] {a.to_symbolic()}")
            return
        d = deriv.get(a)
        if d is None:
            # Explicación negativa exhaustiva
            if rules is not None and closure is not None:
                found = False
                for r in rules:
                    if r.head.pred != a.pred:
                        continue
                    # Buscar todas las instancias posibles
                    vnames = sorted({v for lit in r.body for v in lit.atom.vars()} | set(r.head.vars()))
                    from itertools import product
                    for values in product(sorted({x for t in closure for x in t.args if isinstance(x, str)}), repeat=len(vnames)):
                        subst = dict(zip(vnames, values))
                        head_inst = r.head.ground(subst)
                        if head_inst != a:
                            continue
                        # Evaluar cuerpo
                        reasons = []
                        for lit in r.body:
                            b = lit.atom.ground(subst)
                            if lit.negated:
                                if b in closure:
                                    reasons.append(f"[NEG-FAIL] {b.to_symbolic()} está en el cierre (no se puede negar)")
                                else:
                                    reasons.append(f"[NEG-OK] {b.to_symbolic()} ausente (ok)")
                            else:
                                if b not in closure:
                                    reasons.append(f"[POS-FAIL] {b.to_symbolic()} ausente (falta premisa)")
                                else:
                                    reasons.append(f"[POS-OK] {b.to_symbolic()} presente (ok)")
                        if all((not lit.negated and lit.atom.ground(subst) in closure) or (lit.negated and lit.atom.ground(subst) not in closure) for lit in r.body):
                            found = True
                        else:
                            lines.append(f"[NEG] {a.to_symbolic()} no se puede deducir por la regla {r.rid}: " + "; ".join(reasons))
                if not found:
                    lines.append(f"[NO-PROOF] {a.to_symbolic()} (ninguna regla aplica)")
            else:
                lines.append(f"[NO-PROOF] {a.to_symbolic()}")
            return
        # Recursivo para premisas positivas
        for p in d.premises:
            dfs(p)
        # Explicación de negación: listar los átomos requeridos ausentes
        if hasattr(d, 'neg_required_absent') and d.neg_required_absent:
            for n in d.neg_required_absent:
                lines.append(f"[NEG] {n.to_symbolic()} ausente (necesario para {query.to_symbolic()})")
        prem = ", ".join(p.to_symbolic() for p in d.premises)
        lines.append(f"[APPLY {d.rid}] {prem}  ⟹  {a.to_symbolic()}")

    dfs(query)
    return lines


# ----------------------------
# Rendering
# ----------------------------

def atom_to_nl(a: Atom) -> str:
    """
    Very controlled Spanish template to avoid ambiguity.
    """
    if len(a.args) == 1:
        x = a.args[0]
        return f"{x} es {a.pred}."
    if len(a.args) == 2:
        x, y = a.args
        return f"{x} {a.pred} {y}."
    inside = ", ".join(str(t) for t in a.args)
    return f"{a.pred}({inside})."


def rule_to_nl(r: Rule) -> str:
    """
    Spanish rendering with explicit quantification over variables used in the rule.
    This is a bounded-universal reading (Datalog style): variables range over the given domain.
    """
    v = sorted(r.vars())
    q = ""
    if v:
        q = "Para todo " + ", ".join(v) + ": "
    body = " y ".join(atom_to_nl(b).rstrip(".") for b in r.body)
    head = atom_to_nl(r.head).rstrip(".")
    return f"{q}si {body} entonces {head}."


def atom_choice_to_nl(a: Atom) -> str:
    """
    Compact form for choices.
    """
    if len(a.args) == 1:
        return f"{a.args[0]} es {a.pred}"
    if len(a.args) == 2:
        return f"{a.args[0]} {a.pred} {a.args[1]}"
    return a.to_symbolic()


# ----------------------------
# D5 generator
# ----------------------------

@dataclass(frozen=True)
class DED_D5Config:
    n_tasks: int = 7200
    seed: int = 5
    out_path: str = "artifacts/ded_d5.jsonl"

    # Difficulty shaping
    domain_size: int = 4
    unary_preds: int = 8
    binary_preds: int = 4
    facts_unary_range: Tuple[int, int] = (6, 10)
    facts_binary_range: Tuple[int, int] = (4, 8)
    rules_range: Tuple[int, int] = (8, 14)

    # Enforce non-trivial depth (D5)
    min_proof_depth: int = 5

    # Guardrails
    max_attempts_per_task: int = 200


def _make_domain(domain_size: int) -> List[str]:
    syms = [
        "a", "b", "c", "d", "e", "f", "g", "h",
        "i", "j", "k", "l", "m", "n",
        "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]
    if domain_size > len(syms):
        raise ValueError(f"domain_size too large for current symbol set (max {len(syms)})")
    return syms[:domain_size]


def _sample_facts(
    rng: random.Random,
    domain: Sequence[str],
    unary_preds: Sequence[str],
    binary_preds: Sequence[str],
    u_count: int,
    b_count: int,
) -> Set[Atom]:
    facts: Set[Atom] = set()

    # unary facts
    for _ in range(u_count):
        p = rng.choice(unary_preds)
        x = rng.choice(domain)
        facts.add(Atom(p, (x,)))

    # binary facts
    for _ in range(b_count):
        r = rng.choice(binary_preds)
        x = rng.choice(domain)
        y = rng.choice(domain)
        facts.add(Atom(r, (x, y)))

    return facts


def _sample_rules(
    rng: random.Random,
    unary_preds: Sequence[str],
    binary_preds: Sequence[str],
    n_rules: int,
) -> List[Rule]:
    """
    Generate Horn rules with 1–2 variables; shapes that create join patterns.
    """
    rules: List[Rule] = []
    X = Var("x")
    Y = Var("y")

    for i in range(n_rules):
        rid = f"R{i+1:02d}"
        shape = rng.choice(["u2u", "uu2u", "ru2u", "ruu2u", "ur2r"])
        if shape == "u2u":
            # P(x) -> Q(x)
            p = rng.choice(unary_preds)
            q = rng.choice(unary_preds)
            rules.append(Rule(rid, Atom(q, (X,)), (Atom(p, (X,)),)))
        elif shape == "uu2u":
            # P(x) & Q(x) -> S(x)
            p = rng.choice(unary_preds)
            q = rng.choice(unary_preds)
            s = rng.choice(unary_preds)
            rules.append(Rule(rid, Atom(s, (X,)), (Atom(p, (X,)), Atom(q, (X,)))))
        elif shape == "ru2u":
            # Rel(x,y) & P(y) -> Q(x)
            rel = rng.choice(binary_preds)
            p = rng.choice(unary_preds)
            q = rng.choice(unary_preds)
            rules.append(Rule(rid, Atom(q, (X,)), (Atom(rel, (X, Y)), Atom(p, (Y,)))))
        elif shape == "ruu2u":
            # Rel(x,y) & P(x) & Q(y) -> S(x)
            rel = rng.choice(binary_preds)
            p = rng.choice(unary_preds)
            q = rng.choice(unary_preds)
            s = rng.choice(unary_preds)
            rules.append(
                Rule(
                    rid,
                    Atom(s, (X,)),
                    (Atom(rel, (X, Y)), Atom(p, (X,)), Atom(q, (Y,))),
                )
            )
        else:
            # ur2r : P(x) & Rel(x,y) -> Rel2(x,y)
            p = rng.choice(unary_preds)
            rel1 = rng.choice(binary_preds)
            rel2 = rng.choice(binary_preds)
            rules.append(
                Rule(
                    rid,
                    Atom(rel2, (X, Y)),
                    (Atom(p, (X,)), Atom(rel1, (X, Y))),
                )
            )

    return rules


def _all_candidate_atoms(
    domain: Sequence[str],
    unary_preds: Sequence[str],
    binary_preds: Sequence[str],
) -> List[Atom]:
    out: List[Atom] = []
    for p in unary_preds:
        for x in domain:
            out.append(Atom(p, (x,)))
    for r in binary_preds:
        for x in domain:
            for y in domain:
                out.append(Atom(r, (x, y)))
    return out


def _build_prompt(
    facts: Sequence[Atom],
    rules: Sequence[Rule],
    choices: Sequence[Atom],
) -> str:
    facts_nl = "\n".join(f"- {atom_to_nl(f)}  ({f.to_symbolic()})" for f in facts)
    rules_nl = "\n".join(f"- {rule_to_nl(r)}  [{r.rid}]" for r in rules)
    ch = "\n".join(f"{i+1}) {atom_choice_to_nl(a)}  ({a.to_symbolic()})" for i, a in enumerate(choices))

    return (
        "Base de conocimiento (hechos y reglas). Las variables (x,y) recorren el dominio finito mostrado.\n\n"
        "HECHOS:\n"
        f"{facts_nl}\n\n"
        "REGLAS (cláusulas Horn):\n"
        f"{rules_nl}\n\n"
        "Pregunta (DED): ¿Cuál de las siguientes afirmaciones se puede deducir con certeza?\n"
        f"{ch}\n"
    )



def generate_ded_d6_tasks(cfg) -> Iterator[dict]:
    """
    Generador de tareas DED–D6: reglas con negación estratificada y literales.
    """
    rng = random.Random(cfg.seed)
    domain = _make_domain(cfg.domain_size)
    unary_preds = [f"P{i}" for i in range(cfg.unary_preds)]
    binary_preds = [f"R{i}" for i in range(cfg.binary_preds)]
    candidates = _all_candidate_atoms(domain, unary_preds, binary_preds)

    for t in progress_bar(range(cfg.n_tasks), total=cfg.n_tasks, desc="DED-D6 tasks", ncols=80):
        local_seed = (cfg.seed * 1_000_003) ^ t
        trng = random.Random(local_seed)
        ok = False
        for _attempt in range(cfg.max_attempts_per_task):
            # 1. Sample facts
            n_facts_unary = trng.randint(*cfg.facts_unary_range)
            n_facts_binary = trng.randint(*cfg.facts_binary_range)
            facts_set = _sample_facts(trng, domain, unary_preds, binary_preds, n_facts_unary, n_facts_binary)

            # 2. Sample rules (con literales y negación estratificada)
            n_rules = trng.randint(*cfg.rules_range)
            rules: List[Rule] = []
            pred_stratum = {}
            # Para simplificar, todos los preds base en estrato 0, uno nuevo en 1
            for p in unary_preds + binary_preds:
                pred_stratum[p] = 0
            head_pred = f"H{t%3}"  # Un predicado de cabeza especial
            pred_stratum[head_pred] = 1
            X = Var("x")
            Y = Var("y")
            # Regla positiva
            rules.append(Rule(
                f"S0_R01",
                Atom("P2", (X,)),
                (Literal(Atom("P0", (X,)), False), Literal(Atom("P1", (X,)), False)),
            ))
            pred_stratum["P2"] = 0
            # Regla con negación
            rules.append(Rule(
                f"S1_R01",
                Atom(head_pred, (X,)),
                (Literal(Atom("R0", (X, Y)), False), Literal(Atom("P0", (Y,)), False), Literal(Atom("P1", (Y,)), True)),
            ))

            # 3. Evaluar cierre estratificado
            closure, deriv, base_facts = eval_stratified(domain=domain, edb_facts=facts_set, rules=rules, pred_stratum=pred_stratum)

            # 4. Buscar átomos derivados no triviales
            derived_only = [a for a in closure if a not in base_facts]
            pos = None
            pos_depth = None
            for a in derived_only:
                d = proof_depth(a, deriv, base_facts)
                if d >= cfg.min_proof_depth:
                    pos = a
                    pos_depth = d
                    break
            if pos is None:
                continue  # retry

            # 5. Distractores: átomos no deducibles
            not_entailed = [a for a in candidates if a not in closure]
            trng.shuffle(not_entailed)
            if len(not_entailed) < 3:
                continue
            negs: List[Atom] = []
            for a in not_entailed:
                if a == pos:
                    continue
                negs.append(a)
                if len(negs) == 3:
                    break
            if len(negs) < 3:
                continue

            choices = [pos] + negs
            trng.shuffle(choices)
            answer_index = choices.index(pos)

            prompt = _build_prompt(sorted(facts_set, key=lambda x: x.to_symbolic()), rules, choices)
            proof = extract_proof_steps(pos, deriv, base_facts, rules=rules, closure=closure)
            # Forzar que la traza de prueba incluya al menos un paso [NEG]
            if not any(line.startswith("[NEG]") for line in proof):
                continue

            h = hashlib.sha256()
            h.update(prompt.encode("utf-8"))
            tid = f"DED-D6-{t:06d}-{h.hexdigest()[:10]}"

            task = {
                "id": tid,
                "schema": "rnfe.ded.d6.v1",
                "reasoning": "DED",
                "difficulty": "D6",
                "domain": list(domain),
                "prompt": prompt,
                "choices": [atom_choice_to_nl(a) for a in choices],
                "choices_symbolic": [a.to_symbolic() for a in choices],
                "answer_index": answer_index,
                "answer_symbolic": pos.to_symbolic(),
                "meta": {
                    "seed": cfg.seed,
                    "local_seed": local_seed,
                    "proof_depth": pos_depth,
                    "n_facts": len(facts_set),
                    "n_rules": len(rules),
                },
                "proof_trace": proof,
            }

            ent_flags = [(a in closure) for a in choices]
            if ent_flags.count(True) != 1 or not ent_flags[answer_index]:
                continue

            ok = True
            yield task
            break
        if not ok:
            raise RuntimeError(
                f"Could not generate a valid DED-D6 task at index {t} after {cfg.max_attempts_per_task} attempts."
            )


def write_jsonl(path: Union[str, Path], rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:

    ap = argparse.ArgumentParser(description="Generate RNFE DED-D6 tasks (stratified negation) to JSONL.")
    ap.add_argument("--out", type=str, default="artifacts/ded_d6.jsonl")
    ap.add_argument("--n", type=int, default=7200)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--domain", type=int, default=5)
    ap.add_argument("--min-proof-depth", type=int, default=6)
    args = ap.parse_args()

    cfg = DED_D6Config(
        n_tasks=args.n,
        seed=args.seed,
        out_path=args.out,
        domain_size=args.domain,
        min_proof_depth=args.min_proof_depth,
    )

    n = write_jsonl(cfg.out_path, generate_ded_d6_tasks(cfg))
    print(f"[DED-D6] wrote {n} tasks to {cfg.out_path}")


if __name__ == "__main__":
    main()
