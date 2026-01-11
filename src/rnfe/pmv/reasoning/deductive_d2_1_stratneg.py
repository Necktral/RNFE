from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json
import random
import re
import time

# Reutilizamos el backend ProbLog de D2 (no duplicamos el motor).
from rnfe.pmv.reasoning.deductive_d2_problog import problog_entails_many

# ============================================================
# DED–D2.1: Negación estratificada + safety + datasets ID/OOD
# ============================================================

@dataclass(frozen=True, slots=True)
class Ded2_1Task:
    program: str                  # programa ProbLog/Prolog SIN query(...)
    query: str                    # átomo ground, ej: "path(c0,c7)"
    expected: str                 # "valid" | "invalid"
    label: Optional[bool]         # solo para expected="valid"
    difficulty: str               # "easy" | "mid" | "hard"
    split: str                    # "id" | "ood"
    family: str                   # "stratified" | "trap"
    meta: Dict[str, int]


# ----------------------------
# Parsing mínimo de reglas
# ----------------------------

_RULE_RE = re.compile(r"^\s*(.*?)\s*:-\s*(.*?)\s*\.\s*$")
_FACT_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\)\s*\.\s*$|^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*$")

_VAR_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")

def _split_body_literals(body: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            lit = "".join(buf).strip()
            if lit:
                out.append(lit)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out

def _is_negated_literal(lit: str) -> bool:
    s = lit.strip()
    if s.startswith(r"\+"):
        return True
    if s.lower().startswith("not "):
        return True
    return False

def _strip_negation(lit: str) -> str:
    s = lit.strip()
    if s.startswith(r"\+"):
        return s[2:].strip()
    if s.lower().startswith("not "):
        return s[3:].strip()
    return s

def _pred_name(atom_like: str) -> Optional[str]:
    s = atom_like.strip()
    if not s:
        return None
    m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(\(|$)", s)
    return m.group(1) if m else None

@dataclass(frozen=True, slots=True)
class ParsedRule:
    head_pred: str
    body_pos_preds: Tuple[str, ...]
    body_neg_preds: Tuple[str, ...]
    pos_vars: Set[str]
    neg_vars: Set[str]

def parse_program_rules(program: str) -> List[ParsedRule]:
    rules: List[ParsedRule] = []
    for raw in program.splitlines():
        line = raw.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        if line.startswith("query("):
            continue
        m = _RULE_RE.match(line)
        if not m:
            continue
        head_txt = m.group(1).strip()
        body_txt = m.group(2).strip()
        hp = _pred_name(head_txt)
        if hp is None:
            continue
        lits = _split_body_literals(body_txt)
        pos_preds: List[str] = []
        neg_preds: List[str] = []
        pos_vars: Set[str] = set()
        neg_vars: Set[str] = set()
        for lit in lits:
            is_neg = _is_negated_literal(lit)
            atom = _strip_negation(lit)
            bp = _pred_name(atom)
            if bp is None:
                continue
            vars_in_lit = set(_VAR_RE.findall(atom))
            if is_neg:
                neg_preds.append(bp)
                neg_vars |= vars_in_lit
            else:
                pos_preds.append(bp)
                pos_vars |= vars_in_lit
        rules.append(
            ParsedRule(
                head_pred=hp,
                body_pos_preds=tuple(pos_preds),
                body_neg_preds=tuple(neg_preds),
                pos_vars=pos_vars,
                neg_vars=neg_vars,
            )
        )
    return rules

# ----------------------------
# Estratificación: SCC con aristas negativas
# ----------------------------

def _tarjan_scc(nodes: Sequence[str], edges: Dict[str, List[Tuple[str, bool]]]) -> List[List[str]]:
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    sccs: List[List[str]] = []
    def strongconnect(v: str) -> None:
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for (w, _neg) in edges.get(v, []):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)
    for n in nodes:
        if n not in idx:
            strongconnect(n)
    return sccs

def validate_stratified_and_safe(program: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    prules = parse_program_rules(program)
    preds: Set[str] = set()
    edges: Dict[str, List[Tuple[str, bool]]] = {}
    for r in prules:
        preds.add(r.head_pred)
        for bp in r.body_pos_preds:
            preds.add(bp)
            edges.setdefault(r.head_pred, []).append((bp, False))
        for bn in r.body_neg_preds:
            preds.add(bn)
            edges.setdefault(r.head_pred, []).append((bn, True))
        missing = r.neg_vars - r.pos_vars
        if missing:
            errors.append(
                f"Safety violation en regla con cabeza '{r.head_pred}': "
                f"vars en negación sin anclaje positivo: {sorted(missing)}"
            )
    nodes = sorted(preds)
    sccs = _tarjan_scc(nodes, edges)
    scc_id: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for p in comp:
            scc_id[p] = i
    for u, outs in edges.items():
        for (v, is_neg) in outs:
            if is_neg and scc_id.get(u) == scc_id.get(v):
                errors.append(
                    f"Negative cycle detectado: '{u}' depende negativamente de '{v}' en el mismo SCC"
                )
    return (len(errors) == 0), errors

# ----------------------------
# Programas base D2.1: path con bloqueos
# ----------------------------

def _rules_path_blocked() -> List[str]:
    return [
        "path(X,Y) :- edge(X,Y), \\+ blocked(Y).",
        "path(X,Z) :- edge(X,Y), path(Y,Z), \\+ blocked(Y).",
    ]

def _mk_const(i: int) -> str:
    return f"c{i}"

def _edge(a: str, b: str) -> str:
    return f"edge({a},{b})."

def _blocked(x: str) -> str:
    return f"blocked({x})."

def _noise_fact(pred: str, args: Sequence[str]) -> str:
    inner = ",".join(args)
    return f"{pred}({inner})."

def _gen_reachable_graph(
    rng: random.Random,
    *,
    chain_depth: int,
    chain_nodes: Sequence[str],
    side_nodes: Sequence[str],
    extra_edges: int,
) -> List[str]:
    all_nodes = list(chain_nodes) + list(side_nodes)
    index_of = {n: i for i, n in enumerate(all_nodes)}
    edges: Set[Tuple[str, str]] = set()
    for i in range(chain_depth):
        edges.add((chain_nodes[i], chain_nodes[i + 1]))
    tries = 0
    while len(edges) < (chain_depth + extra_edges) and tries < 200_000:
        tries += 1
        a = rng.choice(all_nodes)
        b = rng.choice(all_nodes)
        if a == b:
            continue
        if index_of[a] < index_of[b]:
            edges.add((a, b))
    return [_edge(a, b) for (a, b) in edges]

def _build_program(
    *,
    edge_lines: Sequence[str],
    blocked_lines: Sequence[str],
    noise_lines: Sequence[str],
    rules: Sequence[str],
) -> str:
    return "\n".join(list(edge_lines) + list(blocked_lines) + list(noise_lines) + list(rules)) + "\n"

# ----------------------------
# Generación datasets
# ----------------------------

def generate_ded_d2_1_tasks(
    *,
    seed: int,
    n_kb: int,
    queries_per_kb: int,
    split: str,        # id | ood
    difficulty: str,   # easy | mid | hard
) -> List[Ded2_1Task]:
    if split not in {"id", "ood"}:
        raise ValueError("split debe ser 'id' o 'ood'")
    if difficulty not in {"easy", "mid", "hard"}:
        raise ValueError("difficulty debe ser 'easy' o 'mid' o 'hard'")
    rng = random.Random(seed)
    rules = _rules_path_blocked()
    if difficulty == "easy":
        depth_id = (2, 3)
        extra_edges_id = (0, 3)
        noise_id = (0, 4)
        side_id = (3, 6)
        blocked_id = (1, 2)
        n_consts_id = 14
    elif difficulty == "mid":
        depth_id = (4, 6)
        extra_edges_id = (5, 15)
        noise_id = (5, 15)
        side_id = (6, 12)
        blocked_id = (2, 5)
        n_consts_id = 24
    else:
        depth_id = (7, 10)
        extra_edges_id = (15, 45)
        noise_id = (15, 45)
        side_id = (12, 22)
        blocked_id = (5, 12)
        n_consts_id = 36
    if split == "ood":
        depth = (depth_id[1] + 3, depth_id[1] + 6)
        extra_edges = (extra_edges_id[1] + 10, extra_edges_id[1] + 60)
        noise = (noise_id[1] + 10, noise_id[1] + 60)
        side = (side_id[1] + 6, side_id[1] + 18)
        blocked = (blocked_id[1] + 3, blocked_id[1] + 12)
        n_consts = n_consts_id + 12
    else:
        depth = depth_id
        extra_edges = extra_edges_id
        noise = noise_id
        side = side_id
        blocked = blocked_id
        n_consts = n_consts_id
    noise_preds = ["foo", "bar", "baz", "tag", "rel"]
    tasks: List[Ded2_1Task] = []
    for _ in range(n_kb):
        consts = [_mk_const(i) for i in range(n_consts)]
        d = rng.randint(depth[0], depth[1])
        n_side = rng.randint(side[0], side[1])
        n_extra = rng.randint(extra_edges[0], extra_edges[1])
        n_noise = rng.randint(noise[0], noise[1])
        chain_nodes = consts[: d + 1]
        side_nodes = consts[d + 1 : d + 1 + n_side]
        neg_pool = consts[d + 1 + n_side :]
        edge_lines = _gen_reachable_graph(
            rng,
            chain_depth=d,
            chain_nodes=chain_nodes,
            side_nodes=side_nodes,
            extra_edges=n_extra,
        )
        blocked_nodes = set(rng.sample(side_nodes, k=min(len(side_nodes), rng.randint(blocked[0], blocked[1])))) if side_nodes else set()
        blocked_lines = [_blocked(x) for x in sorted(blocked_nodes)]
        noise_lines: List[str] = []
        for _j in range(n_noise):
            pred = rng.choice(noise_preds)
            arity = rng.choice([1, 2])
            args = [rng.choice(consts) for _k in range(arity)]
            noise_lines.append(_noise_fact(pred, args))
        program = _build_program(
            edge_lines=edge_lines,
            blocked_lines=blocked_lines,
            noise_lines=noise_lines,
            rules=rules,
        )
        ok, errs = validate_stratified_and_safe(program)
        if not ok:
            raise RuntimeError("Generador produjo programa no estratificable o no-safe:\n" + "\n".join(errs))
        src = chain_nodes[0]
        pos_target = chain_nodes[-1]
        pos_q = f"path({src},{pos_target})"
        blocked_list = sorted(blocked_nodes)
        for qi in range(queries_per_kb):
            if qi % 2 == 0:
                q = pos_q
            else:
                if qi % 4 == 1 and blocked_list:
                    tgt = rng.choice(blocked_list)
                    q = f"path({src},{tgt})"
                else:
                    if neg_pool:
                        tgt = rng.choice(neg_pool)
                    else:
                        tgt = _mk_const(n_consts + 999)
                    q = f"path({src},{tgt})"
            labels, _dt = problog_entails_many(program, [q])
            lbl = bool(labels.get(q, False))
            tasks.append(
                Ded2_1Task(
                    program=program,
                    query=q,
                    expected="valid",
                    label=lbl,
                    difficulty=difficulty,
                    split=split,
                    family="stratified",
                    meta={
                        "depth": d,
                        "n_consts": n_consts,
                        "n_edges": program.count("edge("),
                        "n_blocked": len(blocked_nodes),
                        "n_noise": n_noise,
                        "n_rules": len(rules),
                    },
                )
            )
    return tasks

def generate_ded_d2_1_trap_tasks(*, seed: int, n: int = 200) -> List[Ded2_1Task]:
    rng = random.Random(seed)
    tasks: List[Ded2_1Task] = []
    for i in range(n):
        c = _mk_const(rng.randint(0, 50))
        program = "\n".join([
            f"dom({c}).",
            "a(X) :- dom(X), \\+ b(X).",
            "b(X) :- dom(X), \\+ a(X).",
        ]) + "\n"
        q = f"a({c})"
        tasks.append(
            Ded2_1Task(
                program=program,
                query=q,
                expected="invalid",
                label=None,
                difficulty="trap",
                split="trap",
                family="trap",
                meta={"i": i},
            )
        )
    return tasks

# ----------------------------
# Evaluación (métricas)
# ----------------------------

@dataclass(frozen=True, slots=True)
class EvalMetrics:
    n: int
    acc_valid: float
    exception_rate_valid: float
    invalid_detection_rate: float
    wall_seconds_total: float
    wall_seconds_problog: float

def evaluate_d2_1(tasks: Sequence[Ded2_1Task]) -> EvalMetrics:
    t0 = time.perf_counter()
    n_valid = 0
    n_valid_correct = 0
    n_valid_ex = 0
    n_invalid = 0
    n_invalid_detected = 0
    wall_problog = 0.0
    by_prog: Dict[str, List[Ded2_1Task]] = {}
    for t in tasks:
        by_prog.setdefault(t.program, []).append(t)
    for prog, group in by_prog.items():
        ok, errs = validate_stratified_and_safe(prog)
        valids = [g for g in group if g.expected == "valid"]
        invalids = [g for g in group if g.expected == "invalid"]
        if not ok:
            for g in invalids:
                n_invalid += 1
                n_invalid_detected += 1
            for g in valids:
                n_valid += 1
                n_valid_ex += 1
            continue
        if valids:
            qs = [g.query for g in valids]
            try:
                labels, dt = problog_entails_many(prog, qs)
                wall_problog += dt
                for g in valids:
                    n_valid += 1
                    pred = bool(labels.get(g.query, False))
                    if g.label is not None and pred == g.label:
                        n_valid_correct += 1
            except Exception:
                for _g in valids:
                    n_valid += 1
                    n_valid_ex += 1
        for g in invalids:
            n_invalid += 1
            try:
                _labels, dt = problog_entails_many(prog, [g.query])
                wall_problog += dt
            except Exception:
                n_invalid_detected += 1
    wall_total = time.perf_counter() - t0
    acc_valid = (n_valid_correct / n_valid) if n_valid else 0.0
    ex_rate_valid = (n_valid_ex / n_valid) if n_valid else 0.0
    invalid_rate = (n_invalid_detected / n_invalid) if n_invalid else 0.0
    return EvalMetrics(
        n=len(tasks),
        acc_valid=acc_valid,
        exception_rate_valid=ex_rate_valid,
        invalid_detection_rate=invalid_rate,
        wall_seconds_total=wall_total,
        wall_seconds_problog=wall_problog,
    )

def export_jsonl(tasks: Sequence[Ded2_1Task], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            obj = {
                "program": t.program,
                "query": t.query,
                "expected": t.expected,
                "label": t.label,
                "difficulty": t.difficulty,
                "split": t.split,
                "family": t.family,
                "meta": t.meta,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
