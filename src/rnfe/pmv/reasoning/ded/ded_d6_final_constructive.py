
from __future__ import annotations
__all__ = [
    "DED_D6FinalConfig",
    "generate_ded_d6_final_tasks",
]
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import argparse
import hashlib
import json
import random
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# ----------------------------
# Core structures
# ----------------------------

@dataclass(frozen=True)
class Var:
    name: str

Term = Union[str, Var]  # constant or variable

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
        g: List[Term] = []
        for a in self.args:
            if isinstance(a, Var):
                g.append(subst[a.name])
            else:
                g.append(a)
        return Atom(self.pred, tuple(g))  # type: ignore[arg-type]

    def to_symbolic(self) -> str:
        inside = ",".join(str(a) for a in self.args)
        return f"{self.pred}({inside})"

@dataclass(frozen=True)
class Literal:
    atom: Atom
    neg: bool = False

    def vars(self) -> Set[str]:
        return self.atom.vars()

    def ground(self, subst: Dict[str, str]) -> "Literal":
        return Literal(self.atom.ground(subst), self.neg)

    def to_symbolic(self) -> str:
        return ("not " if self.neg else "") + self.atom.to_symbolic()

@dataclass(frozen=True)
class Rule:
    rid: str
    head: Atom
    body: Tuple[Literal, ...]

    def vars(self) -> Set[str]:
        out = set(self.head.vars())
        for lit in self.body:
            out |= lit.vars()
        return out

    def pos_body_vars(self) -> Set[str]:
        out: Set[str] = set()
        for lit in self.body:
            if not lit.neg:
                out |= lit.vars()
        return out

    def is_safe(self) -> bool:
        # Safety / range restriction: all variables must appear in positive body
        return self.vars().issubset(self.pos_body_vars())

    def ground_all(self, domain: Sequence[str]) -> Iterator["GroundRule"]:
        vnames = sorted(self.vars())
        if not vnames:
            yield GroundRule(self.rid, self.head, self.body)
            return
        for values in product(domain, repeat=len(vnames)):
            subst = dict(zip(vnames, values))
            yield GroundRule(
                self.rid,
                self.head.ground(subst),
                tuple(l.ground(subst) for l in self.body),
            )

@dataclass(frozen=True)
class GroundRule:
    rid: str
    head: Atom
    body: Tuple[Literal, ...]

    def __post_init__(self) -> None:
        if not self.head.is_ground():
            raise ValueError("GroundRule head must be ground")
        for lit in self.body:
            if not lit.atom.is_ground():
                raise ValueError("GroundRule body must be ground")

@dataclass(frozen=True)
class Derivation:
    fact: Atom
    rid: str
    pos_premises: Tuple[Atom, ...]
    neg_required_absent: Tuple[Atom, ...]

def _check_stratification(pred_stratum: Dict[str, int], rules: Sequence[Rule]) -> None:
    """
    Stratified negation:
      - positive deps: stratum(head) >= stratum(body_pred)
      - negative deps: stratum(head) >  stratum(body_pred)
    """
    for r in rules:
        sh = pred_stratum[r.head.pred]
        for lit in r.body:
            sb = pred_stratum[lit.atom.pred]
            if not lit.neg:
                if sh < sb:
                    raise ValueError(f"Not stratified (positive dep upwards): {r.rid}")
            else:
                if sh <= sb:
                    raise ValueError(f"Not stratified (negative dep not strictly down): {r.rid}")

def eval_stratified(
    *,
    domain: Sequence[str],
    edb_facts: Iterable[Atom],
    rules: Sequence[Rule],
    pred_stratum: Dict[str, int],
    max_steps: int = 200000,
) -> Tuple[Set[Atom], Dict[Atom, Derivation], Set[Atom]]:
    base_facts: Set[Atom] = set()
    for f in edb_facts:
        if not f.is_ground():
            raise ValueError("All EDB facts must be ground")
        base_facts.add(f)

    for r in rules:
        if not r.is_safe():
            raise ValueError(f"Unsafe rule: {r.rid}")

    _check_stratification(pred_stratum, rules)

    n_strata = 1 + max(pred_stratum.values())
    rules_by_s: List[List[Rule]] = [[] for _ in range(n_strata)]
    for r in rules:
        rules_by_s[pred_stratum[r.head.pred]].append(r)

    closure: Set[Atom] = set(base_facts)
    deriv: Dict[Atom, Derivation] = {}

    steps = 0
    for s in range(n_strata):
        grounded: List[GroundRule] = []
        for r in rules_by_s[s]:
            grounded.extend(list(r.ground_all(domain)))

        lower_snapshot = set(closure)  # negation checks are against fixed lower strata

        changed = True
        while changed:
            changed = False
            for gr in grounded:
                if steps >= max_steps:
                    raise RuntimeError("eval_stratified hit max_steps")
                if gr.head in closure:
                    continue

                pos_ok = True
                pos_prem: List[Atom] = []
                neg_abs: List[Atom] = []

                for lit in gr.body:
                    if not lit.neg:
                        if lit.atom not in closure:
                            pos_ok = False
                            break
                        pos_prem.append(lit.atom)
                    else:
                        if lit.atom in lower_snapshot:
                            pos_ok = False
                            break
                        neg_abs.append(lit.atom)

                if pos_ok:
                    closure.add(gr.head)
                    deriv[gr.head] = Derivation(
                        fact=gr.head,
                        rid=gr.rid,
                        pos_premises=tuple(pos_prem),
                        neg_required_absent=tuple(neg_abs),
                    )
                    changed = True
                    steps += 1

    return closure, deriv, base_facts

def proof_depth(a: Atom, deriv: Dict[Atom, Derivation], base_facts: Set[Atom], memo: Optional[Dict[Atom, int]] = None) -> int:
    """
    Depth:
      base fact -> 1
      derived   -> 1 + max(depth(pos_premises)) + (1 if uses negation)
    """
    if memo is None:
        memo = {}
    if a in memo:
        return memo[a]
    if a in base_facts:
        memo[a] = 1
        return 1
    d = deriv.get(a)
    if d is None:
        memo[a] = 0
        return 0
    ds = [proof_depth(p, deriv, base_facts, memo) for p in d.pos_premises]
    if any(x <= 0 for x in ds):
        memo[a] = 0
        return 0
    extra = 1 if d.neg_required_absent else 0
    memo[a] = 1 + max(ds) + extra
    return memo[a]

def extract_proof_trace(a: Atom, deriv: Dict[Atom, Derivation], base_facts: Set[Atom]) -> List[str]:
    seen: Set[Atom] = set()
    lines: List[str] = []

    def dfs(x: Atom) -> None:
        if x in seen:
            return
        seen.add(x)

        if x in base_facts:
            lines.append(f"[FACT] {x.to_symbolic()}")
            return

        d = deriv.get(x)
        if d is None:
            lines.append(f"[NO-PROOF] {x.to_symbolic()}")
            return

        for p in d.pos_premises:
            dfs(p)
        for na in d.neg_required_absent:
            lines.append(f"[NEG] not {na.to_symbolic()}  (ausente en estratos inferiores)")

        prem = ", ".join(p.to_symbolic() for p in d.pos_premises)
        negs = ", ".join(f"not {n.to_symbolic()}" for n in d.neg_required_absent)
        body = prem if prem else ""
        if negs:
            body = (body + ", " if body else "") + negs
        lines.append(f"[APPLY {d.rid}] {body}  ⟹  {x.to_symbolic()}")

    dfs(a)
    return lines

# ----------------------------
# D6 (final) constructive generator
# ----------------------------

@dataclass(frozen=True)
class DED_D6FinalConfig:
    n_tasks: int = 7200
    seed: int = 6
    out_path: str = "artifacts/ded_d6.jsonl"
    domain_size: int = 8
    min_proof_depth: int = 6

    # difficulty split (IID/OOD inside D6)
    iid_ratio: float = 0.75
    chain_len_iid: Tuple[int, int] = (4, 8)   # edges
    chain_len_ood: Tuple[int, int] = (9, 18)  # edges

    max_eval_steps: int = 200000

def _make_domain(n: int) -> List[str]:
    syms = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
        "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]
    if n > len(syms):
        raise ValueError(f"domain_size too large for current symbol set (max {len(syms)})")
    return syms[:n]

def _atom_to_nl(a: Atom) -> str:
    if len(a.args) == 1:
        return f"{a.args[0]} es {a.pred}."
    if len(a.args) == 2:
        return f"{a.args[0]} {a.pred} {a.args[1]}."
    return a.to_symbolic() + "."

def _choice_to_nl(a: Atom) -> str:
    if len(a.args) == 1:
        return f"{a.args[0]} es {a.pred}"
    if len(a.args) == 2:
        return f"{a.args[0]} {a.pred} {a.args[1]}"
    return a.to_symbolic()

def _build_prompt(domain: Sequence[str], facts: Sequence[Atom], rules: Sequence[Rule], choices: Sequence[Atom]) -> str:
    dom = ", ".join(domain)
    facts_nl = "\n".join(f"- {_atom_to_nl(f)}  ({f.to_symbolic()})" for f in facts)
    rules_nl = "\n".join(
        f"- {r.rid}: {r.head.to_symbolic()} :- " + ", ".join(l.to_symbolic() for l in r.body)
        for r in rules
    )
    ch = "\n".join(f"{i+1}) {_choice_to_nl(a)}  ({a.to_symbolic()})" for i, a in enumerate(choices))
    return (
        f"Dominio finito: {{{dom}}}\n\n"
        "HECHOS (EDB):\n"
        f"{facts_nl}\n\n"
        "REGLAS (estrato 0 positivo + estrato 1 con negación estratificada):\n"
        f"{rules_nl}\n\n"
        "Pregunta (DED-D6 final): ¿Cuál afirmación se deduce con certeza?\n"
        f"{ch}\n"
    )

def _construct_kb(
    rng: random.Random,
    domain: Sequence[str],
    chain_len: int,
) -> Tuple[List[Atom], List[Rule], Dict[str, int], Atom, Atom, Atom]:
    """
    Build a guaranteed-positive deep example:
      - Stratum 0:
          Edge(x,y) facts form a chain: v0->v1->...->vL
          Reach(x,y) rules: base + recursion (positive)
          Bad(y) some facts
      - Stratum 1:
          OK(y) :- Reach(v0,y), not Bad(y)
    Returns:
      facts, rules, pred_stratum, pos_query(OK(vL)), bad_choice(OK(v_bad)), unreach_choice(OK(v_u))
    """
    # predicates
    EDGE = "Edge"
    REACH = "Reach"
    BAD = "Bad"
    OK = "OK"

    # strata
    pred_stratum = {EDGE: 0, REACH: 0, BAD: 0, OK: 1}

    # pick nodes for chain
    if chain_len + 2 > len(domain):
        raise ValueError("domain too small for requested chain_len")
    nodes = domain[:]  # deterministic symbol pool
    rng.shuffle(nodes)

    v0 = nodes[0]
    chain_nodes = [v0] + nodes[1 : chain_len + 1]  # length chain_len edges => chain_len+1 nodes
    vT = chain_nodes[-1]

    # choose a reachable bad node distinct from vT
    v_bad = rng.choice(chain_nodes[:-1])

    # choose an unreachable node (not in chain)
    v_u = nodes[chain_len + 1]

    facts: List[Atom] = []
    # edge facts for chain
    for a, b in zip(chain_nodes[:-1], chain_nodes[1:]):
        facts.append(Atom(EDGE, (a, b)))

    # Bad facts (ensure vT is NOT bad)
    facts.append(Atom(BAD, (v_bad,)))

    # Some noise edges from chain to unrelated nodes, but NEVER into vT (avoid shortcuts)
    # Also avoid connecting to v_u (keep it unreachable)
    noise_targets = [x for x in domain if x not in chain_nodes and x != v_u]
    for _ in range(rng.randint(1, 4)):
        src = rng.choice(chain_nodes[:-1])
        if noise_targets:
            tgt = rng.choice(noise_targets)
            facts.append(Atom(EDGE, (src, tgt)))

    # Rules (stratum 0): Reach from Edge (positive recursion)
    X = Var("x")
    Y = Var("y")
    Z = Var("z")
    r1 = Rule("S0_R01", Atom(REACH, (X, Y)), (Literal(Atom(EDGE, (X, Y))),))
    r2 = Rule(
        "S0_R02",
        Atom(REACH, (X, Z)),
        (Literal(Atom(EDGE, (X, Y))), Literal(Atom(REACH, (Y, Z)))),
    )

    # Rule (stratum 1): OK(y) if reachable from v0 and not bad
    # OK(Y) :- Reach(v0, Y), not Bad(Y)
    r3 = Rule(
        "S1_R01",
        Atom(OK, (Y,)),
        (Literal(Atom(REACH, (v0, Y))), Literal(Atom(BAD, (Y,)), neg=True)),
    )

    rules = [r1, r2, r3]

    pos = Atom(OK, (vT,))
    bad_choice = Atom(OK, (v_bad,))
    unreach_choice = Atom(OK, (v_u,))
    reach_unreach = Atom(REACH, (v0, v_u))

    return facts, rules, pred_stratum, pos, bad_choice, unreach_choice, reach_unreach

def generate_ded_d6_final_tasks(cfg: DED_D6FinalConfig) -> Iterator[dict]:

    rng = random.Random(cfg.seed)
    domain = _make_domain(cfg.domain_size)

    # FAIL EARLY: evita generar IID y caer en OOD después
    max_chain_requested = max(cfg.chain_len_iid[1], cfg.chain_len_ood[1])
    min_domain_required = max_chain_requested + 2
    if len(domain) < min_domain_required:
        raise ValueError(
            f"domain_size={len(domain)} too small for chain_len_max={max_chain_requested}. "
            f"Use --domain >= {min_domain_required} (e.g., {min_domain_required + 2})."
        )

    n_iid = int(round(cfg.n_tasks * cfg.iid_ratio))
    n_ood = cfg.n_tasks - n_iid

    def pick_chain_len(difficulty: str, trng: random.Random) -> int:
        if difficulty == "iid":
            return trng.randint(*cfg.chain_len_iid)
        return trng.randint(*cfg.chain_len_ood)

    # ensure chain length always sufficient for min_proof_depth
    # OK depth = Reach depth + 2-ish => use chain_len >= min_proof_depth
    min_chain = max(3, cfg.min_proof_depth)  # conservative
    if cfg.chain_len_iid[1] < min_chain:
        raise ValueError("chain_len_iid too small for min_proof_depth")
    if cfg.chain_len_ood[1] < min_chain:
        raise ValueError("chain_len_ood too small for min_proof_depth")

    idx = 0
    total = cfg.n_tasks
    # Progreso global sobre todas las tareas
    for difficulty, n in (("iid", n_iid), ("ood", n_ood)):
        for _ in tqdm(range(n), desc=f"{difficulty.upper()} ({n})", leave=True):
            local_seed = (cfg.seed * 1_000_003) ^ idx
            trng = random.Random(local_seed)

            chain_len = pick_chain_len(difficulty, trng)
            chain_len = max(chain_len, min_chain)

            facts, rules, pred_stratum, pos, c2, c3, c4 = _construct_kb(trng, domain, chain_len)

            closure, deriv, base = eval_stratified(
                domain=domain,
                edb_facts=facts,
                rules=rules,
                pred_stratum=pred_stratum,
                max_steps=cfg.max_eval_steps,
            )

            # Build choices: exactly one entailed
            choices = [pos, c2, c3, c4]
            # verify invariant
            flags = [a in closure for a in choices]
            if flags.count(True) != 1 or not (pos in closure):
                raise AssertionError("Construction invariant broken (should not happen).")

            # depth gate
            d = proof_depth(pos, deriv, base)
            if d < cfg.min_proof_depth:
                raise AssertionError(f"Depth gate broken: depth={d} < {cfg.min_proof_depth}")

            trng.shuffle(choices)
            answer_index = choices.index(pos)

            facts_sorted = sorted(facts, key=lambda x: x.to_symbolic())
            prompt = _build_prompt(domain, facts_sorted, rules, choices)

            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
            tid = f"DED-D6-{difficulty.upper()}-{idx:06d}-{h}"

            row = {
                "id": tid,
                "schema": "rnfe.ded.d6.v1",
                "reasoning": "DED",
                "difficulty": "D6",
                "subdifficulty": difficulty,
                "domain": list(domain),
                "prompt": prompt,
                "choices": [_choice_to_nl(a) for a in choices],
                "choices_symbolic": [a.to_symbolic() for a in choices],
                "answer_index": answer_index,
                "answer_symbolic": pos.to_symbolic(),
                "meta": {
                    "seed": cfg.seed,
                    "local_seed": local_seed,
                    "chain_len": chain_len,
                    "proof_depth": d,
                    "n_facts": len(facts),
                    "n_rules": len(rules),
                    "generator": "constructive_chain_reach_notbad",
                },
                "proof_trace": extract_proof_trace(pos, deriv, base),
            }

            yield row
            idx += 1

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
    ap = argparse.ArgumentParser(description="Generate RNFE DED-D6 FINAL tasks (constructive stratified negation).")
    ap.add_argument("--out", type=str, default="artifacts/ded_d6.jsonl")
    ap.add_argument("--n", type=int, default=7200)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--domain", type=int, default=8)
    ap.add_argument("--min-proof-depth", type=int, default=6)
    ap.add_argument("--iid-ratio", type=float, default=0.75)
    args = ap.parse_args()

    cfg = DED_D6FinalConfig(
        n_tasks=args.n,
        seed=args.seed,
        out_path=args.out,
        domain_size=args.domain,
        min_proof_depth=args.min_proof_depth,
        iid_ratio=args.iid_ratio,
    )

    n = write_jsonl(cfg.out_path, generate_ded_d6_final_tasks(cfg))
    print(f"[DED-D6 FINAL] wrote {n} tasks to {cfg.out_path}")

if __name__ == "__main__":
    main()
