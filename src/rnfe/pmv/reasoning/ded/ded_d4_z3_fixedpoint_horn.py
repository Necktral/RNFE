from __future__ import annotations


import dataclasses
import heapq
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Barra de progreso
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import z3  # type: ignore
except Exception:  # pragma: no cover
    z3 = None  # type: ignore

# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class WeightedEdge:
    u: int
    v: int
    w: int

    def to_dict(self) -> Dict[str, int]:
        return {"u": self.u, "v": self.v, "w": self.w}


@dataclass(frozen=True)
class DedD4Task:
    """
    DED-D4 task:
      - KB: grafo dirigido acíclico con pesos positivos
      - Query: existe un camino s -> t con costo total <= B ?
      - Label: True/False
    """
    task_id: str
    difficulty: str  # "iid" | "ood"
    n_nodes: int
    edges: List[WeightedEdge]
    source: int
    target: int
    budget: int
    label: bool

    # Explicación por construcción (camino mínimo)
    proof_edge_indices: Optional[List[int]] = None
    shortest_cost: Optional[int] = None

    # Diagnóstico z3 fixedpoint
    z3_status: Optional[str] = None  # "sat" | "unsat" | "unknown"
    z3_answer: Optional[str] = None

    meta: Optional[Dict[str, object]] = None

    def to_json(self) -> str:
        payload = {
            "task_id": self.task_id,
            "task_type": "DED-D4/Horn-CostReach",
            "difficulty": self.difficulty,
            "n_nodes": self.n_nodes,
            "edges": [e.to_dict() for e in self.edges],
            "query": {"source": self.source, "target": self.target, "budget": self.budget},
            "label": self.label,
        }
        if self.proof_edge_indices is not None:
            payload["proof_edge_indices"] = self.proof_edge_indices
        if self.shortest_cost is not None:
            payload["shortest_cost"] = self.shortest_cost
        if self.z3_status is not None:
            payload["z3_status"] = self.z3_status
        if self.z3_answer is not None:
            payload["z3_answer"] = self.z3_answer
        if self.meta is not None:
            payload["meta"] = self.meta
        return json.dumps(payload, ensure_ascii=False)


# -----------------------------
# Ground-truth labeling (Dijkstra in DAG; works for any nonnegative weights)
# -----------------------------

def _dijkstra_with_prev(
    n: int, edges: Sequence[WeightedEdge], s: int
) -> Tuple[List[int], List[Optional[Tuple[int, int]]]]:
    """
    Returns:
      dist[v] = shortest distance from s to v (or INF)
      prev[v] = (prev_node, edge_index) to reconstruct path
    """
    INF = 10**18
    g: List[List[Tuple[int, int, int]]] = [[] for _ in range(n)]  # (to, w, edge_idx)
    for idx, e in enumerate(edges):
        g[e.u].append((e.v, e.w, idx))

    dist = [INF] * n
    prev: List[Optional[Tuple[int, int]]] = [None] * n
    dist[s] = 0
    pq: List[Tuple[int, int]] = [(0, s)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w, edge_idx in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, edge_idx)
                heapq.heappush(pq, (nd, v))
    return dist, prev


def _reconstruct_edge_path(prev: Sequence[Optional[Tuple[int, int]]], t: int) -> Optional[List[int]]:
    if prev[t] is None:
        return None
    out: List[int] = []
    cur = t
    while prev[cur] is not None:
        pu, edge_idx = prev[cur]
        out.append(edge_idx)
        cur = pu
    out.reverse()
    return out


# -----------------------------
# Z3 Fixedpoint oracle (Horn rules)
# -----------------------------

class Z3FixedpointCostReachOracle:
    """
    Usa Z3 Fixedpoint (muZ) para decidir si hay derivación de:
        path(s, t, C) ∧ C <= B

    Z3 Fixedpoint está pensado para Datalog/Constrained Horn Clauses. :contentReference[oaicite:2]{index=2}
    Convención de query: sat = hay derivación; unsat = no hay derivación. :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, timeout_ms: int = 5000, engine: str = "spacer"):
        if z3 is None:
            raise RuntimeError("z3-solver no está instalado. Instalar con: pip install z3-solver")
        self.timeout_ms = int(timeout_ms)
        self.engine = engine

    def _build_fp(self, n_nodes: int, edges: Sequence[WeightedEdge]) -> "z3.Fixedpoint":  # type: ignore[name-defined]
        fp = z3.Fixedpoint()
        # engine="spacer" suele manejar mejor CHC con aritmética que el datalog básico.
        fp.set(engine=self.engine)
        fp.set(timeout=self.timeout_ms)

        Node = z3.IntSort()
        Cost = z3.IntSort()

        edge = z3.Function("edge", Node, Node, Cost, z3.BoolSort())
        path = z3.Function("path", Node, Node, Cost, z3.BoolSort())

        fp.register_relation(edge, path)

        X, Y, Z = z3.Ints("X Y Z")
        W, C, C1 = z3.Ints("W C C1")
        fp.declare_var(X, Y, Z, W, C, C1)

        # Facts: edge(u,v,w)
        for e in edges:
            fp.fact(edge(z3.IntVal(e.u), z3.IntVal(e.v), z3.IntVal(e.w)))

        # Rules:
        # path(X,Y,C) :- edge(X,Y,W) & C == W
        fp.rule(
            path(X, Y, C),
            [edge(X, Y, W), C == W],
            name="p_base",
        )

        # path(X,Z,C) :- edge(X,Y,W) & path(Y,Z,C1) & C == W + C1
        fp.rule(
            path(X, Z, C),
            [edge(X, Y, W), path(Y, Z, C1), C == W + C1],
            name="p_rec",
        )

        return fp

    def exists_path_with_cost_leq(
        self, n_nodes: int, edges: Sequence[WeightedEdge], source: int, target: int, budget: int
    ) -> Tuple[bool, str, Optional[str]]:
        fp = self._build_fp(n_nodes, edges)

        Node = z3.IntSort()
        Cost = z3.IntSort()
        path = z3.Function("path", Node, Node, Cost, z3.BoolSort())

        S = z3.IntVal(int(source))
        T = z3.IntVal(int(target))
        B = z3.IntVal(int(budget))
        Cq = z3.Int("Cq")
        fp.declare_var(Cq)

        q = z3.And(path(S, T, Cq), Cq <= B)
        st = fp.query(q)

        # st is one of: sat / unsat / unknown
        st_s = str(st)

        if st == z3.sat:
            ans = fp.get_answer()
            return True, "sat", str(ans) if ans is not None else None
        if st == z3.unsat:
            return False, "unsat", None
        return False, "unknown", fp.reason_unknown()


# -----------------------------
# Generator
# -----------------------------

def _sample_dag_edges(
    rng: random.Random,
    n_nodes: int,
    m_edges: int,
    w_min: int,
    w_max: int,
) -> List[WeightedEdge]:
    """
    DAG: u < v siempre, para evitar ciclos -> número finito de caminos/costos.
    """
    edges: List[WeightedEdge] = []
    used = set()
    tries = 0
    while len(edges) < m_edges and tries < m_edges * 30:
        tries += 1
        u = rng.randrange(0, n_nodes - 1)
        v = rng.randrange(u + 1, n_nodes)
        if (u, v) in used:
            continue
        used.add((u, v))
        w = rng.randint(w_min, w_max)
        edges.append(WeightedEdge(u=u, v=v, w=w))
    return edges


def generate_ded_d4_tasks(
    *,
    num_tasks: int = 7200,
    seed: int = 1337,
    iid_ratio: float = 0.75,
    validate_with_z3: bool = True,
    timeout_ms: int = 5000,
) -> List[DedD4Task]:
    rng = random.Random(seed)
    n_iid = int(round(num_tasks * iid_ratio))
    n_ood = num_tasks - n_iid

    oracle = Z3FixedpointCostReachOracle(timeout_ms=timeout_ms, engine="spacer") if validate_with_z3 else None

    tasks: List[DedD4Task] = []
    total = n_iid + n_ood
    use_tqdm = tqdm is not None
    outer = [("iid", n_iid), ("ood", n_ood)]
    if use_tqdm:
        pbar = tqdm(total=total, desc="DED-D4 tasks", ncols=80)
    try:
        for difficulty, n_tasks in outer:
            for i in range(n_tasks):
                task_seed = rng.getrandbits(64)
                trng = random.Random(task_seed)

                if difficulty == "iid":
                    n_nodes = trng.randint(10, 24)
                    m_edges = trng.randint(n_nodes + 6, 3 * n_nodes + 12)
                    w_min, w_max = 1, 9
                else:
                    n_nodes = trng.randint(28, 60)
                    m_edges = trng.randint(3 * n_nodes, 7 * n_nodes)
                    w_min, w_max = 1, 25

                edges = _sample_dag_edges(trng, n_nodes, m_edges, w_min, w_max)

                # Pick query
                s = trng.randrange(n_nodes)
                t = trng.randrange(n_nodes)
                if s == t:
                    t = (t + 1) % n_nodes

                dist, prev = _dijkstra_with_prev(n_nodes, edges, s)
                INF = 10**18
                reachable = dist[t] < INF
                shortest = int(dist[t]) if reachable else None
                proof = _reconstruct_edge_path(prev, t) if reachable else None

                # Create label-balanced budgets
                want_true = trng.random() < 0.5
                if not reachable:
                    budget = trng.randint(0, 30 if difficulty == "iid" else 120)
                    label = False
                else:
                    assert shortest is not None
                    if want_true:
                        slack = trng.randint(0, 3 if difficulty == "iid" else 8)
                        budget = shortest + slack
                        label = True
                    else:
                        cut = trng.randint(1, 5 if difficulty == "iid" else 15)
                        budget = max(0, shortest - cut)
                        label = False

                task_id = f"DED-D4-{difficulty.upper()}-{i:05d}"
                task = DedD4Task(
                    task_id=task_id,
                    difficulty=difficulty,
                    n_nodes=n_nodes,
                    edges=edges,
                    source=s,
                    target=t,
                    budget=int(budget),
                    label=label,
                    proof_edge_indices=proof if label else None,
                    shortest_cost=shortest,
                    meta={"seed": task_seed, "generator": "dag+dijkstra+budget"},
                )

                if oracle is not None:
                    ok, status, answer = oracle.exists_path_with_cost_leq(
                        n_nodes=n_nodes, edges=edges, source=s, target=t, budget=int(budget)
                    )
                    task = dataclasses.replace(task, z3_status=status, z3_answer=answer)
                    # si st es "sat" o "unsat" exigimos consistencia con el label
                    if status in ("sat", "unsat") and ok != label:
                        raise AssertionError(
                            f"Label mismatch on {task_id}: generator={label} z3={ok} status={status}"
                        )

                tasks.append(task)
                if use_tqdm:
                    pbar.update(1)
    finally:
        if use_tqdm:
            pbar.close()
    rng.shuffle(tasks)
    return tasks


def export_jsonl(tasks: Sequence[DedD4Task], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t.to_json() + "\n")


def build_manifest(tasks: Sequence[DedD4Task]) -> Dict[str, object]:
    n = len(tasks)
    iid = sum(1 for t in tasks if t.difficulty == "iid")
    ood = n - iid
    pos = sum(1 for t in tasks if t.label)
    neg = n - pos
    unknown = sum(1 for t in tasks if t.z3_status == "unknown")
    return {
        "task_type": "DED-D4/Horn-CostReach",
        "n_tasks": n,
        "iid": iid,
        "ood": ood,
        "label_true": pos,
        "label_false": neg,
        "true_ratio": (pos / n) if n else 0.0,
        "z3_unknown": unknown,
    }


def export_manifest(manifest: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
