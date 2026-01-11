import pytest

z3 = pytest.importorskip("z3")

from rnfe.pmv.reasoning.ded.ded_d4_z3_fixedpoint_horn import (  # noqa: E402
    WeightedEdge,
    Z3FixedpointCostReachOracle,
    generate_ded_d4_tasks,
)


def test_fixedpoint_oracle_cost_reachability():
    # 0 -> 1 (2), 1 -> 2 (3) => shortest 0->2 cost=5
    edges = [WeightedEdge(0, 1, 2), WeightedEdge(1, 2, 3)]
    oracle = Z3FixedpointCostReachOracle(timeout_ms=3000, engine="spacer")

    ok1, st1, _ = oracle.exists_path_with_cost_leq(n_nodes=3, edges=edges, source=0, target=2, budget=5)
    assert st1 == "sat"
    assert ok1 is True

    ok2, st2, _ = oracle.exists_path_with_cost_leq(n_nodes=3, edges=edges, source=0, target=2, budget=4)
    assert st2 == "unsat"
    assert ok2 is False


def test_generate_small_dataset_with_validation():
    tasks = generate_ded_d4_tasks(num_tasks=40, seed=123, validate_with_z3=True, timeout_ms=3000)
    assert len(tasks) == 40
    assert any(t.label for t in tasks)
    assert any(not t.label for t in tasks)
    # Si hay mismatch generator vs Z3 en sat/unsat, levanta AssertionError.
