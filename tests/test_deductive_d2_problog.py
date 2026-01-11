import pytest

pytest.importorskip("problog")

from rnfe.pmv.reasoning.deductive_d2_problog import problog_entails_many

def test_ded_d2_simple_path_true_false():
    program = "\n".join([
        "edge(c0,c1).",
        "edge(c1,c2).",
        "edge(c2,c3).",
        "path(X,Y) :- edge(X,Y).",
        "path(X,Z) :- edge(X,Y), path(Y,Z).",
        "",
    ])

    labels, _dt = problog_entails_many(program, ["path(c0,c3)", "path(c0,c4)"])
    assert labels["path(c0,c3)"] is True
    assert labels["path(c0,c4)"] is False
