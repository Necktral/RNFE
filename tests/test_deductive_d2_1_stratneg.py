import pytest

pytest.importorskip("problog")

from rnfe.pmv.reasoning.deductive_d2_1_stratneg import (
    validate_stratified_and_safe,
    evaluate_d2_1,
    Ded2_1Task,
)

from rnfe.pmv.reasoning.deductive_d2_problog import problog_entails_many


def test_stratified_program_validates_and_runs():
    program = "\n".join([
        "edge(c0,c1).",
        "edge(c1,c2).",
        "blocked(c9).",
        "path(X,Y) :- edge(X,Y), \\+ blocked(Y).",
        "path(X,Z) :- edge(X,Y), path(Y,Z), \\+ blocked(Y).",
        "",
    ])

    ok, errs = validate_stratified_and_safe(program)
    assert ok is True, f"errs={errs}"

    labels, _dt = problog_entails_many(program, ["path(c0,c2)", "path(c0,c9)"])
    assert labels["path(c0,c2)"] is True
    assert labels["path(c0,c9)"] is False


def test_negative_cycle_detected_by_validator():
    program = "\n".join([
        "dom(c0).",
        "a(X) :- dom(X), \\+ b(X).",
        "b(X) :- dom(X), \\+ a(X).",
        "",
    ])
    ok, errs = validate_stratified_and_safe(program)
    assert ok is False
    assert any("Negative cycle" in e for e in errs)


def test_eval_counts_invalid_as_detected():
    # 1 valid, 1 invalid
    valid_prog = "\n".join([
        "edge(c0,c1).",
        "edge(c1,c2).",
        "blocked(c9).",
        "path(X,Y) :- edge(X,Y), \\+ blocked(Y).",
        "path(X,Z) :- edge(X,Y), path(Y,Z), \\+ blocked(Y).",
        "",
    ])
    invalid_prog = "\n".join([
        "dom(c0).",
        "a(X) :- dom(X), \\+ b(X).",
        "b(X) :- dom(X), \\+ a(X).",
        "",
    ])

    tasks = [
        Ded2_1Task(program=valid_prog, query="path(c0,c2)", expected="valid", label=True,
                  difficulty="easy", split="id", family="stratified", meta={}),
        Ded2_1Task(program=invalid_prog, query="a(c0)", expected="invalid", label=None,
                  difficulty="trap", split="trap", family="trap", meta={}),
    ]

    m = evaluate_d2_1(tasks)
    assert m.acc_valid == 1.0
    assert m.invalid_detection_rate == 1.0
