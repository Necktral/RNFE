from rnfe.pmv.reasoning.deductive_d1 import Atom, Rule, ForwardChainer

def test_simple_chain_entails():
    p = Atom("p", ("a",))
    q = Atom("q", ("a",))
    r = Atom("r", ("a",))
    s = Atom("s", ("a",))

    facts = [p, q]
    rules = [
        Rule(head=r, body=(p, q)),
        Rule(head=s, body=(r,)),
    ]

    ch = ForwardChainer(facts, rules)
    ch.saturate()

    assert ch.entails(p) is True
    assert ch.entails(r) is True
    assert ch.entails(s) is True

    proof = ch.get_proof_tree(s)
    assert proof is not None
    assert proof["atom"] == str(s)

def test_negative_query_not_entailed():
    a = Atom("a")
    b = Atom("b")
    c = Atom("c")

    facts = [a]
    rules = [Rule(head=b, body=(Atom("x"),))]  # premisa nunca aparece

    ch = ForwardChainer(facts, rules)
    ch.saturate()

    assert ch.entails(c) is False
    assert ch.get_proof_tree(c) is None

def test_cycle_does_not_loop():
    a = Atom("a")
    rules = [Rule(head=a, body=(a,))]  # a -> a
    ch = ForwardChainer([a], rules)
    ch.saturate()
    assert ch.entails(a) is True
