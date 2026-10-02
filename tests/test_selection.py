from lamarck.selection import ARCHIVED, IMPROVER, NEUTRAL, STEPPING_STONE, GateConfig, classify, compare, confirm, heldout_gate, screen


def test_compare_gain_and_preservation_regression():
    parent = {"a": 1.0, "b": 0.5, "c": 0.0}
    child = {"a": 0.75, "b": 0.0, "c": 1.0}
    c = compare(parent, child)
    assert c.gain == -0.25 + -0.5 + 1.0
    assert c.regression == 0.25  # only 'a' is in the preservation set (>= 0.8)


def test_classify():
    assert classify({"a"}, {"a", "b"}) == IMPROVER
    assert classify({"a"}, {"a"}) == NEUTRAL
    assert classify({"a", "b"}, {"a"}) == STEPPING_STONE
    assert classify({"a"}, {"b"}) == ARCHIVED


def test_screen_and_confirm():
    cfg = GateConfig(max_regression=0.5)
    good = compare({"a": 1.0, "b": 0.0, "c": 0.0}, {"a": 1.0, "b": 1.0, "c": 1.0})
    assert screen(good, 0.05, 3, cfg).accept
    assert confirm(good, 0.05, 3, cfg).accept
    lossy = compare({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0})
    assert not screen(lossy, 0.05, 2, cfg).accept  # gain 0
    small = compare({"a": 0.0, "b": 0.0}, {"a": 0.25, "b": 0.0})
    assert screen(small, 0.2, 2, cfg).accept  # permissive about trying
    assert not confirm(small, 0.2, 2, cfg).accept  # strict about trusting: inside the noise floor


def test_regression_beyond_delta_rejected():
    cfg = GateConfig(max_regression=0.5)
    c = compare({"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}, {"a": 0.0, "b": 1.0, "c": 1.0, "d": 1.0})
    assert c.gain > 0 and c.regression == 1.0
    assert not screen(c, 0.0, 4, cfg).accept


def test_cost_rule_rejects_expensive_small_gains():
    cfg = GateConfig(tokens_per_point=100)
    c = compare({"a": 0.0}, {"a": 0.5})
    assert not screen(c, 0.0, 1, cfg, token_delta_per_rollout=1000).accept
    assert screen(c, 0.0, 1, cfg, token_delta_per_rollout=10).accept


def test_heldout_gate():
    assert heldout_gate(0.40, 0.60, 0.05).accept
    assert not heldout_gate(0.40, 0.42, 0.05).accept
