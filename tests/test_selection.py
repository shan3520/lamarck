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


def test_per_success_cost_rule_replays_real_repo_run6():
    # run-6: parent solves 29.2% of 6 tasks at 35,008 tokens/rollout. The old absolute rule (4000 tokens
    # per task of gain) rejected the g=+1.25 candidate before its re-test.
    parent = {f"t{i}": r for i, r in enumerate([0.25, 0.5, 0.0, 0.5, 0.25, 0.25])}
    cfg = GateConfig()  # per-success rule by default

    def child(g):
        return {t: r + g / 6 for t, r in parent.items()}

    strong = compare(parent, child(1.25))
    assert screen(strong, 0.0, 6, cfg, token_delta_per_rollout=8914, parent_tokens=35008).accept
    weak = compare(parent, child(0.25))
    assert not screen(weak, 0.0, 6, cfg, token_delta_per_rollout=6793, parent_tokens=35008).accept
    assert not screen(weak, 0.0, 6, cfg, token_delta_per_rollout=5674, parent_tokens=35008).accept
    assert not screen(strong, 0.0, 6, GateConfig(tokens_per_point=4000), token_delta_per_rollout=8914, parent_tokens=35008).accept


def test_per_success_rule_allows_any_cost_when_parent_solves_nothing():
    c = compare({"a": 0.0, "b": 0.0}, {"a": 0.5, "b": 0.0})
    assert screen(c, 0.0, 2, GateConfig(), token_delta_per_rollout=50_000, parent_tokens=1000).accept
