from propax.core.tolerances import TOL32, TOL64


def test_precision_moves_the_accuracies():
    assert TOL32.acc != TOL64.acc


def test_caps_are_precision_free():
    assert TOL32.caps == TOL64.caps


def test_domain_is_precision_free():
    assert TOL32.domain == TOL64.domain


def test_single_precision_loosens_every_accuracy_and_tightens_none():
    for f in TOL64.acc.__dataclass_fields__:
        a, b = getattr(TOL64.acc, f), getattr(TOL32.acc, f)
        if isinstance(a, dict):
            assert all(b[k] >= a[k] for k in a), f
        else:
            assert b >= a, f
