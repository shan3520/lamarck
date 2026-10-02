import pytest

from lamarck.evaluate import FatalRunError, evaluate
from lamarck.harness import Harness
from lamarck.tasks import Task


class Billing(Exception):
    status_code = 402


class Flaky(Exception):
    status_code = 529


class Target:
    name = "t"

    def __init__(self, exc):
        self.exc = exc

    def run(self, harness, task, seed):
        raise self.exc


TASKS = [Task("a", "p", {"type": "exact", "value": "x"})]


def test_billing_error_aborts_the_run():
    with pytest.raises(FatalRunError, match="out of credits"):
        evaluate(Target(Billing()), Harness({}), TASKS, k=2, workers=1)


def test_transient_error_is_recorded_not_fatal():
    res = evaluate(Target(Flaky()), Harness({}), TASKS, k=2, workers=2)
    assert res.infra_errors == 2 and res.rates["a"] == 0.0
