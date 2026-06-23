import pytest

from benchpress.core import registry


@pytest.fixture(autouse=True)
def clean_registries():
    # Each test starts from empty registries and restores afterwards.
    saved = (
        dict(registry.MODULES),
        dict(registry.PART_SCORERS),
        dict(registry.METRICS),
    )
    registry.MODULES.clear()
    registry.PART_SCORERS.clear()
    registry.METRICS.clear()
    yield
    registry.MODULES.clear()
    registry.MODULES.update(saved[0])
    registry.PART_SCORERS.clear()
    registry.PART_SCORERS.update(saved[1])
    registry.METRICS.clear()
    registry.METRICS.update(saved[2])


def test_registered_part_scorer_is_retrievable():
    @registry.register_part_scorer("set_match")
    def score(gold, raw):
        return raw

    assert registry.get_part_scorer("set_match") is score


def test_register_part_scorer_returns_the_function_unchanged():
    def score(gold, raw):
        return raw

    decorated = registry.register_part_scorer("numeric_tolerance")(score)
    assert decorated is score


def test_registered_module_is_retrievable():
    @registry.register_module("causal")
    def generate(seed, difficulty):
        return ([], None)

    assert registry.get_module("causal") is generate


def test_registered_metric_is_retrievable():
    @registry.register_metric("headline_accuracy")
    def metric(results):
        return {}

    assert registry.get_metric("headline_accuracy") is metric


def test_duplicate_registration_raises():
    @registry.register_part_scorer("set_match")
    def score_a(gold, raw):
        return raw

    with pytest.raises(ValueError):
        @registry.register_part_scorer("set_match")
        def score_b(gold, raw):
            return raw


def test_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        registry.get_part_scorer("does_not_exist")
