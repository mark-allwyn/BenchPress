"""End-to-end tracer: one model's answers flow generate -> run -> score -> board,
using a mocked provider so no network/API is touched."""

import benchpress.modules.causal  # noqa: F401
import benchpress.scorers  # noqa: F401
from benchpress import stats
from benchpress.core import registry
from benchpress.providers.base import CompletionResult
from benchpress.runner import persist, run_model, score_model
from benchpress.runner.board import leaderboard


def _correct_answer(item) -> str:
    adj = next(p for p in item.parts if p.part_id == "ADJUSTMENT_SET")
    est = next(p for p in item.parts if p.part_id == "ESTIMATE")
    confounder = next(iter(adj.expected))
    return f"ADJUSTMENT_SET: {{{confounder}}}\nESTIMATE: {est.expected:.2f}\nIDENTIFIABLE: yes"


class FakeProvider:
    def __init__(self, answers: dict):
        self.answers = answers
        self.calls = 0

    def complete(self, prompt: str) -> CompletionResult:
        self.calls += 1
        return self.answers[prompt]


def _build(items):
    answers = {}
    # item 0: fully correct
    answers[items[0].prompt] = CompletionResult(content=_correct_answer(items[0]), stop_reason="end_turn")
    # item 1: wrong estimate
    answers[items[1].prompt] = CompletionResult(content="ADJUSTMENT_SET: {X}\nESTIMATE: 9.99\nIDENTIFIABLE: yes", stop_reason="end_turn")
    # item 2: refusal
    answers[items[2].prompt] = CompletionResult(content="", stop_reason="refusal")
    # items 3,4: no protocol -> invalid_answer
    for it in items[3:]:
        answers[it.prompt] = CompletionResult(content="I am not certain about this.", stop_reason="end_turn")
    return FakeProvider(answers)


def test_full_pipeline_one_model(tmp_path):
    items, meta = registry.get_module("causal")(seed=7)
    provider = _build(items)
    path = tmp_path / "fake.json"

    ran = run_model(provider, items, path, model_name="fake", benchmark="causal", version=meta.version)
    assert ran == len(items)
    assert provider.calls == len(items)

    score_model(items, path)
    results = persist.load_scored(items, path)
    acc = stats.accuracy(results)

    n = len(items)
    assert acc["attempted"] == n
    assert acc["correct"] == 1          # only item 0 answered correctly
    assert acc["refusals"] == 1         # item 2
    assert acc["invalids"] == n - 3     # items 3..n-1 gave no protocol
    assert acc["accuracy"] == 1 / n


def test_score_and_rerun_never_recall_provider(tmp_path):
    items, meta = registry.get_module("causal")(seed=7)
    provider = _build(items)
    path = tmp_path / "fake.json"

    run_model(provider, items, path, model_name="fake", benchmark="causal", version=meta.version)
    calls_after_first = provider.calls

    # Re-running run skips already-completed items (no new calls).
    run_model(provider, items, path, model_name="fake", benchmark="causal", version=meta.version)
    assert provider.calls == calls_after_first

    # Scoring is offline and idempotent; reruns append history, not overwrite.
    score_model(items, path)
    score_model(items, path)
    data = persist.load(path)
    assert all(len(runs) == 1 for runs in data["runs"].values())


def test_unparseable_part_scores_wrong_without_voiding_item(tmp_path):
    items, _ = registry.get_module("causal")(seed=7)
    item = items[0]
    # Adjustment set present and correct, estimate missing entirely.
    confounder = next(iter(next(p for p in item.parts if p.part_id == "ADJUSTMENT_SET").expected))
    content = f"ADJUSTMENT_SET: {{{confounder}}}\nIDENTIFIABLE: yes"
    provider = FakeProvider({item.prompt: CompletionResult(content=content, stop_reason="end_turn")})
    path = tmp_path / "p.json"
    run_model(provider, [item], path, model_name="p", benchmark="causal", version="1")
    score_model([item], path)
    result = persist.load_scored([item], path)[0]
    assert result.status == "ok"  # some fields parsed, so not invalid_answer
    assert result.item_correct is False
    by_id = {p.part_id: p for p in result.parts}
    assert by_id["ADJUSTMENT_SET"].correct is True
    assert by_id["ESTIMATE"].correct is False
    assert by_id["ESTIMATE"].parsed is None


def test_leaderboard_renders_model_and_accuracy(tmp_path):
    items, meta = registry.get_module("causal")(seed=7)
    provider = _build(items)
    path = tmp_path / "fake.json"
    run_model(provider, items, path, model_name="fake", benchmark="causal", version=meta.version)
    score_model(items, path)
    text = leaderboard([path])
    assert "fake" in text
    assert f"{100 / len(items):.1f}%" in text
