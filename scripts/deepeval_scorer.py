"""DeepEval-based multi-dimensional scoring for eval responses.

Uses DeepEval's G-Eval metrics to provide correctness, coherence, and
instruction-following scores alongside the existing LLM judge.
"""


def _lazy_imports():
    """Import deepeval lazily so the module doesn't break when deepeval isn't installed."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    return GEval, LLMTestCase, LLMTestCaseParams


# Metric definitions - each returns a 0-1 score

def _build_correctness_metric(model_name: str):
    GEval, _, LLMTestCaseParams = _lazy_imports()
    return GEval(
        name="Correctness",
        model=model_name,
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "Heavily penalize omission of important detail",
            "Vague language or differing opinions are acceptable",
            "Penalize hallucinated facts or fabricated references",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
    )


def _build_coherence_metric(model_name: str):
    GEval, _, LLMTestCaseParams = _lazy_imports()
    return GEval(
        name="Coherence",
        model=model_name,
        evaluation_steps=[
            "Evaluate whether the response has a clear logical flow",
            "Check that the response is well-structured and complete",
            "Assess whether complex ideas are presented clearly",
            "Identify any contradictions or confusing sections",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )


def _build_instruction_following_metric(model_name: str):
    GEval, _, LLMTestCaseParams = _lazy_imports()
    return GEval(
        name="Instruction Following",
        model=model_name,
        evaluation_steps=[
            "Check whether the response addresses all parts of the input prompt",
            "Verify adherence to any format, length, or constraint requirements in the context",
            "Penalize responses that ignore specific instructions or criteria",
            "Reward responses that follow implicit and explicit instructions precisely",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
    )


def score_with_deepeval(prompt_meta: dict, response: str, config: dict) -> dict:
    """Score a response using DeepEval G-Eval metrics.

    Args:
        prompt_meta: Prompt metadata from evals/default.json
        response: The model's response text
        config: Full config dict (uses deepeval and judge sections)

    Returns:
        {
            "deepeval_scores": {"correctness": 0.85, "coherence": 0.9, ...},
            "deepeval_avg": 0.87
        }
        On error, scores are None and deepeval_avg is None.
    """
    deepeval_cfg = config.get("deepeval", {})
    enabled_metrics = deepeval_cfg.get("metrics", ["correctness", "coherence", "instruction_following"])

    # Use judge model as evaluator, fallback to gpt-4.1
    judge_cfg = config.get("judge", {})
    models_cfg = config.get("models", {})
    judge_model_name = judge_cfg.get("model", "gpt-4.1")
    evaluator_model = models_cfg.get(judge_model_name, {}).get("model", "gpt-4.1")

    _, LLMTestCase, _ = _lazy_imports()

    # Build test case
    criteria = prompt_meta.get("criteria", [])
    if isinstance(criteria, list):
        context = criteria
    else:
        context = [str(criteria)]

    test_case = LLMTestCase(
        input=prompt_meta.get("prompt", ""),
        actual_output=response,
        expected_output=prompt_meta.get("ideal", ""),
        context=context,
    )

    # Build requested metrics
    metric_builders = {
        "correctness": _build_correctness_metric,
        "coherence": _build_coherence_metric,
        "instruction_following": _build_instruction_following_metric,
    }

    scores = {}
    for metric_name in enabled_metrics:
        builder = metric_builders.get(metric_name)
        if not builder:
            continue
        try:
            metric = builder(evaluator_model)
            metric.measure(test_case)
            scores[metric_name] = round(metric.score, 4) if metric.score is not None else None
        except Exception as e:
            print(f"      DeepEval {metric_name} error: {e}")
            scores[metric_name] = None

    valid_scores = [s for s in scores.values() if s is not None]
    avg = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    return {
        "deepeval_scores": scores,
        "deepeval_avg": avg,
    }
