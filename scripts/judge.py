"""LLM-as-judge scoring for eval responses.

Uses a separate LLM call to score each response against the prompt's
ideal answer and criteria, replacing manual human scoring.
"""

import json
import re


JUDGE_PROMPT = """\
You are an expert evaluator for LLM responses. Score the response on a 1-5 scale:

  5 - Excellent: Fully addresses the prompt, accurate, well-structured, meets all criteria
  4 - Good: Mostly correct with minor gaps or style issues
  3 - Adequate: Partially addresses the prompt, some errors or missing elements
  2 - Poor: Significant errors, missing key requirements, or off-topic
  1 - Failing: Wrong, harmful, empty, or completely misses the point

Guidelines:
- Compare the response against the ideal answer and criteria provided
- Penalise hallucinated facts, fabricated references, or confident wrong answers
- Reward appropriate hedging, asking for clarification, or refusing harmful requests
- Consider auto-check flags as additional signal (failures should lower the score)
- Be strict but fair - a 3 is average, 5 is genuinely excellent

Return ONLY a JSON object (no markdown fences, no extra text):
{"score": <1-5>, "rationale": "<1-2 sentence explanation>"}
"""


def build_judge_prompt(prompt_meta: dict, response: str, auto_checks: dict) -> str:
    """Assemble the user message for the judge LLM call."""
    parts = [
        JUDGE_PROMPT,
        "\n--- ORIGINAL PROMPT ---",
        prompt_meta.get("prompt", ""),
        "\n--- IDEAL ANSWER ---",
        prompt_meta.get("ideal", ""),
        "\n--- CRITERIA ---",
    ]

    criteria = prompt_meta.get("criteria", [])
    if isinstance(criteria, list):
        for c in criteria:
            parts.append(f"- {c}")
    else:
        parts.append(str(criteria))

    flags = auto_checks.get("flags", [])
    if flags:
        parts.append("\n--- AUTO-CHECK FLAGS ---")
        for f in flags:
            parts.append(f"- {f}")

    parts.append("\n--- RESPONSE TO EVALUATE ---")
    parts.append(response)

    return "\n".join(parts)


def _extract_json_object(text: str) -> str | None:
    """Find the outermost {...} in text, handling nested braces."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_judge_response(raw: str) -> dict:
    """Extract score and rationale JSON from judge output.

    Handles markdown fences, leading text, and whitespace.
    Returns {"score": int|None, "rationale": str}.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to find JSON object in text
    extracted = _extract_json_object(text)
    if extracted:
        text = extracted

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"score": None, "rationale": f"Failed to parse judge response: {raw[:200]}"}

    score = parsed.get("score")
    rationale = parsed.get("rationale", "")

    if not isinstance(score, int) or score < 1 or score > 5:
        return {"score": None, "rationale": f"Invalid score value: {score}"}

    return {"score": score, "rationale": rationale}


def judge_response(judge_provider, judge_params: dict, prompt_meta: dict,
                   response: str, auto_checks: dict) -> dict:
    """Score a response using an LLM judge.

    Returns {"judge_score": int|None, "judge_rationale": str}.
    Never raises - catches all exceptions so a judge failure won't crash the eval.
    """
    try:
        user_msg = build_judge_prompt(prompt_meta, response, auto_checks)
        content, _usage = judge_provider.complete(user_msg, judge_params)
        result = parse_judge_response(content)
        return {
            "judge_score": result["score"],
            "judge_rationale": result["rationale"],
        }
    except Exception as e:
        return {
            "judge_score": None,
            "judge_rationale": f"Judge error: {e}",
        }
