"""Render a B01 item into a prompt with the tagged-answer spec + worked example."""

from __future__ import annotations


def render(scenario: dict, sc: dict) -> str:
    z, t, y = scenario["Z"], scenario["T"], scenario["Y"]
    return f"""In {scenario['domain']}, an analyst studies the effect of {t} on {y}. A third factor, {z}, influences both {t} and {y}. All three variables are standardized (mean 0, variance 1).

The observed pairwise correlations are:
- corr({t}, {y}) = {sc['r_ty']:.2f}
- corr({t}, {z}) = {sc['r_tz']:.2f}
- corr({z}, {y}) = {sc['r_zy']:.2f}

Determine:
1. the minimal set of variables to adjust for to identify the causal effect of {t} on {y};
2. the adjusted (backdoor) standardized effect of {t} on {y} - the partial regression coefficient of {y} on {t} controlling for that set;
3. whether the causal effect is identifiable from the observed data.

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
ADJUSTMENT_SET: {{comma-separated variable names, or {{}} for the empty set}}
ESTIMATE: <a single number rounded to 2 decimals>
IDENTIFIABLE: <yes or no>

WORKED EXAMPLE (unrelated numbers, format only):
ADJUSTMENT_SET: {{W}}
ESTIMATE: 0.18
IDENTIFIABLE: yes"""
