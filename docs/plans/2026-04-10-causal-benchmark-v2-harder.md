# Plan: Causal Benchmark v2 - Making It Much Harder

## Current State

Haiku 3 (cheapest, oldest Anthropic model) scored **86/100 (86%)** on the bundled causal benchmark v1. That means frontier models will likely score 95%+, giving almost no differentiation. The goal is to get Haiku down to 30-40% so stronger models land in the 50-70% range - creating real separation.

## What We Learned from Haiku's Results

### Variant difficulty (clear gradient exists)
| Variant | Haiku Score | Assessment |
|---|---|---|
| Flip | 20/20 (100%) | Useless for differentiation - remove or redesign |
| Base | 19/20 (95%) | Too easy in current form |
| Transfer | 18/20 (90%) | Some signal, needs harder domain shifts |
| Numeric | 17/20 (85%) | Better - calculations help, but need to be harder |
| Analyst | 12/20 (60%) | Best differentiator - "which analyst is right" works |

### Bundle-level patterns
- 10/20 bundles scored 5/5 (perfect) - these concepts are too easy
- B15 (mentoring/projects) was hardest at 2/5 - multi-step reasoning with post-treatment control
- B17 (handwashing campaign) at 3/5 - natural experiment reasoning
- The analyst variant (Q5) accounted for 8 of 14 wrong answers

### Key failure modes
- Haiku defaults to answer C when confused (9 of 14 wrong answers were C)
- Analyst debates where both analysts sound reasonable but differ subtly - Haiku picks the more "balanced" sounding one
- Numeric questions where the right answer requires rejecting an intuitive but wrong calculation
- Multi-step causal chains where conditioning on a later variable interacts with an earlier selection problem

## Design Principles for v2

### 1. Every question must have a plausible wrong answer that sounds more sophisticated

The current wrong options are often obviously wrong ("the data are unusable", "this is impossible"). Replace with wrong answers that demonstrate partial understanding - the kind of answer a student who read one textbook chapter would give. The correct answer should require deeper reasoning that goes against the "textbook" instinct.

### 2. Eliminate the flip variant - replace with "trap" variant

Flip (inverted scenario) was 100% - models just pattern-match "randomisation fixes it". Replace with a **trap** variant: a scenario that *looks* like the base concept applies but actually doesn't, or where the obvious causal concern is a red herring and the real issue is something else. This forces the model to reason about when a principle does NOT apply.

### 3. Make analyst debates much harder

Analyst is already the best differentiator (60%). Make it harder:
- Both analysts should cite real methodological principles
- The wrong analyst should use correct reasoning applied to the wrong part of the problem
- Add analysts who are "right for the wrong reason" vs "wrong but closer to the right framing"
- Include cases where the answer is "both are wrong" or "both raise valid but incomplete concerns"

### 4. Numeric questions should require multi-step calculation and table reading

Current numeric questions have one calculation step. v2 should:
- Present data tables where Simpson's paradox is hidden across 3+ subgroups, not 2
- Require calculating conditional probabilities, not just comparing rates
- Include scenarios where the numbers support the wrong conclusion unless you notice a compositional shift
- Force the model to identify which comparison is meaningful (within-group vs aggregate) from ambiguous framing

### 5. Transfer should cross into unfamiliar domains

Current transfers go from business to medical or education. v2 should transfer into:
- Ecology (predator-prey observation studies)
- Social media algorithm evaluation (engagement metrics as colliders)
- Sports analytics (selection on performance for team analysis)
- Criminal justice (sentencing data with arrest as a collider)
- Climate policy (emissions attribution with economic confounders)
These domains have less training data for causal reasoning, forcing actual understanding over memorisation.

### 6. Add "which is the LEAST defensible" questions

Instead of "which concern is strongest" (positive identification), ask "which of these four concerns is LEAST relevant" where three distractors are genuine concerns and one is subtly wrong. This requires evaluating all four options rather than recognising one correct pattern.

### 7. Multi-step synthesis should be the norm, not the exception

B15 (2/5) and B17 (3/5) were hardest because they combined multiple biases. v2 should make every base question involve at least two interacting causal issues (e.g. selection + post-treatment, confounding + mediator, collider + attrition). Single-concept questions become the easy warm-up at most.

## Proposed v2 Structure

### 100 questions, 20 bundles, 5 variants each

| Variant | Name | Design | Target Haiku accuracy |
|---|---|---|---|
| Q1 | Base | Multi-step scenario combining 2-3 causal issues | 50% |
| Q2 | Trap | Looks like the base concept but the obvious answer is wrong | 30% |
| Q3 | Transfer | Same reasoning, obscure domain (ecology, sports, criminal justice) | 40% |
| Q4 | Numeric | Multi-step calculation with tables, conditional probabilities | 35% |
| Q5 | Analyst | Two sophisticated analysts, both partially right, subtle distinction | 25% |

**Target overall Haiku accuracy: 35-40%**

### Bundle concept families (20)

Keep the 10 concepts Haiku got 5/5 on but make them multi-step:

1. Confounding + selection interaction (was: baseline confounding)
2. Post-treatment mediator + attrition (was: pre/post-treatment)
3. Collider + downstream selection (was: selection among finalists)
4. Pre-trend + compositional shift (was: pre-trend comparison)
5. Self-selection + post-adoption behaviour (was: self-selection)
6. Retention selection + survivor conditioning (was: retention selection)
7. Mediator chain + baseline confounding (was: mediator homework)
8. Confounding by indication + follow-up attrition (was: confounding by indication)
9. Selected sample + multiple post-treatment controls (was: admitted + seminar)
10. Simpson's paradox with 3+ subgroups + compositional confounding (was: composition shift)

Add 10 new harder concepts:

11. Berkson's bias in observational health data
12. M-bias (adjusting for a pre-treatment collider)
13. Noncompliance + instrument validity
14. Interference/spillover effects between units
15. Measurement error + differential misclassification
16. Time-varying confounding in longitudinal studies
17. Immortal time bias in survival analysis
18. Ecological fallacy across aggregation levels
19. Regression to the mean masquerading as treatment effect
20. External validity failure (valid internal estimate, wrong population)

### Wrong answer design

Every wrong option should fall into one of these categories:
- **Partial insight**: Identifies one real issue but misses the more critical one
- **Reversed logic**: Correct concept, applied backwards (e.g. "adjusting fixes it" when adjusting creates the problem)
- **Textbook reflex**: The answer a first-year stats student would give (e.g. "randomise" when randomisation doesn't help here)
- **Sophisticated distractor**: Uses advanced terminology correctly but applies it to the wrong part of the problem

### Scoring additions for v2

1. **Pair consistency** - track which bundles a model gets 5/5, 4/5, etc.
2. **Trap detection rate** - how often models avoid the trap in Q2
3. **Calculation accuracy** - for numeric Q4, track whether the error was conceptual or arithmetic
4. **Weakest variant** - per model, which variant type is the consistent failure mode

## Implementation Steps

1. Draft the 20 concept descriptions with the multi-step causal structure for each
2. For each concept, write all 5 variants simultaneously (ensures internal consistency)
3. Have each wrong answer explicitly tagged with its distractor type (partial insight, reversed logic, textbook reflex, sophisticated distractor)
4. Pilot on 3 models (Haiku 3, a mid-tier model, a frontier model) to verify the difficulty gradient
5. Adjust questions where all 3 models get the same answer (too easy or too ambiguous)
6. Finalise answer key and scoring guide

## What NOT to Do

- Don't just make questions longer or more jargon-heavy - that tests reading comprehension, not causal reasoning
- Don't make correct answers ambiguous - each question should have one clearly defensible best answer
- Don't add "trick" formatting (unusual option ordering, double negatives) - test reasoning, not attention
- Don't reuse scenarios from v1 with minor wording changes - models may have seen these patterns in training data
