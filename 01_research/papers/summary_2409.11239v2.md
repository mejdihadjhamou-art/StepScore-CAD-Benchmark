# Summary of 2409.11239v2

## Paper
**Title:** LLM-as-a-Judge & Reward Model: What They Can and Cannot Do  
**Authors:** Guijin Son, Hyunwoo Ko, Hoyoung Lee, Yewon Kim, Seunghyeok Hong  
**Version:** arXiv:2409.11239v2 (Oct 2, 2024)

## What the paper is about
This paper studies how reliable automated evaluators are, specifically:
- **LLM-as-a-Judge** (generative judging models), and
- **Reward Models (RMs)** (scoring models used in preference learning).

The core question is: *When can these evaluators be trusted, and where do they fail?*

To answer that, the authors build **KUDGE**, a bilingual (Korean/English) meta-evaluation benchmark with:
- **Pointwise** judging (score one answer),
- **Pairwise** judging (choose better answer), and
- A **Challenge subset** with harder STEM reasoning questions.

They evaluate many proprietary and open models and run targeted analyses for language transfer, bias, factual error detection, and hard-question judging.

## Main contributions
1. Introduce **KUDGE**, a bilingual meta-evaluation dataset for Korean and English.
2. Provide broad empirical evaluation of modern LLM judges and reward models.
3. Show where transfer to a new language works and where it breaks.
4. Show important failure modes in factual verification and difficult reasoning tasks.
5. Release dataset/code for further research.

## Key findings

### 1) Cross-lingual transfer works better than expected
- Evaluators trained/effective in English often transfer well to Korean judging.
- Performance on an English evaluator benchmark (RewardBench) predicts Korean judging performance better than Korean language benchmark proficiency alone.
- Implication: evaluator capability is partly language-agnostic.

### 2) Strong models still fail on factual/cultural error detection
- Even top proprietary judges struggled to consistently detect subtle factual errors and cultural misrepresentations.
- They were better when errors were large/obvious (e.g., paragraph-level corruption), worse for subtle word/sentence-level mistakes.
- In pairwise setups, detection improved somewhat, but models still produced fabricated/incorrect rationales in some cases.

### 3) Hard reasoning questions are a major weakness
- On challenge tasks (especially GPQA-like hard questions), performance drops significantly.
- Many models stay uncomfortably close to random-guessing territory.
- Reward models that do well on easy subsets can fail badly on hard ones.

### 4) Ensembling judges gives only limited gains
- Majority-vote/aggregation helps slightly, but not enough to beat top single models in their setup.
- High collinearity between judges limits diversity benefits.

## Methodological details worth noting
- They evaluate both **pointwise** and **pairwise** settings.
- They examine score distribution behavior, not just accuracy.
- They include targeted error probes (unwanted language artifacts, incomplete answers, factual corruptions).
- They analyze feature correlations and regression controls (including model-size-adjusted effects).

## Limitations discussed/visible from the study
- Evaluator reliability is highly context-dependent.
- Performance on standard benchmarks does not guarantee robust behavior under subtle factual/cultural perturbations.
- Current automated judges are weak on hardest reasoning scenarios.

---

## Why this paper is useful for your work (CAD benchmark context)
Although your project is CAD generation rather than text QA judging, the lesson is directly transferable: **automated evaluators can look strong on average while still failing in critical edge cases**. For your CAD benchmark, this supports building (1) a **core deterministic track** and a **challenge track**, (2) explicit failure probes (e.g., subtle dimensional errors, topology violations, culturally/domain-specific prompt ambiguity), and (3) **meta-evaluation of your own metrics pipeline** (alignment failure detection, disagreement flags, human spot-check loops). In short, the paper reinforces that robust benchmarking must include stress tests and calibration, not just headline average scores.
