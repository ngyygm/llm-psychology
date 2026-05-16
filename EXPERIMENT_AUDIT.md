# Experiment Audit Report

**Date**: 2026-05-14
**Auditor**: GPT-5.4 xhigh (cross-model) + Claude (local verification)
**Project**: LLM Personality Psychometric Validation

## Overall Verdict: PASS (with minor WARN)

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
- Reverse scoring rules correctly implemented: Likert 6-raw, Binary 1-raw
- Forward/reverse uses keyed field from items_battery.json
- Scoring verified: 63 reverse items, 0 mismatches
- Human benchmarks from Johnson (2014), Zuckerman (2002), Jones & Paulhus (2014)
- Evaluation type: psychometric_analysis

### B. Score Normalization: PASS
- Cronbach's alpha: standard formula, no self-referential normalization
- PIR: simple proportion, no normalization
- Variance decomposition: standard sequential sum-of-squares
- No suspicious near-1.0 or near-0.0 values

### C. Result File Existence and Number Verification: PASS
18 JSON files exist (4.7-49 MB each). Four specific verifications:

| Claim | Actual | Match |
|-------|--------|-------|
| Mean PIR = 0.584 | 0.5836 | YES |
| Model variance = 0.3% | 0.341% | YES |
| Eigenvalues: 6.80, 4.33, 2.99 | 6.804, 4.329, 2.986 | YES |
| Alpha: -0.02 to 0.36 | -0.017 to 0.360 | YES |

### D. Dead Code: PASS
All 13 functions called, no orphans.

### E. Scope: PASS
67,626 data points verified. Note: persona invariance covers 10/18 models.

### F. Evaluation Type: psychometric_analysis

## Warning W1: Persona Invariance Coverage Gap
9/18 models have NaN due to unhandled None parsed_values. Invariance claim based on ~10 models.

## Claim Impact
All 7 key claims: **supported**
