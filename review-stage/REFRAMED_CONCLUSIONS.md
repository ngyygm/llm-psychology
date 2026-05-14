# Reframed Conclusions

## What We Show (narrowed claims)

1. **Poor internal consistency**: IPIP domain Cronbach's α ranges from -0.02 to 0.36 in LLMs, compared to 0.87-0.90 in human normative samples. This demonstrates that items within each Big Five domain do not cohere in LLM responses.

2. **Reduced factor structure**: Under pooled role-conditioned questionnaire administration, canonical human trait structure is not recovered. Both domain-level EFA (3 factors) and item-level SVD analysis (poor domain recovery, all φ < 0.5) show that LLM responses do not organize into the expected Big Five pattern.

3. **Reverse-item inconsistency**: 58.4% [95% CI: 53.5%, 63.6%] of forward-reverse item pairs show inconsistent raw responses. This is an **upper bound** under a single administration at temperature = 0.7; the true inconsistency may be lower or higher under different decoding conditions.

4. **Acquiescence as associated mechanism**: Reverse-item agree rate is strongly associated with PIR (r = 0.726). Models that agree more with reverse items show higher inconsistency. However, this is correlational evidence; a clean decomposition of acquiescence from stochastic decoding variance is not possible with single-administration data.

5. **Negligible inter-model variance**: Model identity explains only 0.3% [1.1%, 4.0%] of Likert response variance. Cross-model personality comparisons are not meaningful at this measurement resolution.

6. **Failed convergent validity**: The central Extraversion × Sociability relation fails (r = -0.09, CI includes zero). 7/8 pairs show correct sign direction, but magnitudes are substantially weaker than human baselines.

## What We Do NOT Show (acknowledged limitations)

- We do NOT demonstrate that alignment training specifically causes these artifacts
- We do NOT rule out stochastic decoding as a co-contributor to PIR
- We do NOT have human or synthetic baselines run through the same pipeline
- We do NOT have multi-seed or multi-temperature data
- We do NOT have base vs instruct model comparisons
- The safety-null analysis is exploratory; we cannot rule out broader alignment effects

## Core Contribution (negative result)

Validated human psychometric instruments do not transport cleanly to LLMs. The observed response patterns are better explained by response-style contamination (particularly acquiescence bias) than by genuine trait-like personality structure. Future work using these instruments with LLMs must first establish measurement invariance and account for response style.
