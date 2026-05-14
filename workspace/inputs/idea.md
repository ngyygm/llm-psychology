## Problem Statement

When human-validated personality questionnaires (Big Five, Dark Triad, etc.) are administered to large language models (LLMs), a fundamental measurement problem arises: these instruments were designed for human respondents whose trait structure has been validated across decades of psychometric research. LLMs, however, do not have "personality" in the human sense — their responses are shaped by alignment training, safety constraints, and token-level next-word prediction. This creates systematic artifacts that invalidate standard psychometric interpretation.

Existing work (Sühr et al., 2025; Acerbi & Stubbersfield, 2024; Serapio-García et al., 2025) has documented isolated aspects of this problem — reverse-item inconsistency, social desirability bias, and reliability failures. But no study has systematically tested whether validated personality instruments **transport** to LLMs with their psychometric properties intact, using the full battery of psychometric validation methods (factor analysis, reliability, convergent validity, measurement invariance, response style decomposition).

## Core Hypothesis

We hypothesize that **human personality instruments do not transport cleanly to LLMs**. Specifically:

1. The Big Five factor structure (5 independent factors) will **collapse** to fewer factors, revealing that LLM responses are organized around alignment-driven dimensions rather than human-like traits.
2. Reverse-coded items will show systematic **acquiescence bias** — LLMs will agree with both forward and reverse items, producing high pairwise inconsistency rates (PIR).
3. Inter-model variance will be **negligible** (<1%), meaning that personality differences between LLMs are effectively noise — all aligned models converge on a similar "moderate prosocial" profile.
4. Social desirability response (SDR) bias will be the **primary driver** of response patterns, not genuine trait variation.

## Proposed Methodology

We administered a **221-item psychometric battery** comprising four validated instruments:

1. **IPIP-NEO-120** (Johnson, 2014): 120 Likert-5 items measuring Big Five domains × 6 facets each, with 41 reverse-coded items.
2. **SD3 Short Dark Triad** (Jones & Paulhus, 2014): 27 Likert-5 items measuring Machiavellianism, Narcissism, Psychopathy, with 5 reverse-coded items.
3. **ZKPQ-50-CC** (Aluja et al., 2006): 50 True/False items measuring 5 Zuckerman-Kuhlman traits, with 12 reverse-coded items.
4. **EPQR-A** (Francis et al., 1992): 24 Yes/No items measuring 4 Eysenck traits, with 5 reverse-coded items.

This battery was administered to **18 LLMs** (including GPT-5.5, GPT-5.2, DeepSeek-V4, Qwen3, Gemini-3 series, etc.) across **17 role-conditioned prompts** (Default + 16 MBTI personas), at temperature = 0.7, yielding **67,626 item-level responses**.

We then applied six psychometric validation analyses:
1. Exploratory Factor Analysis (EFA) with Kaiser criterion and parallel analysis
2. Pairwise Inconsistency Rate (PIR) with SDR cross-validation
3. Sequential variance decomposition (Model × Domain × Persona × Item)
4. Differential Item Functioning (DIF) across model families
5. Response style quantification (acquiescence, extreme response, midpoint)
6. Measurement invariance across persona conditions

## Expected Contribution

1. **First systematic psychometric transportability study**: We show that four independently validated personality instruments all fail basic quality checks when applied to LLMs, with Cronbach's alpha ranging from -0.02 to 0.36 (vs 0.87-0.90 in human norms).
2. **Factor structure collapse**: We demonstrate that the canonical Big Five 5-factor structure collapses to 3 factors in LLM data (robust across leave-one-model-out and leave-one-persona-out), providing the strongest evidence that LLM "personality" is structurally different from human personality.
3. **Acquiescence as mechanism**: We identify acquiescence bias (r = 0.726 with PIR) as the primary driver of reverse-item inconsistency, explaining 58.4% of forward-reverse pair failures.
4. **Negligible inter-model variance**: We quantify that model identity explains only 0.3% of Likert response variance, undermining the validity of cross-model personality comparisons.
5. **Methodological caution**: We provide a template for future researchers to validate psychometric instruments before applying them to LLMs, including reliability, factor structure, convergent validity, and measurement invariance checks.
