# Paper Improvement Log

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (original) | — | — | Baseline: first complete draft |
| Round 1 | 6/10 | Almost | C1/C3: Added measurement framework caveat. C2: Softened mechanism claims. M1: Expanded ICC discussion. M2: Added SiliconFlow caveat. M4: Elevated human baseline. m3: Temperature justification. |
| Round 2 | 7/10 | Accept | Added HEXACO-H dimensional structure sentence. All CRITICAL/MAJOR issues addressed. |

## Round 1 Review & Fixes

<details>
<summary>MiniMax-M2.7 Review (Round 1) — Score: 6/10 (Almost)</summary>

### Critical Issues
- C1: Construct validity unestablished for LLM respondents — instruments lack validation for non-human respondents
- C2: Mechanism attribution exceeds evidence — "alignment training" claims unsupported by data
- C3: Framing conflates human psychology and LLM behavior — oscillates between two framings

### Major Issues
- M1: ICC (.22-.46) implies substantial within-model variance not adequately discussed
- M2: Generalization scope poorly bounded (Chinese models, single API)
- M3: Scale usage interpretation unclear (2-3.5 range on 1-5 scale)
- M4: Human baseline absence understated

### Minor Issues
- m3: Temperature=0.7 unjustified

</details>

### Fixes Implemented
1. **C1/C3 (Construct validity + framing)**: Added explicit measurement caveat: "We use these instruments as standardized probes for detecting response variation; we do not assume they measure identical constructs in LLMs and humans." Consistent "systematic response variation" framing throughout.
2. **C2 (Mechanism overclaim)**: Replaced "alignment training" → "model-specific training dynamics" everywhere. Added: "we cannot disambiguate alignment from architecture, data, or scale effects."
3. **M1 (ICC)**: Expanded: "ICC(1,1) values of .22-.46 indicate moderate within-model consistency: while models differ systematically, individual model instances show considerable variation across seeds."
4. **M2 (Generalization)**: Added SiliconFlow caveat in limitations.
5. **M4 (Human baseline)**: Elevated to first limitation item with strong framing.
6. **m3 (Temperature)**: Added justification: "chosen to reflect typical deployment settings while allowing within-model variance estimation."

## Round 2 Review & Fixes

<details>
<summary>MiniMax-M2.7 Review (Round 2) — Score: 7/10 (Accept)</summary>

All CRITICAL and MAJOR issues adequately addressed. Consistent framing, honest limitations, appropriate hedging. Only MINOR suggestions remain: brief "Contribution in Context" paragraph, figure placement verification, one additional HEXACO-H sentence.

Verdict: Accept (high confidence)

</details>

### Fixes Implemented
1. Added HEXACO-H dimensional structure sentence: "the dimensional structure of LLM responses may differ fundamentally from human personality structure, warranting future work on LLM-native psychometric frameworks."
2. Skipped "Contribution in Context" paragraph to stay within 4-page limit.

## Format Check (Final)

- **Pages**: 4 (within EMNLP Findings limit)
- **Overfull hbox**: 0
- **Undefined references**: 0
- **Undefined citations**: 0
- **Errors**: 0

## PDFs
- `main_round0_original.pdf` — Original generated paper
- `main_round1.pdf` — After Round 1 fixes (6/10 → 7/10)
- `main_round2.pdf` — Final version after Round 2 (7/10, Accept)
- `emnlp2026_improved.pdf` — Current working copy (= main_round2.pdf)
