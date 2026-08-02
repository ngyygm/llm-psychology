# Citation Audit Report — "Acquiescence Is Not Personality"

**Date:** 2026-07-31
**Audited files:** `paper-orc/final/refs.bib` (81 entries), `paper-orc/final/paper.tex`
**Submitted bibliography:** `refs.bib` (the only file loaded via `\bibliography{refs}`). `custom.bib` is the stock ACL template (Aho, Chandra, Gusfield, etc.) and is **not cited** — irrelevant.
**Method:** every recent (2022–2026) entry independently verified against arXiv, DOI, ACL Anthology, NeurIPS/AAAI proceedings, PubMed, DBLP, Semantic Scholar, and publisher pages. Classic psychometrics/statistics/ML entries (Cronbach, Kaiser, Efron, InstructGPT, DPO, etc.) confirmed by canonical recognition.

---

## Executive summary

**The bibliography was machine-generated and never genuinely verified.** The file header states it directly: *"Compiled by the literature-review-agent (Step 3 of PaperOrchestra)... new entries verified via Semantic Scholar."* That verification step failed systematically.

The defect pattern is a textbook **LLM-hallucinated bibliography**: the generator retrieved real paper *titles* and *DOIs/arXiv IDs* but fabricated plausible **author names**, and invented a handful of **phantom papers** outright.

**Scope is larger than the 9 citations flagged by the ACL RR editors.** Of the entries actually *cited* in the paper:

| Defect class (cited entries) | Count |
|---|---|
| Fully **fabricated** (no such paper exists) | **3** |
| Real paper, **fabricated/wrong authors** | **8** |
| Real paper, **wrong metadata** (arXiv ID, year, given names, venue) | **~8** |
| **Correct** | ~51 |

So **~19 of the ~70 cited references are defective**, including **3 phantom papers** and **8 with hallucinated author lists**. The editors caught 9; this audit finds **at least 10 more** defective cited entries they did not list.

---

## CRITICAL — act before any resubmission or editor correspondence

### REMOVE — fabricated (no such paper exists)

#### `schaekermann2024personality` — ❌ NO SUCH PAPER
- As cited: Schaeckermann, Chuang, Zecevic & Hahn (2024), *CogSci*.
- Reality: no paper with this title, author set, or venue exists in any index.
- Used in text: `\citet{schaekermann2024personality}` — "report partial reliability checks."
- **Action:** remove citation; re-source the claim or delete it.

#### `hagendorff2024machine` — ❌ PHANTOM + WRONG arXiv ID
- As cited: Hagendorff (2024), "Machine Personality Traits: A Test of Psychometric Properties," arXiv:2402.11596.
- Reality: arXiv:2402.11596 is *"Faster Algorithms on Linear Delta-Matroids"* by Koana & Wahlström (math). No such Hagendorff title exists. Hagendorff's real adjacent work is *"Machine Psychology,"* arXiv:2303.13988 (2023).
- Used in text: `\citet{hagendorff2024machine}` — "report partial reliability checks."
- **Action:** remove. If the claim ("partial psychometric-reliability checks on LLMs") must be supported, consider Hagendorff's real *"Machine Psychology"* (2303.13988) — but only after confirming it actually makes that claim.

#### `serpell2024validity` — ❌ NO SUCH PAPER
- As cited: Serpell & Ney (2024), "Construct Validity in Computational Social Science," arXiv preprint (no ID).
- Reality: no such paper or author pair exists. (A real, *different* paper is Bean et al. 2025, "Construct Validity in LLM Benchmarks," arXiv:2511.04703 — but it is about benchmarks, not the claim made here.)
- Used in text: `\citep{serpell2024validity,laverghetta2022predicting}` — "measures the alignment layer."
- **Action:** remove citation; the sentence still stands on `laverghetta2022predicting`.

---

### FIX — real paper, fabricated/wrong authors (provide correct attribution)

#### `acerbi2024llm` — duplicate of `salecha2024llm`, wrong authors
- As cited: Acerbi & Stubbersfield (2024), PNAS Nexus pgae533.
- Reality: pgae533 is **Salecha, Ireland, Subrahmanya, Sedoc, Ungar & Eichstaedt** — i.e., the *same* paper as the (correct) `salecha2024llm`. Acerbi & Stubbersfield did not write it.
- **Done (merged):** `acerbi2024llm` removed from bib; both `\citet{acerbi2024llm}` uses in text repointed to `salecha2024llm`.

#### `pelet2025persistent` — author is Tosato, not Pelet
- Correct: Tosato, T. et al. (2025/26). "Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History." AAAI (v40i44, art. 41133); arXiv:2508.04826.
- Note: bib DOI `10.1609/aaai.v39i23.41133` is likely wrong (article is v40i44).

#### `chen2025humanizing` — author is Dong, not Chen
- Correct: Dong, W. et al. (2025). "Humanizing LLMs: A Survey of Psychological Measurements with Tools, Datasets, and Human-Agent Applications." arXiv:2505.00049.

#### `wang2024aligning` — authors are Barnhart et al., not Wang
- Correct: Barnhart, L.; Akbarian Bafghi, R.; Becker, S.; Raissi, M. (2025). "Aligning to What? Limits to RLHF Based Alignment." arXiv:2503.09025 (ACL 2025).

#### `acquiescence2025emnlp` — author is Braun, not "{Responsible NLP Group}"
- Correct: Braun, D. (2025). "Acquiescence Bias in Large Language Models." Findings of EMNLP 2025; arXiv:2509.08480. The anonymous group author is a hallucination.

#### `treaux2025forced` — authors are Li et al., not "Treaux"
- Correct: Li, X.; Shi, H.; Yu, Z.; Tu, Y.; Zheng, C. (2025). "Decoding LLM Personality Measurement: Forced-Choice vs. Likert." Findings of ACL 2025, pp. 9234–9247.

#### `zheng2024agreeableness` — authors are Shah et al., not Zheng
- Correct: Shah, A.; Mishra, D. et al. (2026). "Too Nice to Tell the Truth: Quantifying Agreeableness-Driven Sycophancy in Role-Playing Language Models." arXiv:2604.10733 (ACL 2026).

#### `xie2025aipsychobench` — author is Wei Xie, not "Yijin Xie"; venue is wrong
- Correct: Xie, W. et al. (2025). "AIPsychoBench: Understanding the Psychometric Differences between LLMs and Humans." *Topics in Cognitive Science* (Wiley), arXiv:2509.16530, DOI 10.1111/tops.70041. (Bib's "IEEE Trans. Privacy-Preserving and Security" is wrong.)

---

### FIX — real paper, wrong metadata (authors essentially correct)

#### `horton2023homo` — wrong NBER number
- Correct: Horton, J. J. (2023). "Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?" **NBER Working Paper No. 31122** (not 31194); arXiv:2301.07543.

#### `suh2024rediscovering` — wrong arXiv ID AND wrong given names
- Correct: arXiv:**2409.09905**; authors **Joseph Suh, Suhong Moon, Minwoo Kang** (the bib's 'Sangwon/Seongyong' were also fabricated). Fixed.

#### `cau2025language` — wrong given names
- Correct: **Cau, Elia** (not Erica); **Pansanella, Pietro** (not Valentina); Pedreschi, Dino. arXiv:2502.19098. (Surnames were right; first names hallucinated.)

#### `miotto2023gpt3` — year off by one
- Correct: year **2022** (arXiv:2209.14338; NLP4PSI @ EMNLP 2022), not 2023. Authors correct.

#### `lee2025trait` — wrong first-author given name
- Correct: first author **Seungbeen Lee** (not "Seolhwa"). Full title: "Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset designed for LLMs with Psychometrics." NAACL 2025 Findings; arXiv:2406.14703.

#### `chuang2025debate` — missing arXiv ID
- Correct: arXiv:**2510.25110**. First three authors **Chuang, Tu, Dai** verified against arXiv. Fixed.

#### `dominguez2024questioning` — wrong given name
- Correct: **Domínguez-Olmedo, Ricardo** (not "Jon"); with Hardt, M. NeurIPS 2024; arXiv:2306.07951.

#### `bodroza2024personality` — confirmed correct (no change needed)
- Third author is **Ljubiša Bojić** (RSOS / PubMed PMID 39386990). The bib was already right.

---

### Also defective but NOT cited (won't appear in reference list — fix before any reuse)

- `sorokovikova2024evaluating` — real paper (NeurIPS 2023, the MPI "Evaluating and Inducing Personality" paper) but author list is wrong (real first author is **Jiang**, not Sorokovikova). This is the editors' flagged #7.
- `huang2024reliability` — real paper (EMNLP 2024) but first author is **Jiaan Huang** (bib has "Evan").

---

## KEEP — verified correct (cited)

Core profiling/silicon-sample/simulation: `miotto2023gpt3` (yr fix only), `serapio2025psychometric`, `jiang2023personallm`, `huang2023psychobench`, `bodroza2024personality`, `argyle2023one`, `aher2023using`, `park2023generative`, `ferreira2025matching`, `chuang2023simulating`, `mannekote2025roleplaying`, `larooij2025validation`, `salecha2024llm`, `suhr2025challenging`, `kirk2023understanding`, `han2025personality`, `binz2023psychology`, `tjuatja2023response`, `laverghetta2022predicting`.

Alignment/RLHF: `ouyang2022training`, `christiano2017deep`, `rafailov2023dpo`, `bai2022constitutional`, `wei2022flan`, `perez2023discovering`, `sharma2024sycophancy`.

Model tech reports: `openai2023gpt4`, `gemini2024v15`, `deepseek2024v3`, `qwen2024qwen25`, `glm2024chatglm`, `moonshot2025kimi`, `minimax2025minimax01`.

Classic instruments: `johnson2014measuring`, `jones2014introducing`, `aluja2006cross`, `francis1992development` (+ uncited `zuckerman2002zuckerman`, `mccrae2005universal`, `mccrae1992five`, `john1999bigfive`, `soto2017bfi2`, `goldberg1992bigfive`, `digman1990personality`).

Classic statistics/psychometrics: `cronbach1951coefficient`, `couch1960yeasayers`, `crowne1960social`, `billiet2000modeling`, `paulhus1991response`, `kaiser1958varimax`, `kaiser1960application`, `horn1965parallel`, `tucker1951method`, `lord1968statistical`, `vandenberg2000review`, `meredith1993measurement`, `brown2011forcedchoice`, `efron1979bootstrap`, `mahalanobis1936generalised`. Uncited-but-correct: `bhandari2025evaluating`.

---

## Recommended sequence

**Status (2026-07-31): steps 1–4 complete.** `refs.bib` corrected in place, 3 fabricated entries removed, `acerbi2024llm` duplicate merged into `salecha2024llm`, all questioned given names verified against arXiv. Paper recompiles clean: **66 references, 0 undefined citations, 0 bibtex errors, 17 pages.**

1. **FIX corrections applied** to metadata/author entries in `refs.bib` (original submitted bibliography preserved as `refs.submitted.bib` and in git history; paper recompiled clean).
2. **Re-source or delete** the 3 fabricated citations (`schaekermann`, `hagendorff`, `serpell`) — requires reading what claim each was supporting.
3. **Merge the duplicate** `acerbi2024llm` → `salecha2024llm` in both bib and text.
4. **Recompile** and confirm zero undefined citations.
5. **Re-run this audit** on the corrected bib before any resubmission.
