# Conference Guidelines

Target venue: ***ACL Anthology conference** (e.g., ACL / EMNLP / NAACL / EACL — current paper draft is anonymous-review ACL style).

> The authoritative formatting specification is `formatting.md` in this directory; this document summarizes only the constraints PaperOrchestra needs at planning time.

## Submission Format

- **LaTeX template:** `template.tex` (loads `acl.sty` with `[review]` option for anonymous review and includes `acl_natbib.bst` for citations). Do not switch to a different style file.
- **Bibliography style:** `acl_natbib`. Use `\citep{}` for parenthetical citations and `\citet{}` for textual citations. Authors must merge any project-specific entries into `custom.bib` (already present) — never edit `anthology.bib.txt`.
- **Fonts and packages:** Times Roman, `microtype`, `booktabs`, `graphicx` (`\graphicspath{{figures/}}`), `subcaption`, `multirow`, `array`, `tabularx`, `adjustbox`, `placeins`, `cleveref`, `enumitem`. Already configured in `template.tex`; do not remove.

## Page Limits (long-paper track)

| Stage | Content pages | Allowance |
|---|---:|---|
| Review version | **8** | unlimited references |
| Final / camera-ready | **9** | unlimited acknowledgements + references |

- All main-text figures and tables must fit inside the content limit.
- Appendices are allowed but must follow `\appendix` after `\bibliography{}` and `\clearpage`. They are not counted toward the content limit.
- Review versions must be self-contained — reviewers are not obliged to read appendices or supplementary material.

## Anonymity (review version)

- No author names, affiliations, or URLs in the review version. The `[review]` option to `\usepackage{acl}` enforces this; keep it.
- Self-references must be rewritten (no "we previously showed (Smith, 2023)" with author identity). Cite past work in third person.
- No acknowledgements section in the review version.
- Project URLs / repos / data sources that would deanonymize the authors must not appear in the review version. The compiled paper currently writes "Anonymous Authors"; preserve that.

## Mandatory Sections

In order:

1. `\title{}` and `\author{Anonymous Authors}` (one block)
2. `\begin{abstract} ... \end{abstract}` — single paragraph, ~200–250 words preferred.
3. `\section{Introduction}` — motivation, prior-work gap, contributions; at least one teaser figure is conventional but optional.
4. `\section{Related Work}` — substantive; for this venue, ~1.5 pages with structured paragraphs is appropriate.
5. `\section{Methodology}` — instruments / setup / equations.
6. `\section{Experiments and Results}` (or just `\section{Experiments}`) — every reported number must be traceable to a CSV in `code/results/` or to a tagged equation in the methodology.
7. `\section{Conclusion}` — restates the causal chain or central finding; can include a brief "what should researchers do" prescription.
8. `\section{Limitations}` — **required by ACL/EMNLP venues; do not omit.** 1 paragraph is acceptable.
9. `\section*{Ethics Statement}` — required when LLMs / human subjects / sensitive data are involved.
10. `\bibliography{refs}` (or `\bibliography{refs,custom}` if drawing from custom.bib).
11. `\clearpage \appendix` — followed by all supplementary sections (`\section{Persona Steering Supplement}`, `\section{DIF Across Model Families}`, `\section{Missing-Data Handling}`, `\section{Within-Sample Response Consistency}`, `\section{Full Refusal Responses}` in the current draft).

## Tables and Figures

- Use `booktabs` rules (`\toprule \midrule \bottomrule`); avoid vertical rules.
- Wide tables that overflow the column use `\begin{table*}` and `\resizebox{\columnwidth}{!}{...}` if compressed.
- Figures must be placed near their first reference; use `\FloatBarrier` (from `placeins`) where ordering matters.
- Caption format: short title-case caption followed by a 1–2 sentence explanation. Captions go **below** figures and **above** tables.
- All figures are PNG or PDF; vector PDF preferred for line plots, PNG acceptable for heatmaps.

## Citation Norms for This Paper

- Use `\citep{}` for parenthetical citations and `\citet{}` for narrative citations. Avoid `\cite{}`.
- All citations must come from `refs.bib` (project bibliography) or `custom.bib` (one-off entries the literature-review agent has verified through Semantic Scholar). Do **not** introduce entries without a verified DOI or arXiv ID.
- Honour the 90% verified-citation rule that PaperOrchestra applies — every cited work in Introduction / Related Work must appear in the verified pool produced by the literature-review-agent.

## Numeric Reporting Conventions

- Report Cronbach's α to 3 decimal places with SD where applicable.
- Report correlation coefficients to 2 decimal places.
- Report p-values as `p < 0.0001` if below threshold, otherwise to 3 decimal places.
- Effect sizes (Cohen's d, β) to 2 decimal places.
- Sample sizes always shown explicitly (e.g., n=20, n=320).

## Reproducibility

The paper must reference the public code/data location only after acceptance. In the review version, refer to "anonymous repository" or describe contents in prose without a URL. Hugging Face dataset link (`heihei/llm-psychology-raw-data`) must be replaced with an anonymized placeholder in the review version.

## Hard Stops

- Do not exceed 8 content pages in the review version.
- Do not include author names or institutional URLs.
- Do not drop the Limitations or Ethics Statement sections.
- Do not cite a work that does not appear in `refs.bib` or `custom.bib`.
