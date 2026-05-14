# Conference Guidelines (ACL 2026 Findings)

## Submission deadline

June 15, 2026.

The Literature Review Agent should derive `cutoff_date = 2026-06-15` from this deadline. Papers published after this date may be cited only as concurrent work.

## Page limit

The main paper is limited to **8 pages** of single-column text, excluding references and appendices. The appendix may be up to 4 additional pages. Reviewers are not obligated to read past the page limit.

## Mandatory sections

The submission MUST contain, in this order:

1. Abstract (single paragraph, ~150-250 words)
2. Introduction
3. Related Work
4. Methodology (or Method)
5. Experiments and Results
6. Discussion
7. Limitations
8. Conclusion
9. References
10. Appendix (optional)

## Formatting rules

- Use the ACL 2026 LaTeX template (`acl2026.sty` provided in `template.tex`)
- Single-column, 11pt font
- Citations via natbib: `\citep{...}` for parenthetical, `\citet{...}` for textual
- Figures saved at 300 DPI minimum, placed in-text or at top/bottom of page
- Tables use booktabs package (`\toprule`, `\midrule`, `\bottomrule`)
- All figures and tables MUST appear before the Conclusion section
- Anonymized for double-blind review: no author names, affiliations, or acknowledgements
- Ethics statement required if the work involves human subjects or sensitive data

## Topic scope

ACL 2026 welcomes submissions on NLP evaluation, LLM behavior analysis, psycholinguistics, computational social science, and AI safety. This work on psychometric validation of LLM personality measurement fits the "Analysis and Evaluation of LLMs" track.

## Review criteria

Reviewers will assess: soundness (methodology rigor), novelty, clarity, significance (practical/theoretical impact), and reproducibility (implementation details, seeds, model access reported).

## Language

English. The paper should be written in clear academic English.
