
Paper Summary:
This paper investigates whether human personality questionnaires validly measure personality traits in LLMs, or merely elicit plausible-looking trait profiles. The authors administer a 221-item battery spanning four instruments to 20 instruction-tuned LLMs under a default prompt and 16 MBTI persona prompts, totaling 375,700 responses. They evaluate several psychometric diagnostics including internal consistency, reverse-item consistency, exploratory factor analysis, convergent validity, variance decomposition, and prompt-robustness analysis.

The main finding is that standard self-report personality instruments behave poorly on these LLMs. The paper reports substantial reverse-item inconsistency, severe reliability degradation in reverse-heavy domains, collapse of the expected Big Five structure into three factors, weak cross-instrument convergence for some constructs, and very small inter-model variance. However, persona prompts strongly shift responses, with profile shifts exceeding 1 SD on average and prompted MBTI types recovered with 99.7% fidelity. The authors interpret this as evidence that LLMs can enact personas, but questionnaire-derived scores do not provide valid measurement of stable latent personality traits.

The reviewer finds the paper important and ambitious, likely to influence future work on LLM personality evaluation and social simulation. The main reservation concerns the statistical framing of several central analyses, where some analyses are nonstandard or mislabeled, and certain claims are stated more strongly than the design can fully support.

Summary Of Strengths:
The paper addresses an important and timely question at impressive scale. The study covers 20 models, 4 instruments, 17 prompt conditions, and 375,700 responses. This breadth gives the paper substantial value even apart from any single analysis choice.

The empirical audit is much more comprehensive than simple score reporting. The paper evaluates internal consistency, reverse-item inconsistency, factor structure, convergent validity, and prompt sensitivity/persona steering. That broader psychometric perspective is a real contribution.

The reverse-item inconsistency evidence is clear and central. The PIR result in Section 4.3 is easy to interpret and directly relevant to whether these questionnaires are functioning as intended. The link between reverse-item agreement and PIR (Figure 4; Table 4), together with the reliability collapse in reverse-heavy domains (Table 3), gives the paper a coherent empirical story.

The paper usefully separates persona-following from valid trait measurement. The persona results are informative: MBTI prompts produce large, structured shifts and near-perfect prompt recoverability (Section 4.8) yet they do not restore forward/reverse consistency or the expected factor structure. This distinction is practically important for role-playing agents and simulation work.

The paper has clear relevance for downstream uses of LLM “personalities.” The discussion of silicon samples and persona-conditioned simulations is well motivated. Even without directly testing a downstream simulation, the paper raises an important measurement-validity warning for those applications.

Summary Of Weaknesses:
Insufficient differentiation from Sühr et al. (2025) (Section 2.1). The core findings that acquiescence on reverse-coded items and failure to replicate five-factor structure, were independently reported by Sühr et al. (2025) on BFI-2 with GPT-4/3.5/Llama-2. While the submitted paper extends this substantially in scale and scope, it does not clearly articulate what is conceptually new versus a scaled replication. A dedicated paragraph explicitly enumerating the overlap and novel extensions would help reviewers assess incremental contribution.

Causal attribution to alignment without base-model evidence. The paper repeatedly attributes acquiescence to alignment training ("alignment training creates the same acquiescence in every model") while acknowledging in limitations that no base-model comparison exists. Pre-training data distributions (containing abundant agreeable, polite text) could produce similar patterns independently of RLHF. The inconsistency between confident main-text claims and honest limitations caveats weakens credibility. At minimum, the causal language should be softened throughout.

Variance decomposition methodology is underspecified. The paper presents SS_total = SS_model + SS_domain + SS_persona + SS_item + SS_residual as "marginal components" while Table 6's caption calls it "sequential variance decomposition." With non-orthogonal, nested factors (items within domains), entry order matters for sequential decomposition, and marginal components don't sum to 100%. This ambiguity directly affects the headline "<1% model variance" claim.

Key statistical claims rest on minimal data points (Section 4.2). The Spearman ρ = −1.0 between α and reverse-item percentage is computed on only 5 data points (5 IPIP domains). While the pattern is descriptively clear, presenting this as a strong statistical result (p < 0.01) overstates the evidence. The correlation should be computed across all 17 domains or presented as a descriptive observation.

5.Missing engagement with contradictory published results (Section 2.1). Serapio-García et al. claims psychometrically valid personality measurement is possible in some LLMs. The submitted paper directly contests this but never explains why the two studies reach opposite conclusions. Without addressing this discrepancy, the paper's negative findings may be attributed to methodological choices rather than genuine invalidity.

Dense and difficult-to-parse key visualizations. Figure 1 is overly cluttered with tiny text and implicit reading order. Figure 2 attempts to show the entire diagnostic pipeline in a single dense figure. Table 4 (20 rows × ~12 columns) is extremely hard to parse without visual aids. These presentation issues may impede reader comprehension.
reference
Sühr, T., Dorner, F. E., Salaudeen, O., Kelava, A., & Samadi, S. (2025). Stop evaluating ai with human tests, develop principled, ai-specific tests instead. arXiv preprint arXiv:2507.23009.

Serapio-García, G., Safdari, M., Crepy, C., Sun, L., Fitz, S., Romero, P., ... & Matarić, M. (2025). A psychometric framework for evaluating and shaping personality traits in large language models. Nature Machine Intelligence, 1-15.

Comments Suggestions And Typos:
See the weaknesses part.

Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 2.5
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
