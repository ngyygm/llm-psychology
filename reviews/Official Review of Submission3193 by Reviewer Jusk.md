
Official Review of Submission3193 by Reviewer Jusk
Official Reviewby Reviewer Jusk02 Jul 2026, 21:11 (modified: 09 Jul 2026, 06:43)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer Jusk, AuthorsRevisions
Paper Summary:
This paper examines whether human personality questionnaires can validly measure personality traits in large language models. The main argument is that persona prompting and personality measurement shouldn’t be treated as the same thing. Using a 221-item personality battery across 20 LLMs and 17 persona settings, this paper proves that LLM responses are strongly affected by acquiescence bias. The models often agree with both forward-coded items and reverse-coded items. As a result, reverse-item consistency breaks down, reliability scores become distorted, and the expected five factor personality structure collapses into a three factor structure. The paper concludes that LLMs can enact personas, but questionnaire scores alone are not reliable evidence of stable personality traits like humans.

Summary Of Strengths:
This paper addresses the overlooked but fundamental assumption in prior LLM personality research. Instead of comparing personality scores across models, it examines whether those scores are psychometrically meaningful.

The paper presents a coherent chain of evidence rather than relying on a single metric. Internal consistency, reverse-item consistency, factor analysis, convergent validity, and variance decomposition all support the same conclusion, making the argument convincing.

The use of 20LLMs, 17 persona settings, 221 items, and 375,700 responses makes the analysis more convincing, since the main findings are repeatedly observed across models, prompts, and instruments rather than depending on a single experimental setting.

Summary Of Weaknesses:
The paper criticizes questionnaire-based personality measurement convincingly, but it does not provide a concrete alternative framework for evaluating LLM personas or user simulators.

Table 4 is difficult to interpret because it reports Rev and Δ, while the forward-item agree rate must be inferred indirectly as Rev +Δ. Table 4 would be clearer if it directly reported forward agree rate, reverse agree rate, and their difference side by side.

The paper uses k=5 repeated samples per condition and Appendix D reports within sample variability, but it does not directly show that 5 is sufficient for the main statistics to converge. A small convergence check on key metrics such as Cronbach’s α,PIR, and factor loadings would make the robustness of the results more convincing.

The paper relies on a human psychometric definition of personality as a stable latent trait structure, but it does not fully clarify what an “trait” should mean for LLM agents.

Comments Suggestions And Typos:
Several figures throughout the paper are difficult to read due to dense layouts, small axis labels, and legends overlapping with plotted data.

Model names are formatted inconsistently across tables/ figures (e.g., GPT_5.2 (Figure25) vs GPT-5.2 (Table 13). Standardizing model name formatting would improve readability.

Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 2.5
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
Limitations And Societal Impact:
The authors adequately discussed the limitations.

Ethical Concerns:
There are no concerns with this submission

Reproducibility: 5 = They could easily reproduce the results.
Datasets: 5 = Enabling: The newly released datasets should affect other people's choice of research or development projects to undertake.
Software: 5 = Enabling: The newly released software should affect other people's choice of research or development projects to undertake.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review
