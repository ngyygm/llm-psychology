
Official Review of Submission3193 by Reviewer ys6N
Official Reviewby Reviewer ys6N14 Jul 2026, 08:07Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer ys6N
Paper Summary:
This paper examines whether human personality questionnaires provide valid measurements of personality when applied to large language models. The authors administer a 221-item psychometric battery across 20 LLMs and 17 persona prompting conditions, collecting over 375,000 responses. Through a series of psychometric analyses, including reliability, reverse-item consistency, factor analysis, convergent validity, and measurement invariance, the paper argues that LLM questionnaire responses are dominated by acquiescence rather than stable personality traits. The authors conclude that persona prompting enables LLMs to role-play personalities but does not produce reliable personality simulations suitable for social science applications

Summary Of Strengths:
The paper studies an increasingly important question as LLM-based agents become more common in behavioral simulation and social science research.

The empirical evaluation is comprehensive, covering multiple psychometric instruments, 20 modern LLMs, numerous persona settings, and a large response dataset.

The paper is generally well organized, with extensive analyses spanning reliability, factor structure, convergent validity, and measurement invariance.

The work provides a useful empirical characterization of how current LLMs respond to personality questionnaires and highlights limitations practitioners should consider when using such evaluations.

Summary Of Weaknesses:
Limited conceptual novelty. The paper’s central finding, namely that LLMs tend to agree with questionnaire items regardless of item direction and therefore fail to exhibit coherent personality measurements, is closely related to the already well-established phenomena of LLM sycophancy, acquiescence, and prompt sensitivity. While the paper provides a thorough psychometric analysis, the primary conclusion feels largely confirmatory rather than introducing a fundamentally new insight.

The main conclusion is largely expected. The conclusion that LLM agents are unreliable for personality simulation follows naturally from previously observed inconsistencies and alignment-induced behaviors in instruction-tuned models. Although the experiments are extensive, I found that they primarily validate an intuition already present in the literature rather than substantially advancing our understanding of LLM behavior.

Limited practical implications. The paper convincingly demonstrates limitations of questionnaire-based personality assessment, but it provides relatively little guidance on what practitioners should use instead. It would strengthen the paper to discuss alternative evaluation methodologies or more reliable approaches for constructing personality-conditioned LLM agents.

Some visualizations could be improved. Figure 3 is difficult to interpret due to the dense heatmap, limited color contrast, and crowded presentation. A simpler visualization or clearer annotations would make the main findings substantially easier to understand.

Comments Suggestions And Typos:
The paper is carefully executed and presents a thorough empirical evaluation of personality questionnaires for LLMs. However, I was not convinced that the central contribution is sufficiently novel for a flagship ACL venue, as the main findings largely reinforce previously known characteristics of instruction-tuned LLMs.

Suggestions:

Better clarify how the contribution differs from prior work on LLM sycophancy, acquiescence bias, and prompt sensitivity.

Expand the discussion of practical alternatives to questionnaire-based personality evaluation.

Improve Figure 3 and other dense visualizations for readability.

More explicitly discuss what new scientific insight is obtained beyond confirming previously observed LLM behaviors through a psychometric lens.

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 2.5 = Borderline Findings
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 3 = Potentially useful: Someone might find the new datasets useful for their work.
Software: 3 = Potentially useful: Someone might find the new software useful for their work.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review
