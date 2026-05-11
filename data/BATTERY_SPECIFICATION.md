# 心理测量量表说明书

本文件详细记录当前实验中使用的 4 套标准化心理测量量表，共 221 道题目。
所有量表均为已在人格心理学领域正式发表、经过信效度验证的公开工具。

---

## 一、量表总览

| 量表 | 题数 | 作答方式 | 维度数 | 反向题数 | 来源 |
|------|------|----------|--------|----------|------|
| IPIP-NEO-120 | 120 | 5 点 Likert | 5 大维度 × 6 个侧面 | 41 | Johnson (2014)，公共领域 |
| SD3 (Short Dark Triad) | 27 | 5 点 Likert | 3 | 5 | Jones & Paulhus (2014) |
| ZKPQ-50-CC | 50 | True / False | 5 | 12 | Aluja et al. (2006) |
| EPQR-A | 24 | Yes / No | 4 | 5 | Francis, Brown & Philipchalk (1992) |
| **合计** | **221** | 3 种格式 | **17** | **63** | — |

### 作答格式说明

- **5 点 Likert（IPIP-NEO-120、SD3）**：被试从 1（很不准确 / 非常不同意）到 5（非常准确 / 非常同意）选择一个数字
- **True / False（ZKPQ-50-CC）**：二选一，True = 1 分，False = 0 分
- **Yes / No（EPQR-A）**：二选一，Yes = 1 分，No = 0 分

### 反向计分规则

- **Likert 题**：keyed = "−" 的题目，得分 = 6 − 原始回答（即 5↔1, 4↔2, 3 不变）
- **True/False 题**：keyed = "−" 的题目，得分 = 1 − 原始回答（即 True→0, False→1）
- **Yes/No 题**：keyed = "−" 的题目，得分 = 1 − 原始回答（即 Yes→0, No→1）
- keyed = "+" 的题目，得分 = 原始回答，不做转换

---

## 二、IPIP-NEO-120（国际人格项目池 — 大五 120 题）

### 基本信息

- **来源**：Johnson, J. A. (2014). Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory. *Journal of Research in Personality*, 51, 78-89.
- **获取渠道**：https://ipip.ori.org/newNEOKey.htm （公共领域，无需授权）
- **题目形式**：每题一句话陈述（如 "Worry about things"），被试评价该陈述描述自己的准确程度
- **计分标签**：1 = Very Inaccurate, 2 = Moderately Inaccurate, 3 = Neither, 4 = Moderately Accurate, 5 = Very Accurate
- **维度分数**：每个维度 = 该维度下所有题目（经反向计分后）的平均值

### 结构：5 大维度 × 6 个侧面 × 每个侧面 4 题 = 120 题

120 道题按照 N1→E1→O1→A1→C1→N2→E2→O2→A2→C2→…→N6→E6→O6→A6→C6 的顺序循环排列，共 4 个循环（每个循环 30 题）。每个侧面有 4 道题，分布在不同循环中。

#### Neuroticism（神经质）— 24 题，5 道反向

| 侧面 | 题数 | 反向题数 | 反向题 |
|------|------|----------|--------|
| N1 Anxiety（焦虑） | 4 | 0 | — |
| N2 Anger（愤怒） | 4 | 1 | ipip_096: "Am not easily annoyed" |
| N3 Depression（抑郁） | 4 | 1 | ipip_071: "Feel comfortable with myself" |
| N4 Self-Consciousness（自我意识） | 4 | 0 | — |
| N5 Immoderation（纵欲） | 4 | 3 | ipip_051: "Rarely overindulge", ipip_081: "Am able to control my cravings", ipip_111: "Easily resist temptations" |
| N6 Vulnerability（脆弱） | 4 | 0 | — |

#### Extraversion（外向性）— 24 题，3 道反向

| 侧面 | 题数 | 反向题数 | 反向题 |
|------|------|----------|--------|
| E1 Friendliness（友善） | 4 | 0 | — |
| E2 Gregariousness（合群） | 4 | 1 | ipip_097: "Avoid crowds" |
| E3 Assertiveness（果断） | 4 | 1 | ipip_072: "Wait for others to lead the way" |
| E4 Activity Level（活动水平） | 4 | 1 | ipip_107: "Prefer to be alone" |
| E5 Excitement Seeking（刺激寻求） | 4 | 0 | — |
| E6 Cheerfulness（愉悦） | 4 | 0 | — |

#### Openness（开放性）— 24 题，9 道反向

| 侧面 | 题数 | 反向题数 | 反向题 |
|------|------|----------|--------|
| O1 Imagination（想象力） | 4 | 0 | — |
| O2 Artistic Interests（艺术兴趣） | 4 | 1 | ipip_098: "Do not like art" |
| O3 Emotionality（情感体验） | 4 | 1 | ipip_103: "Seldom get emotional" |
| O4 Adventurousness（冒险精神） | 4 | 3 | ipip_048: "Prefer to stick with things that I know", ipip_078: "Dislike changes", ipip_108: "Dislike new foods" |
| O5 Intellect（智识） | 4 | 2 | ipip_053: "Avoid philosophical discussions", ipip_083: "Am not interested in abstract ideas" |
| O6 Liberalism（自由主义） | 4 | 2 | ipip_088: "Believe that we should be tough on crime", ipip_118: "Tend to vote for conservative political candidates" |

#### Agreeableness（宜人性）— 24 题，17 道反向（反向题最多的维度）

| 侧面 | 题数 | 反向题数 | 反向题 |
|------|------|----------|--------|
| A1 Trust（信任） | 4 | 1 | ipip_094: "Distrust people" |
| A2 Morality（道德感） | 4 | 4 | ipip_009: "Use others for my own ends", ipip_039: "Cheat to get ahead", ipip_069: "Obstruct others' plans", ipip_099: "Take advantage of others" |
| A3 Altruism（利他） | 4 | 2 | ipip_074: "Am not interested in other people's problems", ipip_104: "Am not really interested in others" |
| A4 Cooperation（合作） | 4 | 4 | ipip_019: "Love a good fight", ipip_049: "Yell at people", ipip_079: "Insult people", ipip_109: "Hold a grudge" |
| A5 Modesty（谦逊） | 4 | 4 | ipip_024: "Believe that I am better than others", ipip_054: "Have a high opinion of myself", ipip_084: "Boast about my virtues", ipip_114: "Think highly of myself" |
| A6 Sympathy（同情心） | 4 | 2 | ipip_089: "Am indifferent to the feelings of others", ipip_119: "Am hard to understand" |

#### Conscientiousness（尽责性）— 24 题，7 道反向

| 侧面 | 题数 | 反向题数 | 反向题 |
|------|------|----------|--------|
| C1 Self-Efficacy（自我效能） | 4 | 0 | — |
| C2 Orderliness（秩序感） | 4 | 2 | ipip_040: "Often forget to put things back in their proper place", ipip_070: "Leave a mess in my room" |
| C3 Dutifulness（尽职） | 4 | 1 | ipip_075: "Break rules" |
| C4 Achievement-Striving（成就追求） | 4 | 1 | ipip_110: "Do just enough work to get by" |
| C5 Self-Discipline（自律） | 4 | 0 | — |
| C6 Cautiousness（谨慎） | 4 | 3 | ipip_030: "Rush into things", ipip_060: "Make rash decisions", ipip_090: "Jump into things without thinking" |

---

## 三、SD3（Short Dark Triad — 黑暗三角简版）

### 基本信息

- **来源**：Jones, D. N., & Paulhus, D. L. (2014). Introducing the Short Dark Triad (SD3): A brief measure of dark personality traits. *Assessment*, 21(1), 28-41. DOI: 10.1177/1073191113514105
- **获取渠道**：https://www2.psych.ubc.ca/~dpaulhus/research/DARK_TRAITS/MEASURES/SD3.1.1.doc （作者公开发布）
- **题目形式**：陈述句，被试表达同意程度
- **计分标签**：1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree
- **维度分数**：每个维度 9 道题反向计分后的平均值

### 结构：3 个维度 × 每维度 9 题 = 27 题，共 5 道反向题

#### Machiavellianism（马基雅维利主义）— 9 题，0 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | sd3_001 | It's not wise to tell your secrets. | + |
| 2 | sd3_002 | I like to use clever manipulation to get my way. | + |
| 3 | sd3_003 | Whatever it takes, you must get the important people on your side. | + |
| 4 | sd3_004 | Avoid direct conflict with others because they may be useful in the future. | + |
| 5 | sd3_005 | It's wise to keep track of information that you can use against people later. | + |
| 6 | sd3_006 | You should wait for the right time to get back at people. | + |
| 7 | sd3_007 | There are things you should hide from other people because they don't need to know. | + |
| 8 | sd3_008 | Make sure your plans benefit you, not others. | + |
| 9 | sd3_009 | Most people can be manipulated. | + |

#### Narcissism（自恋）— 9 题，3 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | sd3_010 | People see me as a natural leader. | + |
| 2 | sd3_011 | **I hate being the center of attention.** | **−** |
| 3 | sd3_012 | Many group activities tend to be dull without me. | + |
| 4 | sd3_013 | I know that I am special because everyone keeps telling me so. | + |
| 5 | sd3_014 | I like to get acquainted with important people. | + |
| 6 | sd3_015 | **I feel embarrassed if someone compliments me.** | **−** |
| 7 | sd3_016 | I have been compared to famous people. | + |
| 8 | sd3_017 | **I am an average person.** | **−** |
| 9 | sd3_018 | I insist on getting the respect I deserve. | + |

#### Psychopathy（精神病态）— 9 题，2 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | sd3_019 | I like to get revenge on authorities. | + |
| 2 | sd3_020 | **I avoid dangerous situations.** | **−** |
| 3 | sd3_021 | Payback needs to be quick and nasty. | + |
| 4 | sd3_022 | People often say I'm out of control. | + |
| 5 | sd3_023 | It's true that I can be mean to others. | + |
| 6 | sd3_024 | People who mess with me always regret it. | + |
| 7 | sd3_025 | **I have never gotten into trouble with the law.** | **−** |
| 8 | sd3_026 | I enjoy having sex with people I hardly know. | + |
| 9 | sd3_027 | I'll say anything to get what I want. | + |

---

## 四、ZKPQ-50-CC（Zuckerman-Kuhlman 人格问卷跨文化简版）

### 基本信息

- **来源**：Aluja, A., Rossier, J., Garcia, L. F., Angleitner, A., Kuhlman, M., & Zuckerman, M. (2006). A cross-cultural shortened form of the ZKPQ (ZKPQ-50-CC) adapted to English, French, German, and Spanish languages. *Personality and Individual Differences*, 41(4), 619-628.
- **获取渠道**：https://www.psytoolkit.org/survey-library/zkpq-50-cc.html （PsyToolkit 学术平台验证版本）
- **题目形式**：陈述句，被试回答 True 或 False
- **计分**：True = 1, False = 0；反向题翻转（True→0, False→1）
- **维度分数**：每个维度 10 道题反向计分后的平均值（范围 0~1）

### 结构：5 个维度 × 每维度 10 题 = 50 题，共 12 道反向题

#### Activity（活跃性）— 10 题，2 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | zkpq_001 | I do not like to waste time just sitting around and relaxing. | + |
| 2 | zkpq_002 | I lead a busier life than most people. | + |
| 3 | zkpq_003 | I like to be doing things all of the time. | + |
| 4 | zkpq_004 | **I can enjoy myself just lying around and not doing anything active.** | **−** |
| 5 | zkpq_005 | **I do not feel the need to be doing things all of the time.** | **−** |
| 6 | zkpq_006 | When on vacation I like to engage in active sports rather than just lie around. | + |
| 7 | zkpq_007 | I like to wear myself out with hard work or exercise. | + |
| 8 | zkpq_008 | I like to be active as soon as I wake up in the morning. | + |
| 9 | zkpq_009 | I like to keep busy all the time. | + |
| 10 | zkpq_010 | When I do things, I do them with lots of energy. | + |

#### Aggression-Hostility（攻击-敌意）— 10 题，3 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | zkpq_011 | When I get mad, I say ugly things. | + |
| 2 | zkpq_012 | It's natural for me to curse when I am mad. | + |
| 3 | zkpq_013 | **I almost never feel like I would like to hit someone.** | **−** |
| 4 | zkpq_014 | **If someone offends me, I just try not to think about it.** | **−** |
| 5 | zkpq_015 | If people annoy me I do not hesitate to tell them so. | + |
| 6 | zkpq_016 | When people disagree with me I cannot help getting into an argument with them. | + |
| 7 | zkpq_017 | I have a very strong temper. | + |
| 8 | zkpq_018 | I can't help being a little rude to people I do not like. | + |
| 9 | zkpq_019 | **I am always patient with others even when they are irritating.** | **−** |
| 10 | zkpq_020 | When people shout at me, I shout back. | + |

#### Impulsive Sensation Seeking（冲动感觉寻求）— 10 题，0 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | zkpq_021 | I often do things on impulse. | + |
| 2 | zkpq_022 | I would like to take off on a trip with no preplanned or definite routes or timetables. | + |
| 3 | zkpq_023 | I enjoy getting into new situations where you can't predict how things will turn out. | + |
| 4 | zkpq_024 | I sometimes like to do things that are a little frightening. | + |
| 5 | zkpq_025 | I'll try anything once. | + |
| 6 | zkpq_026 | I would like the kind of life where one is on the move and travelling a lot. | + |
| 7 | zkpq_027 | I sometimes do "crazy" things just for fun. | + |
| 8 | zkpq_028 | I prefer friends who are excitingly unpredictable. | + |
| 9 | zkpq_029 | I often get so carried away by new and exciting things that I never think of possible complications. | + |
| 10 | zkpq_030 | I like "wild" uninhibited parties. | + |

#### Neuroticism-Anxiety（神经质-焦虑）— 10 题，1 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | zkpq_031 | My body often feels all tightened up for no apparent reason. | + |
| 2 | zkpq_032 | I frequently get emotionally upset. | + |
| 3 | zkpq_033 | I tend to be oversensitive and easily hurt by thoughtless remarks. | + |
| 4 | zkpq_034 | I am easily frightened. | + |
| 5 | zkpq_035 | I sometimes feel panicky. | + |
| 6 | zkpq_036 | I often feel unsure of myself. | + |
| 7 | zkpq_037 | I often worry about things that other people think are unimportant. | + |
| 8 | zkpq_038 | I often feel like crying sometimes without a reason. | + |
| 9 | zkpq_039 | **I don't let a lot of trivial things irritate me.** | **−** |
| 10 | zkpq_040 | I often feel uncomfortable and ill at ease for no real reason. | + |

#### Sociability（社交性）— 10 题，6 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | zkpq_041 | **I do not mind going out alone and usually prefer it to being out in a large group.** | **−** |
| 2 | zkpq_042 | I spend as much time with my friends as I can. | + |
| 3 | zkpq_043 | **I do not need a large number of casual friends.** | **−** |
| 4 | zkpq_044 | **I tend to be uncomfortable at big parties.** | **−** |
| 5 | zkpq_045 | At parties, I enjoy mingling with many people whether I already know them or not. | + |
| 6 | zkpq_046 | **I would not mind being socially isolated for some period of time.** | **−** |
| 7 | zkpq_047 | **Generally, I like to be alone so I can do things without social distractions.** | **−** |
| 8 | zkpq_048 | I am a very sociable person. | + |
| 9 | zkpq_049 | **I usually prefer to do things alone.** | **−** |
| 10 | zkpq_050 | I probably spend more time than I should socializing with friends. | + |

---

## 五、EPQR-A（Eysenck 人格问卷修订简版）

### 基本信息

- **来源**：Francis, L. J., Brown, L. B., & Philipchalk, R. (1992). The development of an abbreviated form of the Revised Eysenck Personality Questionnaire (EPQR-A): Its use among students in England, Canada, the U.S.A. and Australia. *Personality and Individual Differences*, 13(4), 443-449. DOI: 10.1016/0191-8869(92)90073-X
- **母量表**：Eysenck, Eysenck & Barrett (1985) 的 EPQ-R 100 题版本。Francis 等人从中选取 24 道题（每个维度 6 道）形成简版
- **题目形式**：问句（如 "Are you a talkative person?"），被试回答 Yes 或 No
- **计分**：Yes = 1, No = 0；反向题翻转（Yes→0, No→1）
- **维度分数**：每个维度 6 道题反向计分后的平均值（范围 0~1）

### 结构：4 个维度 × 每维度 6 题 = 24 题，共 5 道反向题

#### Psychoticism（精神质）— 6 题，0 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | epqr_001 | Would you take drugs which may have strange or dangerous effects? | + |
| 2 | epqr_002 | Do you prefer to go your own way rather than act by the rules? | + |
| 3 | epqr_003 | Do you enjoy hurting people you love? | + |
| 4 | epqr_004 | Do you enjoy practical jokes that can sometimes really hurt people? | + |
| 5 | epqr_005 | Do you sometimes talk about things you know nothing about? | + |
| 6 | epqr_006 | Do you think marriage is old-fashioned and should be done away with? | + |

#### Extraversion（外向性）— 6 题，3 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | epqr_007 | Are you a talkative person? | + |
| 2 | epqr_008 | Are you rather lively? | + |
| 3 | epqr_009 | Can you usually let yourself go and enjoy yourself at a lively party? | + |
| 4 | epqr_010 | **Do you tend to keep in the background on social occasions?** | **−** |
| 5 | epqr_011 | **Do you prefer reading to meeting people?** | **−** |
| 6 | epqr_012 | **Are you mostly quiet when you are with other people?** | **−** |

#### Neuroticism（神经质）— 6 题，0 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | epqr_013 | Does your mood often go up and down? | + |
| 2 | epqr_014 | Do you ever feel 'just miserable' for no reason? | + |
| 3 | epqr_015 | Are your feelings easily hurt? | + |
| 4 | epqr_016 | Are you often troubled about feelings of guilt? | + |
| 5 | epqr_017 | Would you call yourself tense or 'highly-strung'? | + |
| 6 | epqr_018 | Do you worry about awful things that might happen? | + |

#### Lie（说谎量表）— 6 题，2 道反向

| # | ID | 题目 | 方向 |
|---|------|------|------|
| 1 | epqr_019 | If you say you will do something, do you always keep your promise no matter how inconvenient it might be? | + |
| 2 | epqr_020 | **Were you ever greedy by helping yourself to more than your share of anything?** | **−** |
| 3 | epqr_021 | **Have you ever blamed someone for doing something you knew was really your fault?** | **−** |
| 4 | epqr_022 | As a child did you do as you were told immediately and without grumbling? | + |
| 5 | epqr_023 | Do you always practice what you preach? | + |
| 6 | epqr_024 | Are all your habits good and desirable ones? | + |

---

## 六、反向题统计摘要

| 量表 | 维度 | 正向题数 | 反向题数 | 反向题占比 |
|------|------|----------|----------|------------|
| IPIP-NEO-120 | Neuroticism | 19 | 5 | 21% |
| IPIP-NEO-120 | Extraversion | 21 | 3 | 13% |
| IPIP-NEO-120 | Openness | 15 | 9 | 38% |
| IPIP-NEO-120 | Agreeableness | 7 | 17 | 71% |
| IPIP-NEO-120 | Conscientiousness | 17 | 7 | 29% |
| SD3 | Machiavellianism | 9 | 0 | 0% |
| SD3 | Narcissism | 6 | 3 | 33% |
| SD3 | Psychopathy | 7 | 2 | 22% |
| ZKPQ-50-CC | Activity | 8 | 2 | 20% |
| ZKPQ-50-CC | Aggression-Hostility | 7 | 3 | 30% |
| ZKPQ-50-CC | Impulsive Sensation Seeking | 10 | 0 | 0% |
| ZKPQ-50-CC | Neuroticism-Anxiety | 9 | 1 | 10% |
| ZKPQ-50-CC | Sociability | 4 | 6 | 60% |
| EPQR-A | Psychoticism | 6 | 0 | 0% |
| EPQR-A | Extraversion | 3 | 3 | 50% |
| EPQR-A | Neuroticism | 6 | 0 | 0% |
| EPQR-A | Lie | 4 | 2 | 33% |
| **总计** | — | **158** | **63** | **29%** |

> **注意**：IPIP-NEO-120 的 Agreeableness（宜人性）维度有 71% 的反向题（24 题中 17 题反向），这是该量表的设计特点——用大量负面行为描述来间接测量宜人性，使得该维度对 LLM 的"积极回应偏向"特别敏感。

---

## 七、数据文件位置

- **量表题目数据**：`data/items_battery.json`（221 道题的完整元数据）
- **构建脚本**：`data/build_battery.py`（含所有题目原文、计分方向、来源注释）
- **实验执行脚本**：`run_model_experiments.py`（V4.0，加载上述 JSON 文件运行实验）
