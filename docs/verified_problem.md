# 221 题量表来源与正确性验证报告

本文对提交 `b76b197`（*Replace 61 ad-hoc items with 4 validated psychometric scales (221 items)*）引入的
4 套心理测量量表进行交叉核查，判断其是否真实存在、题目与反向计分方向是否与
`data/BATTERY_SPECIFICATION.md` 中标注的原始文献相符。

> 覆盖文件：
> - `data/build_battery.py`（题库构建脚本）
> - `data/items_battery.json`（统一题目数据，2076 行）
> - `data/BATTERY_SPECIFICATION.md`（量表说明书）
> - `run_model_experiments.py`（V4.0，读取题库并执行反向计分的实验脚本）

---

## 一、核查方法

1. **机器层面核查**：直接 `import` `build_battery.py` 中的 4 个 Python 列表，
   统计总题数、每量表题数、每维度题数和反向题数，与 `BATTERY_SPECIFICATION.md` 中的表格
   以及提交信息声称的「221 题 / 63 道反向题」相对照。
2. **源头层面核查**：按说明书给出的出处逐条拉取原始文献 / 官方量表 key，与 `items_battery.json`
   逐题比对题干文本和反向方向：
   - IPIP-NEO-120：ipip.ori.org 上的 Johnson (2014) 官方记分键
     (`30FacetNEO-PI-RItems.htm`) 与 Maples et al. (2014) 记分键
     (`30FacetNEO-PI-RItems_Maples_etal.htm`)。
   - SD3：Jones & Paulhus (2014) 的 SD3.1.1 官方量表（UBC Paulhus 实验室）。
   - ZKPQ-50-CC：PsyToolkit 学术平台公开版本与 Aluja et al. (2006) 原文附表。
   - EPQR-A：Francis, Brown & Philipchalk (1992) 原文的 24 题附表以及
     Lewis, Francis & Ziebertz (2002) 等多项跨文化验证研究的记分指引。
3. **脚本一致性核查**：读 `run_model_experiments.py` 的 `apply_reverse_scoring`
   函数，确认反向计分算法与 Likert-5 / 二元计分的标准做法一致。

---

## 二、量表是否真实存在？——文献与公开来源核查

| 量表 | 宣称出处 | 是否真实存在 | 验证来源 |
|------|----------|--------------|----------|
| IPIP-NEO-120 | Johnson (2014), *JRP* 51: 78-89 | ✅ 真实，PDF/DOI 可查；ipip.ori.org 公开发布官方记分键 | https://ipip.ori.org/30FacetNEO-PI-RItems.htm |
| SD3 | Jones & Paulhus (2014), *Assessment* 21(1): 28-41, DOI 10.1177/1073191113514105 | ✅ 真实，SAGE 已发表；作者 Paulhus 实验室长期公开 SD3.1.1 | https://www2.psych.ubc.ca/~dpaulhus/research/DARK_TRAITS/MEASURES/ |
| ZKPQ-50-CC | Aluja et al. (2006), *PAID* 41(4): 619-628 | ✅ 真实，Elsevier 已发表；PsyToolkit 亦收录 | https://www.psytoolkit.org/survey-library/zkpq-50-cc.html |
| EPQR-A | Francis, Brown & Philipchalk (1992), *PAID* 13(4): 443-449, DOI 10.1016/0191-8869(92)90073-X | ✅ 真实，Elsevier 已发表，是 EPQ-R 100 题的缩短版 | DOI 解析可查，并被大量后续研究引用 |

**结论：四套量表本身都是心理计量学界广泛使用、经同行评审发表的正式工具，来源真实可查。**

---

## 三、题目总数、维度与反向题数的机内一致性

直接运行 `build_battery.py` 中的 4 个列表统计，得到：

| 量表 | 题数 | 反向题数 | 说明书声称 | 是否一致 |
|------|------|----------|------------|----------|
| IPIP-NEO-120 | 120 | 41 | 120 / 41 | ✅ |
| SD3 | 27 | 5 | 27 / 5 | ✅ |
| ZKPQ-50-CC | 50 | 12 | 50 / 12 | ✅ |
| EPQR-A | 24 | 5 | 24 / 5 | ✅ |
| **合计** | **221** | **63** | **221 / 63** | ✅ |

维度级核验（例：IPIP Agreeableness = 24 题 / 17 反向；ZKPQ Sociability = 10 题 / 6 反向；
EPQR Extraversion = 6 题 / 3 反向）均与 `BATTERY_SPECIFICATION.md` 第 307-328 行的汇总表一一对应。
反向题逐条列举的条目（说明书 49-101 行）也和 Python 列表中的 `keyed="-"` 条目完全吻合。

### 3.1 发现的机内不一致（BUG 级别，较轻）

`build_battery.py` 在构建 `battery["scales"]["<name>"]["n_reverse"]` 这个元数据字段时，
若干数值与真实数据不符：

| 量表 | 脚本写入的 `n_reverse` | 实际计数 | 说明书表格 | 结论 |
|------|-----------------------|----------|------------|------|
| IPIP-NEO-120 | **55** | 41 | 41 | ❌ 元数据字段错 |
| SD3 | 5 | 5 | 5 | ✅ |
| ZKPQ-50-CC | 12 | 12 | 12 | ✅ |
| EPQR-A | **9** | 5 | 5 | ❌ 元数据字段错 |

这两个字段只是 `items_battery.json` 中的一个注释型字段，**并不影响反向计分的执行**
（执行时读取的是每个 item 自己的 `keyed` 字段），但会让任何直接读取 `scales.*.n_reverse`
的下游脚本或论文报表出错。建议修复为 41 和 5。

### 3.2 反向计分算法核查

`run_model_experiments.py:393-398`：

```python
def apply_reverse_scoring(raw: int, keyed: str, response_format: str) -> int:
    if keyed != "-":
        return raw
    if response_format == "likert_5":
        return 6 - raw
    return 1 - raw  # binary: 0↔1
```

- Likert-5：`6 − raw` → 5↔1, 4↔2, 3 不变。✅ 与心理测量学标准做法一致。
- 二元（True/False、Yes/No）：`1 − raw` → 1↔0。✅ 标准做法。
- 条件分支仅对 `keyed == "-"` 生效，不会错误翻转正向题。✅

---

## 四、逐条比对 —— 每套量表与原始文献的文本与记分方向

### 4.1 SD3（Short Dark Triad） ✅ 完全对得上

将 `SD3_ITEMS` 与 Jones & Paulhus (2014) 的 SD3.1.1 官方版本逐条比对：

- 题数：27 题 = 马基雅维利 9 + 自恋 9 + 精神病态 9 ✅
- 反向题：正好 5 道，且**位置完全正确**：
  - Narcissism #2 "I hate being the center of attention" (`sd3_011`)
  - Narcissism #6 "I feel embarrassed if someone compliments me" (`sd3_015`)
  - Narcissism #8 "I am an average person" (`sd3_017`)
  - Psychopathy #2 "I avoid dangerous situations" (`sd3_020`)
  - Psychopathy #7 "I have never gotten into trouble with the law" (`sd3_025`)
- 题干文本：全部 27 道题与 SD3.1.1 word 文档逐字一致（连标点、"mess with me" 等措辞都相同）。

**评价：SD3 部分正确、真实、来源与记分方向均无误。**

### 4.2 ZKPQ-50-CC ✅ 基本对得上（仅少量非实质性措辞差异）

将 `ZKPQ_ITEMS` 与 PsyToolkit 公布的 Aluja et al. (2006) 版本比对：

- 题数：50 = 每维度 10 × 5 ✅
- 反向题：共 12 道，位置为 4, 5, 13, 14, 19, 39, 41, 43, 44, 46, 47, 49，和 PsyToolkit 版本完全一致 ✅
- 题干：绝大多数逐字一致。仅发现以下轻微措辞差异（不影响题目含义或记分）：
  - `zkpq_026`：repo = "I would like the kind of life where one is on the move and travelling a lot"；
    PsyToolkit/原文末尾多出 "with lots of change and excitement" 的定语从句。
  - `zkpq_029`：repo 的 "new and exciting things" 原文为 "new and exciting things and ideas"。
  - `zkpq_033`、`zkpq_046`、`zkpq_047`：句尾分别省略了 "and actions of others"、"in some place"、
    "I want to do"。

**评价：ZKPQ-50-CC 部分正确、真实、记分方向无误。措辞上的轻微截断不影响量表心理计量属性，但严格来说
 应保留原文完整句子以便与文献可比。**

### 4.3 EPQR-A ✅ 完全对得上

将 `EPQR_A_ITEMS` 与 Francis et al. (1992) 附录 A 比对：

- 题数：24 = P/E/N/L 每维度 6 道 ✅
- 反向题：正好 5 道：
  - Extraversion：items 10, 11, 12（background / prefer reading / mostly quiet）✅
  - Lie：items 20, 21（were you ever greedy / have you ever blamed）✅
- 题干：24 条全部逐字一致（包括疑问句格式 "Are you...?"、"Do you...?"、Lie 分量表特有的
  "As a child did you do as you were told immediately..." 等）。

**评价：EPQR-A 部分正确、真实、记分方向无误。**

### 4.4 IPIP-NEO-120 ⚠️ **核心问题：题目池与 Johnson (2014) 官方记分键有系统性偏差**

这是本次交叉验证中**最值得关注的问题**。说明书将出处标注为 "Johnson (2014)"，并附
<https://ipip.ori.org/newNEOKey.htm>。但该链接指向的是 **300 题** 原始 IPIP-NEO 词库
（Goldberg 的 1999 版本），**不是** Johnson 2014 的 120 题选集。Johnson 2014 真正的记分键
（每 facet 只选 4 道题）应在：

- <https://ipip.ori.org/30FacetNEO-PI-RItems.htm>（Johnson 版本）
- <https://ipip.ori.org/30FacetNEO-PI-RItems_Maples_etal.htm>（Maples et al., 2014 的另一版 120 题）

将 repo 的 120 道题逐 facet 与上述两份 **官方** 记分键比对，发现：

| Facet | Johnson (2014) 官方 4 题 | repo 的 4 题（异常用粗体） | 与 Johnson 对应数 |
|-------|--------------------------|----------------------------|--------------------|
| N1 Anxiety | Worry about things / Fear for the worst / **Am afraid of many things** / Get stressed out easily | Worry about things / Fear for the worst / **Am easily disturbed** / Am stressed out | 2/4（两道异文或错位） |
| N2 Anger | Get angry easily / Get irritated easily / Lose my temper / Am not easily annoyed (−) | Get angry easily / Lose my temper / **Lose my composure** / Am not easily annoyed (−) | 3/4 |
| N3 Depression | Often feel blue / Dislike myself / Am often down in the dumps / Feel comfortable with myself (−) | Often feel blue / Dislike myself / **Feel desperate** / Feel comfortable with myself (−) | 3/4 |
| N4 Self-Consciousness | Find it difficult to approach others / Am afraid to draw attention to myself / Only feel comfortable with friends / Am not bothered by difficult social situations (−) | Find it difficult to approach others / **Am afraid that I will do the wrong thing** / Only feel comfortable with friends / **Am afraid of many things** | 2/4（其中「Am afraid of many things」本应在 N1） |
| N5 Immoderation | Go on binges / Rarely overindulge (−) / Easily resist temptations (−) / Am able to control my cravings (−) | 同左 | **4/4 ✅** |
| N6 Vulnerability | Panic easily / Become overwhelmed by events / Feel that I'm unable to deal with things / Remain calm under pressure (−) | Panic easily / Become overwhelmed by events / **Can't make up my mind** / **Feel that my life lacks direction** | 2/4 |
| E1 Friendliness | Make friends easily / Feel comfortable around people / Avoid contacts with others (−) / Keep others at a distance (−) | Make friends easily / Feel comfortable around people / **Act comfortable with others** / **Am skilled in handling social situations** | 2/4（且 keyed 方向偏差：应 2+2，repo 成 4+0） |
| E2 Gregariousness | Love large parties / Talk to a lot of different people at parties / Prefer to be alone (−) / Avoid crowds (−) | Love large parties / Talk to a lot of different people at parties / **Enjoy parties** / Avoid crowds (−) | 3/4 |
| E3 Assertiveness | Take charge / Try to lead others / Take control of things / Wait for others to lead the way (−) | Take charge / Try to lead others / Wait for others to lead the way (−) / **Am the life of the party** | 3/4 |
| E4 Activity Level | Am always busy / Am always on the go / Do a lot in my spare time / Like to take it easy (−) | Am always busy / Am always on the go / Do a lot in my spare time / **Prefer to be alone** (−) | 3/4（「Prefer to be alone」本是 E2 反向题） |
| E5 Excitement-Seeking | Love excitement / Seek adventure / Enjoy being reckless / Act wild and crazy | Love excitement / Seek adventure / Act wild and crazy / **Seek danger** | 3/4 |
| E6 Cheerfulness | Radiate joy / Have a lot of fun / Love life / Look at the bright side of life | Radiate joy / Have a lot of fun / Look at the bright side of life / **Am always joking around** | 3/4 |
| O1 Imagination | Have a vivid imagination / Enjoy wild flights of fantasy / Love to daydream / Like to get lost in thought | Have a vivid imagination / Enjoy wild flights of fantasy / **Enjoy daydreaming** / Like to get lost in thought | 3/4（"Enjoy daydreaming" 与 "Love to daydream" 近义但非原文） |
| O2 Artistic Interests | Believe in the importance of art / See beauty in things... / Do not like poetry (−) / Do not enjoy going to art museums (−) | Believe in the importance of art / See beauty in things... / **Like music** / Do not like art (−) | 2/4（应 2+2，repo 成 3+1） |
| O3 Emotionality | Experience my emotions intensely / Feel others' emotions / Rarely notice my emotional reactions (−) / Don't understand people who get emotional (−) | Experience my emotions intensely / Feel others' emotions / **Enjoy thinking about things** / Seldom get emotional (−) | 2/4（"Enjoy thinking about things" 属于 O5 词库） |
| O4 Adventurousness | Prefer variety to routine / Prefer to stick with things that I know (−) / Dislike changes (−) / Am attached to conventional ways (−) | Prefer variety to routine / Prefer to stick with things that I know (−) / Dislike changes (−) / **Dislike new foods** (−) | 3/4 |
| O5 Intellect | Love to read challenging material / Avoid philosophical discussions (−) / Have difficulty understanding abstract ideas (−) / Am not interested in theoretical discussions (−) | Love to read challenging material / Avoid philosophical discussions (−) / **Am not interested in abstract ideas** (−) / **Try to understand myself** (+) | 2/4（应 1+3，repo 成 2+2；"Try to understand myself" 属于 O3 词库） |
| O6 Liberalism | Tend to vote for liberal political candidates / Believe that there is no absolute right and wrong / Tend to vote for conservative political candidates (−) / Believe that we should be tough on crime (−) | 同左，仅"right or wrong"与"right and wrong"一字之差 | **4/4 ✅（近乎完全匹配）** |
| A1 Trust | Trust others / Believe that others have good intentions / Trust what people say / Distrust people (−) | 同左 | **4/4 ✅** |
| A2 Morality | Use others for my own ends (−) / Cheat to get ahead (−) / Take advantage of others (−) / Obstruct others' plans (−) | 同左 | **4/4 ✅** |
| A3 Altruism | Am concerned about others / Love to help others / Am indifferent to the feelings of others (−) / Take no time for others (−) | Am concerned about others / Love to help others / **Am not interested in other people's problems** (−) / **Am not really interested in others** (−) | 2/4（两道反向题互换到了 A6；"Am not really interested in others" 属于 E1 词库） |
| A4 Cooperation | Love a good fight (−) / Yell at people (−) / Insult people (−) / Get back at others (−) | Love a good fight (−) / Yell at people (−) / Insult people (−) / **Hold a grudge** (−) | 3/4 |
| A5 Modesty | Believe that I am better than others (−) / Think highly of myself (−) / Have a high opinion of myself (−) / Boast about my virtues (−) | 同左 | **4/4 ✅** |
| A6 Sympathy | Sympathize with the homeless / Feel sympathy for those who are worse off... / Am not interested in other people's problems (−) / Try not to think about the needy (−) | Sympathize with the homeless / Feel sympathy... / **Am indifferent to the feelings of others** (−) / **Am hard to understand** (−) | 2/4（反向题错位到 A3；"Am hard to understand" 不在任何已知版本的 IPIP-NEO-120 中） |
| C1 Self-Efficacy | Complete tasks successfully / Excel in what I do / Handle tasks smoothly / Know how to get things done | 同左 | **4/4 ✅** |
| C2 Orderliness | Like to tidy up / Often forget to put things back... (−) / Leave a mess in my room (−) / Leave my belongings around (−) | Like to tidy up / Often forget to put things back... (−) / Leave a mess in my room (−) / **Want everything to be 'just right'** (+) | 3/4（应 1+3，repo 成 2+2） |
| C3 Dutifulness | Keep my promises / Tell the truth / Break rules (−) / Break my promises (−) | Keep my promises / Tell the truth / Break rules (−) / **Behave properly** (+) | 3/4（应 2+2，repo 成 3+1） |
| C4 Achievement-Striving | Do more than what's expected of me / Work hard / Put little time and effort... (−) / Do just enough work to get by (−) | Work hard / Do more than what's expected of me / **Set high standards for myself** / Do just enough work to get by (−) | 3/4（应 2+2，repo 成 3+1） |
| C5 Self-Discipline | Am always prepared / Carry out my plans / Waste my time (−) / Have difficulty starting tasks (−) | Carry out my plans / **Have excellent ideas** (+) / **Get to work at once** / **Get chores done right away** | 1/4（"Have excellent ideas" 明显是 Intellect/Ideas 词库；应 2+2，repo 成 4+0） |
| C6 Cautiousness | Jump into things without thinking (−) / Make rash decisions (−) / Rush into things (−) / Act without thinking (−) | Jump into things without thinking (−) / Make rash decisions (−) / Rush into things (−) / **Stick to my chosen path** (+) | 3/4（应 0+4，repo 成 1+3） |

**统计结果：**

- 30 个 facet 中，**只有 7 个**（N5, O6, A1, A2, A5, C1 以及边际的 O6）与 Johnson (2014) 官方 120 题完全匹配。
- 其余 23 个 facet 出现不同程度的题目替换；约 30 条题目要么是 Johnson 记分键中没有的（可能来自 300 题词库
  或别的 facet），要么存在反向方向设错，要么数量比例与官方 "a+b" 分布不符。
- 同时比对 Maples et al. (2014) 的另一版 120 题记分键，吻合程度也更低（例如 Maples 的 A1
  含 "Trust what people say"、A5 含 "Make myself the center of attention"，与 repo 的题目均不一致）。
- 个别题项如 `ipip_119` "Am hard to understand" 在 IPIP 官方 3320 题词库中可检索到，但**被放到
  Agreeableness / Sympathy 维度下作为反向题**，其心理学含义非常可疑——这是一句"内省式"的描述，
  和 Sympathy facet 的核心内容（对他人困境的关心）缺乏理论对应。类似地，`ipip_025` 把
  "Have excellent ideas" 放到 C5 Self-Discipline 下也明显是 facet 错配。

**结论：**

- **量表本身真实**（Johnson 2014 及 IPIP-NEO-120 都是公开发表的工具）。
- **repo 的 120 题并非 Johnson (2014) 官方 4 题子集**；更像是从 IPIP 3320 题词库中按 facet 名称
  重新挑选了 4 道题，过程中出现若干错配（把某 facet 的题放到了另一个 facet，或反向方向搞错了极性）。
- 对 **整体大五维度分** 而言，由于每个 facet 仍是 4 道意思相关的题，影响较小（IPIP 大五总分的
  α 往往在 .85+，稳健性很好）；但对 **facet-level** 的结果（论文第 4 节若用到），这些错配会：
  1. 降低某些 facet 的 α（尤其是 C5 Self-Discipline 被搀入一道 Intellect 题后，内部一致性会塌陷）；
  2. 改变"自恋式回应偏向"对 Agreeableness 的作用方向（因为 A3/A6 的反向题互换）。
- 脚本运行本身不会报错（所有题都会被正常读取和反向记分），但**与文献的可比性不如 SD3/ZKPQ/EPQR-A**。

---

## 五、其他发现

### 5.1 说明书中的一个 URL 误导

`BATTERY_SPECIFICATION.md` 第 38 行把 IPIP-NEO-120 的 "获取渠道" 写成
<https://ipip.ori.org/newNEOKey.htm>。这个链接指向的是 1999 年 Goldberg 的 IPIP 300 题，
**不是** Johnson (2014) 的 120 题记分键。建议改为 <https://ipip.ori.org/30FacetNEO-PI-RItems.htm>。
如果作者真正想用 Johnson (2014) 的 120 题子集，需要把题库替换为上述页面列出的 120 条；
如果只是想借用 IPIP 词库自行挑题，则不应把出处写成 "Johnson (2014)" —— 这样会让读者
以为实验用的是 JRP 上那个经 619,150 人样本验证过的 α 系数对应的具体题集。

### 5.2 API key 明文落库（与本题无关但值得一提）

`run_model_experiments.py:35, 38` 将两套 API key（SiliconFlow 与 YiHe）以明文写入源代码并已经
推入 Git 历史。即使密钥已撤销，也应以 `os.environ` / `.env` 方式加载，以免作者未来再次不慎提交。
本条与 221 题真实性无直接关系，附带提醒。

### 5.3 `scales.*.n_reverse` 元数据错值

见 §3.1，IPIP 写成 55、EPQR 写成 9，与真实反向题数 41 和 5 不符。虽然计分时不会读这两个字段，
但论文或后续脚本若引用这个元数据汇总表就会错。建议修正。

---

## 六、总体判定

| 维度 | 判定 |
|------|------|
| 四套量表本身是否真实存在？ | ✅ 全部真实，均为公开发表且被广泛引用的心理计量工具 |
| 题数 221 / 反向题 63 是否与说明书自洽？ | ✅ 机内完全一致（除 JSON 元数据的两个小 bug） |
| SD3 是否忠实于 Jones & Paulhus (2014)？ | ✅ 27 题逐字一致，5 道反向题位置完全正确 |
| ZKPQ-50-CC 是否忠实于 Aluja et al. (2006)？ | ✅ 50 题结构与 12 道反向题位置完全一致；3–4 条题目末尾被轻度截断（非实质性） |
| EPQR-A 是否忠实于 Francis et al. (1992)？ | ✅ 24 题逐字一致，5 道反向题位置完全正确 |
| IPIP-NEO-120 是否忠实于 Johnson (2014)？ | ⚠️ **部分不一致**：30 个 facet 中只有 7 个与官方 4 题键完全一致；其余 facet 存在题目替换、facet 错位或反向方向不符；也不能对应 Maples et al. (2014) 的另一版 120 题 |
| 反向计分算法是否正确？ | ✅ Likert-5 用 `6−raw`，二元用 `1−raw`，均为标准做法 |

**一句话结论**：**4 套量表都是真实存在的已发表工具；SD3、ZKPQ-50-CC、EPQR-A 的 101 道题
与原文献完全对得上，使用上没有问题。IPIP-NEO-120 的 120 道题则并非 Johnson (2014)
官方 4-题/facet 子集，而是一套从 IPIP 词库中自行挑选的近似版本，其中约 1/4 的题目存在
facet 错位或反向方向不符，影响 facet 层面的可比性与内部一致性；建议在论文披露时要么
换成官方记分键上的 Johnson 120 题，要么把来源改标为 "IPIP 词库（Goldberg, 1999）自选
4 题/facet"。**

---

## 七、修复建议（按优先级）

1. **（高）将 IPIP-NEO-120 120 道题替换成
   <https://ipip.ori.org/30FacetNEO-PI-RItems.htm> 列出的官方 Johnson (2014) 题目**，
   或在说明书中明确说明"自选 4 题/facet，不是 Johnson (2014) 子集"。
2. **（中）修正 `build_battery.py` 第 365、395 行的 `n_reverse` 元数据** 为 41 与 5。
3. **（中）更正 `BATTERY_SPECIFICATION.md` 第 38 行的 URL**：把 `newNEOKey.htm` 改为
   `30FacetNEO-PI-RItems.htm`。
4. **（低）补全 ZKPQ 三条被截断的题干**（`zkpq_026`、`zkpq_029`、`zkpq_033/46/47`）以与原文一致。
5. **（低）把 `run_model_experiments.py` 中的两套 API key 迁移到环境变量**，并从
   Git 历史中撤销泄漏的 token。
