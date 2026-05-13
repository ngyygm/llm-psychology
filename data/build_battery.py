#!/usr/bin/env python3
"""
Build the unified psychometric battery data file (items_battery.json).

Combines items from 4 validated instruments:
1. IPIP-NEO-120 (120 items, 5-point Likert)
2. SD3 Short Dark Triad (27 items, 5-point Likert)
3. ZKPQ-50-CC (50 items, True/False)
4. EPQR-A (24 items, Yes/No)

All items include: scale, domain, facet (where applicable), text, keyed direction,
and full citation information.
"""

import json
from pathlib import Path

# =============================================================================
# IPIP-NEO-120
# Source: Johnson (2014), Table 1. Public domain via ipip.ori.org
# =============================================================================

IPIP_120_ITEMS = [
    # Cycle 1: N1,E1,O1,A1,C1
    {"item": 1,   "text": "Worry about things",                              "domain": "Neuroticism",         "facet": "N1 Anxiety",            "keyed": "+"},
    {"item": 2,   "text": "Make friends easily",                             "domain": "Extraversion",        "facet": "E1 Friendliness",       "keyed": "+"},
    {"item": 3,   "text": "Have a vivid imagination",                        "domain": "Openness",            "facet": "O1 Imagination",        "keyed": "+"},
    {"item": 4,   "text": "Trust others",                                    "domain": "Agreeableness",       "facet": "A1 Trust",              "keyed": "+"},
    {"item": 5,   "text": "Complete tasks successfully",                     "domain": "Conscientiousness",   "facet": "C1 Self-Efficacy",      "keyed": "+"},
    # Cycle 1: N2,E2,O2,A2,C2
    {"item": 6,   "text": "Get angry easily",                                "domain": "Neuroticism",         "facet": "N2 Anger",              "keyed": "+"},
    {"item": 7,   "text": "Love large parties",                              "domain": "Extraversion",        "facet": "E2 Gregariousness",     "keyed": "+"},
    {"item": 8,   "text": "Believe in the importance of art",               "domain": "Openness",            "facet": "O2 Artistic Interests", "keyed": "+"},
    {"item": 9,   "text": "Use others for my own ends",                      "domain": "Agreeableness",       "facet": "A2 Morality",           "keyed": "-"},
    {"item": 10,  "text": "Like to tidy up",                                 "domain": "Conscientiousness",   "facet": "C2 Orderliness",        "keyed": "+"},
    # Cycle 1: N3,E3,O3,A3,C3
    {"item": 11,  "text": "Often feel blue",                                 "domain": "Neuroticism",         "facet": "N3 Depression",         "keyed": "+"},
    {"item": 12,  "text": "Take charge",                                     "domain": "Extraversion",        "facet": "E3 Assertiveness",      "keyed": "+"},
    {"item": 13,  "text": "Experience my emotions intensely",                "domain": "Openness",            "facet": "O3 Emotionality",       "keyed": "+"},
    {"item": 14,  "text": "Love to help others",                             "domain": "Agreeableness",       "facet": "A3 Altruism",           "keyed": "+"},
    {"item": 15,  "text": "Keep my promises",                                "domain": "Conscientiousness",   "facet": "C3 Dutifulness",        "keyed": "+"},
    # Cycle 1: N4,E4,O4,A4,C4
    {"item": 16,  "text": "Find it difficult to approach others",            "domain": "Neuroticism",         "facet": "N4 Self-Consciousness", "keyed": "+"},
    {"item": 17,  "text": "Am always busy",                                  "domain": "Extraversion",        "facet": "E4 Activity Level",     "keyed": "+"},
    {"item": 18,  "text": "Prefer variety to routine",                       "domain": "Openness",            "facet": "O4 Adventurousness",    "keyed": "+"},
    {"item": 19,  "text": "Love a good fight",                               "domain": "Agreeableness",       "facet": "A4 Cooperation",        "keyed": "-"},
    {"item": 20,  "text": "Work hard",                                       "domain": "Conscientiousness",   "facet": "C4 Achievement-Striving","keyed": "+"},
    # Cycle 1: N5,E5,O5,A5,C5
    {"item": 21,  "text": "Go on binges",                                    "domain": "Neuroticism",         "facet": "N5 Immoderation",       "keyed": "+"},
    {"item": 22,  "text": "Love excitement",                                 "domain": "Extraversion",        "facet": "E5 Excitement Seeking", "keyed": "+"},
    {"item": 23,  "text": "Love to read challenging material",               "domain": "Openness",            "facet": "O5 Intellect",          "keyed": "+"},
    {"item": 24,  "text": "Believe that I am better than others",            "domain": "Agreeableness",       "facet": "A5 Modesty",            "keyed": "-"},
    {"item": 25,  "text": "Have excellent ideas",                            "domain": "Conscientiousness",   "facet": "C5 Self-Discipline",    "keyed": "+"},
    # Cycle 1: N6,E6,O6,A6,C6
    {"item": 26,  "text": "Panic easily",                                    "domain": "Neuroticism",         "facet": "N6 Vulnerability",      "keyed": "+"},
    {"item": 27,  "text": "Radiate joy",                                     "domain": "Extraversion",        "facet": "E6 Cheerfulness",       "keyed": "+"},
    {"item": 28,  "text": "Tend to vote for liberal political candidates",   "domain": "Openness",            "facet": "O6 Liberalism",         "keyed": "+"},
    {"item": 29,  "text": "Sympathize with the homeless",                    "domain": "Agreeableness",       "facet": "A6 Sympathy",           "keyed": "+"},
    {"item": 30,  "text": "Rush into things",                                "domain": "Conscientiousness",   "facet": "C6 Cautiousness",       "keyed": "-"},
    # --- Cycle 2 (items 31-60) repeats same facet order ---
    {"item": 31,  "text": "Fear for the worst",                              "domain": "Neuroticism",         "facet": "N1 Anxiety",            "keyed": "+"},
    {"item": 32,  "text": "Feel comfortable around people",                  "domain": "Extraversion",        "facet": "E1 Friendliness",       "keyed": "+"},
    {"item": 33,  "text": "Enjoy wild flights of fantasy",                   "domain": "Openness",            "facet": "O1 Imagination",        "keyed": "+"},
    {"item": 34,  "text": "Believe that others have good intentions",        "domain": "Agreeableness",       "facet": "A1 Trust",              "keyed": "+"},
    {"item": 35,  "text": "Excel in what I do",                              "domain": "Conscientiousness",   "facet": "C1 Self-Efficacy",      "keyed": "+"},
    {"item": 36,  "text": "Lose my temper",                                  "domain": "Neuroticism",         "facet": "N2 Anger",              "keyed": "+"},
    {"item": 37,  "text": "Talk to a lot of different people at parties",    "domain": "Extraversion",        "facet": "E2 Gregariousness",     "keyed": "+"},
    {"item": 38,  "text": "See beauty in things that others might not notice","domain": "Openness",            "facet": "O2 Artistic Interests", "keyed": "+"},
    {"item": 39,  "text": "Cheat to get ahead",                              "domain": "Agreeableness",       "facet": "A2 Morality",           "keyed": "-"},
    {"item": 40,  "text": "Often forget to put things back in their proper place","domain": "Conscientiousness",   "facet": "C2 Orderliness",        "keyed": "-"},
    {"item": 41,  "text": "Dislike myself",                                  "domain": "Neuroticism",         "facet": "N3 Depression",         "keyed": "+"},
    {"item": 42,  "text": "Try to lead others",                              "domain": "Extraversion",        "facet": "E3 Assertiveness",      "keyed": "+"},
    {"item": 43,  "text": "Feel others' emotions",                           "domain": "Openness",            "facet": "O3 Emotionality",       "keyed": "+"},
    {"item": 44,  "text": "Am concerned about others",                       "domain": "Agreeableness",       "facet": "A3 Altruism",           "keyed": "+"},
    {"item": 45,  "text": "Tell the truth",                                  "domain": "Conscientiousness",   "facet": "C3 Dutifulness",        "keyed": "+"},
    {"item": 46,  "text": "Am afraid that I will do the wrong thing",        "domain": "Neuroticism",         "facet": "N4 Self-Consciousness", "keyed": "+"},
    {"item": 47,  "text": "Am always on the go",                             "domain": "Extraversion",        "facet": "E4 Activity Level",     "keyed": "+"},
    {"item": 48,  "text": "Prefer to stick with things that I know",         "domain": "Openness",            "facet": "O4 Adventurousness",    "keyed": "-"},
    {"item": 49,  "text": "Yell at people",                                  "domain": "Agreeableness",       "facet": "A4 Cooperation",        "keyed": "-"},
    {"item": 50,  "text": "Do more than what's expected of me",              "domain": "Conscientiousness",   "facet": "C4 Achievement-Striving","keyed": "+"},
    {"item": 51,  "text": "Rarely overindulge",                              "domain": "Neuroticism",         "facet": "N5 Immoderation",       "keyed": "-"},
    {"item": 52,  "text": "Seek adventure",                                  "domain": "Extraversion",        "facet": "E5 Excitement Seeking", "keyed": "+"},
    {"item": 53,  "text": "Avoid philosophical discussions",                 "domain": "Openness",            "facet": "O5 Intellect",          "keyed": "-"},
    {"item": 54,  "text": "Have a high opinion of myself",                   "domain": "Agreeableness",       "facet": "A5 Modesty",            "keyed": "-"},
    {"item": 55,  "text": "Carry out my plans",                              "domain": "Conscientiousness",   "facet": "C5 Self-Discipline",    "keyed": "+"},
    {"item": 56,  "text": "Become overwhelmed by events",                    "domain": "Neuroticism",         "facet": "N6 Vulnerability",      "keyed": "+"},
    {"item": 57,  "text": "Have a lot of fun",                               "domain": "Extraversion",        "facet": "E6 Cheerfulness",       "keyed": "+"},
    {"item": 58,  "text": "Believe that there is no absolute right or wrong","domain": "Openness",            "facet": "O6 Liberalism",         "keyed": "+"},
    {"item": 59,  "text": "Feel sympathy for those who are worse off than myself","domain": "Agreeableness",       "facet": "A6 Sympathy",           "keyed": "+"},
    {"item": 60,  "text": "Make rash decisions",                             "domain": "Conscientiousness",   "facet": "C6 Cautiousness",       "keyed": "-"},
    # --- Cycle 3 (items 61-90) ---
    {"item": 61,  "text": "Am easily disturbed",                             "domain": "Neuroticism",         "facet": "N1 Anxiety",            "keyed": "+"},
    {"item": 62,  "text": "Act comfortable with others",                     "domain": "Extraversion",        "facet": "E1 Friendliness",       "keyed": "+"},
    {"item": 63,  "text": "Enjoy daydreaming",                               "domain": "Openness",            "facet": "O1 Imagination",        "keyed": "+"},
    {"item": 64,  "text": "Trust what people say",                           "domain": "Agreeableness",       "facet": "A1 Trust",              "keyed": "+"},
    {"item": 65,  "text": "Handle tasks smoothly",                           "domain": "Conscientiousness",   "facet": "C1 Self-Efficacy",      "keyed": "+"},
    {"item": 66,  "text": "Lose my composure",                               "domain": "Neuroticism",         "facet": "N2 Anger",              "keyed": "+"},
    {"item": 67,  "text": "Enjoy parties",                                   "domain": "Extraversion",        "facet": "E2 Gregariousness",     "keyed": "+"},
    {"item": 68,  "text": "Like music",                                      "domain": "Openness",            "facet": "O2 Artistic Interests", "keyed": "+"},
    {"item": 69,  "text": "Obstruct others' plans",                          "domain": "Agreeableness",       "facet": "A2 Morality",           "keyed": "-"},
    {"item": 70,  "text": "Leave a mess in my room",                         "domain": "Conscientiousness",   "facet": "C2 Orderliness",        "keyed": "-"},
    {"item": 71,  "text": "Feel comfortable with myself",                    "domain": "Neuroticism",         "facet": "N3 Depression",         "keyed": "-"},
    {"item": 72,  "text": "Wait for others to lead the way",                 "domain": "Extraversion",        "facet": "E3 Assertiveness",      "keyed": "-"},
    {"item": 73,  "text": "Enjoy thinking about things",                     "domain": "Openness",            "facet": "O3 Emotionality",       "keyed": "+"},
    {"item": 74,  "text": "Am not interested in other people's problems",    "domain": "Agreeableness",       "facet": "A3 Altruism",           "keyed": "-"},
    {"item": 75,  "text": "Break rules",                                     "domain": "Conscientiousness",   "facet": "C3 Dutifulness",        "keyed": "-"},
    {"item": 76,  "text": "Only feel comfortable with friends",              "domain": "Neuroticism",         "facet": "N4 Self-Consciousness", "keyed": "+"},
    {"item": 77,  "text": "Do a lot in my spare time",                       "domain": "Extraversion",        "facet": "E4 Activity Level",     "keyed": "+"},
    {"item": 78,  "text": "Dislike changes",                                 "domain": "Openness",            "facet": "O4 Adventurousness",    "keyed": "-"},
    {"item": 79,  "text": "Insult people",                                   "domain": "Agreeableness",       "facet": "A4 Cooperation",        "keyed": "-"},
    {"item": 80,  "text": "Set high standards for myself",                   "domain": "Conscientiousness",   "facet": "C4 Achievement-Striving","keyed": "+"},
    {"item": 81,  "text": "Am able to control my cravings",                  "domain": "Neuroticism",         "facet": "N5 Immoderation",       "keyed": "-"},
    {"item": 82,  "text": "Act wild and crazy",                              "domain": "Extraversion",        "facet": "E5 Excitement Seeking", "keyed": "+"},
    {"item": 83,  "text": "Am not interested in abstract ideas",             "domain": "Openness",            "facet": "O5 Intellect",          "keyed": "-"},
    {"item": 84,  "text": "Boast about my virtues",                          "domain": "Agreeableness",       "facet": "A5 Modesty",            "keyed": "-"},
    {"item": 85,  "text": "Get to work at once",                             "domain": "Conscientiousness",   "facet": "C5 Self-Discipline",    "keyed": "+"},
    {"item": 86,  "text": "Can't make up my mind",                           "domain": "Neuroticism",         "facet": "N6 Vulnerability",      "keyed": "+"},
    {"item": 87,  "text": "Look at the bright side of life",                 "domain": "Extraversion",        "facet": "E6 Cheerfulness",       "keyed": "+"},
    {"item": 88,  "text": "Believe that we should be tough on crime",        "domain": "Openness",            "facet": "O6 Liberalism",         "keyed": "-"},
    {"item": 89,  "text": "Am indifferent to the feelings of others",        "domain": "Agreeableness",       "facet": "A6 Sympathy",           "keyed": "-"},
    {"item": 90,  "text": "Jump into things without thinking",               "domain": "Conscientiousness",   "facet": "C6 Cautiousness",       "keyed": "-"},
    # --- Cycle 4 (items 91-120) ---
    {"item": 91,  "text": "Am stressed out",                                 "domain": "Neuroticism",         "facet": "N1 Anxiety",            "keyed": "+"},
    {"item": 92,  "text": "Am skilled in handling social situations",        "domain": "Extraversion",        "facet": "E1 Friendliness",       "keyed": "+"},
    {"item": 93,  "text": "Like to get lost in thought",                     "domain": "Openness",            "facet": "O1 Imagination",        "keyed": "+"},
    {"item": 94,  "text": "Distrust people",                                 "domain": "Agreeableness",       "facet": "A1 Trust",              "keyed": "-"},
    {"item": 95,  "text": "Know how to get things done",                     "domain": "Conscientiousness",   "facet": "C1 Self-Efficacy",      "keyed": "+"},
    {"item": 96,  "text": "Am not easily annoyed",                           "domain": "Neuroticism",         "facet": "N2 Anger",              "keyed": "-"},
    {"item": 97,  "text": "Avoid crowds",                                    "domain": "Extraversion",        "facet": "E2 Gregariousness",     "keyed": "-"},
    {"item": 98,  "text": "Do not like art",                                 "domain": "Openness",            "facet": "O2 Artistic Interests", "keyed": "-"},
    {"item": 99,  "text": "Take advantage of others",                        "domain": "Agreeableness",       "facet": "A2 Morality",           "keyed": "-"},
    {"item": 100, "text": "Want everything to be 'just right'",              "domain": "Conscientiousness",   "facet": "C2 Orderliness",        "keyed": "+"},
    {"item": 101, "text": "Feel desperate",                                  "domain": "Neuroticism",         "facet": "N3 Depression",         "keyed": "+"},
    {"item": 102, "text": "Am the life of the party",                        "domain": "Extraversion",        "facet": "E3 Assertiveness",      "keyed": "+"},
    {"item": 103, "text": "Seldom get emotional",                            "domain": "Openness",            "facet": "O3 Emotionality",       "keyed": "-"},
    {"item": 104, "text": "Am not really interested in others",              "domain": "Agreeableness",       "facet": "A3 Altruism",           "keyed": "-"},
    {"item": 105, "text": "Behave properly",                                 "domain": "Conscientiousness",   "facet": "C3 Dutifulness",        "keyed": "+"},
    {"item": 106, "text": "Am afraid of many things",                        "domain": "Neuroticism",         "facet": "N4 Self-Consciousness", "keyed": "+"},
    {"item": 107, "text": "Prefer to be alone",                              "domain": "Extraversion",        "facet": "E4 Activity Level",     "keyed": "-"},
    {"item": 108, "text": "Dislike new foods",                               "domain": "Openness",            "facet": "O4 Adventurousness",    "keyed": "-"},
    {"item": 109, "text": "Hold a grudge",                                   "domain": "Agreeableness",       "facet": "A4 Cooperation",        "keyed": "-"},
    {"item": 110, "text": "Do just enough work to get by",                   "domain": "Conscientiousness",   "facet": "C4 Achievement-Striving","keyed": "-"},
    {"item": 111, "text": "Easily resist temptations",                       "domain": "Neuroticism",         "facet": "N5 Immoderation",       "keyed": "-"},
    {"item": 112, "text": "Seek danger",                                     "domain": "Extraversion",        "facet": "E5 Excitement Seeking", "keyed": "+"},
    {"item": 113, "text": "Try to understand myself",                        "domain": "Openness",            "facet": "O5 Intellect",          "keyed": "+"},
    {"item": 114, "text": "Think highly of myself",                          "domain": "Agreeableness",       "facet": "A5 Modesty",            "keyed": "-"},
    {"item": 115, "text": "Get chores done right away",                      "domain": "Conscientiousness",   "facet": "C5 Self-Discipline",    "keyed": "+"},
    {"item": 116, "text": "Feel that my life lacks direction",               "domain": "Neuroticism",         "facet": "N6 Vulnerability",      "keyed": "+"},
    {"item": 117, "text": "Am always joking around",                         "domain": "Extraversion",        "facet": "E6 Cheerfulness",       "keyed": "+"},
    {"item": 118, "text": "Tend to vote for conservative political candidates","domain": "Openness",            "facet": "O6 Liberalism",         "keyed": "-"},
    {"item": 119, "text": "Am hard to understand",                           "domain": "Agreeableness",       "facet": "A6 Sympathy",           "keyed": "-"},
    {"item": 120, "text": "Stick to my chosen path",                         "domain": "Conscientiousness",   "facet": "C6 Cautiousness",       "keyed": "+"},
]

# =============================================================================
# SD3 - Short Dark Triad
# Source: Jones & Paulhus (2014), SD3.1.1 official instrument document
# =============================================================================

SD3_ITEMS = [
    # Machiavellianism (9 items)
    {"item": 1,  "text": "It's not wise to tell your secrets.",                              "domain": "Machiavellianism", "keyed": "+"},
    {"item": 2,  "text": "I like to use clever manipulation to get my way.",                  "domain": "Machiavellianism", "keyed": "+"},
    {"item": 3,  "text": "Whatever it takes, you must get the important people on your side.","domain": "Machiavellianism", "keyed": "+"},
    {"item": 4,  "text": "Avoid direct conflict with others because they may be useful in the future.", "domain": "Machiavellianism", "keyed": "+"},
    {"item": 5,  "text": "It's wise to keep track of information that you can use against people later.", "domain": "Machiavellianism", "keyed": "+"},
    {"item": 6,  "text": "You should wait for the right time to get back at people.",         "domain": "Machiavellianism", "keyed": "+"},
    {"item": 7,  "text": "There are things you should hide from other people because they don't need to know.", "domain": "Machiavellianism", "keyed": "+"},
    {"item": 8,  "text": "Make sure your plans benefit you, not others.",                     "domain": "Machiavellianism", "keyed": "+"},
    {"item": 9,  "text": "Most people can be manipulated.",                                   "domain": "Machiavellianism", "keyed": "+"},
    # Narcissism (9 items)
    {"item": 10, "text": "People see me as a natural leader.",                                "domain": "Narcissism",       "keyed": "+"},
    {"item": 11, "text": "I hate being the center of attention.",                             "domain": "Narcissism",       "keyed": "-"},
    {"item": 12, "text": "Many group activities tend to be dull without me.",                 "domain": "Narcissism",       "keyed": "+"},
    {"item": 13, "text": "I know that I am special because everyone keeps telling me so.",    "domain": "Narcissism",       "keyed": "+"},
    {"item": 14, "text": "I like to get acquainted with important people.",                   "domain": "Narcissism",       "keyed": "+"},
    {"item": 15, "text": "I feel embarrassed if someone compliments me.",                     "domain": "Narcissism",       "keyed": "-"},
    {"item": 16, "text": "I have been compared to famous people.",                            "domain": "Narcissism",       "keyed": "+"},
    {"item": 17, "text": "I am an average person.",                                           "domain": "Narcissism",       "keyed": "-"},
    {"item": 18, "text": "I insist on getting the respect I deserve.",                        "domain": "Narcissism",       "keyed": "+"},
    # Psychopathy (9 items)
    {"item": 19, "text": "I like to get revenge on authorities.",                             "domain": "Psychopathy",      "keyed": "+"},
    {"item": 20, "text": "I avoid dangerous situations.",                                     "domain": "Psychopathy",      "keyed": "-"},
    {"item": 21, "text": "Payback needs to be quick and nasty.",                              "domain": "Psychopathy",      "keyed": "+"},
    {"item": 22, "text": "People often say I'm out of control.",                              "domain": "Psychopathy",      "keyed": "+"},
    {"item": 23, "text": "It's true that I can be mean to others.",                           "domain": "Psychopathy",      "keyed": "+"},
    {"item": 24, "text": "People who mess with me always regret it.",                         "domain": "Psychopathy",      "keyed": "+"},
    {"item": 25, "text": "I have never gotten into trouble with the law.",                    "domain": "Psychopathy",      "keyed": "-"},
    {"item": 26, "text": "I enjoy having sex with people I hardly know.",                     "domain": "Psychopathy",      "keyed": "+"},
    {"item": 27, "text": "I'll say anything to get what I want.",                             "domain": "Psychopathy",      "keyed": "+"},
]

# =============================================================================
# ZKPQ-50-CC
# Source: Aluja et al. (2006), PsyToolkit verified
# =============================================================================

ZKPQ_ITEMS = [
    # Activity (10 items)
    {"item": 1,  "text": "I do not like to waste time just sitting around and relaxing.",      "domain": "Activity",                      "keyed": "+"},
    {"item": 2,  "text": "I lead a busier life than most people.",                             "domain": "Activity",                      "keyed": "+"},
    {"item": 3,  "text": "I like to be doing things all of the time.",                         "domain": "Activity",                      "keyed": "+"},
    {"item": 4,  "text": "I can enjoy myself just lying around and not doing anything active.","domain": "Activity",                      "keyed": "-"},
    {"item": 5,  "text": "I do not feel the need to be doing things all of the time.",         "domain": "Activity",                      "keyed": "-"},
    {"item": 6,  "text": "When on vacation I like to engage in active sports rather than just lie around.", "domain": "Activity",            "keyed": "+"},
    {"item": 7,  "text": "I like to wear myself out with hard work or exercise.",              "domain": "Activity",                      "keyed": "+"},
    {"item": 8,  "text": "I like to be active as soon as I wake up in the morning.",           "domain": "Activity",                      "keyed": "+"},
    {"item": 9,  "text": "I like to keep busy all the time.",                                  "domain": "Activity",                      "keyed": "+"},
    {"item": 10, "text": "When I do things, I do them with lots of energy.",                   "domain": "Activity",                      "keyed": "+"},
    # Aggression-Hostility (10 items)
    {"item": 11, "text": "When I get mad, I say ugly things.",                                 "domain": "Aggression-Hostility",          "keyed": "+"},
    {"item": 12, "text": "It's natural for me to curse when I am mad.",                        "domain": "Aggression-Hostility",          "keyed": "+"},
    {"item": 13, "text": "I almost never feel like I would like to hit someone.",              "domain": "Aggression-Hostility",          "keyed": "-"},
    {"item": 14, "text": "If someone offends me, I just try not to think about it.",           "domain": "Aggression-Hostility",          "keyed": "-"},
    {"item": 15, "text": "If people annoy me I do not hesitate to tell them so.",              "domain": "Aggression-Hostility",          "keyed": "+"},
    {"item": 16, "text": "When people disagree with me I cannot help getting into an argument with them.", "domain": "Aggression-Hostility","keyed": "+"},
    {"item": 17, "text": "I have a very strong temper.",                                       "domain": "Aggression-Hostility",          "keyed": "+"},
    {"item": 18, "text": "I can't help being a little rude to people I do not like.",          "domain": "Aggression-Hostility",          "keyed": "+"},
    {"item": 19, "text": "I am always patient with others even when they are irritating.",     "domain": "Aggression-Hostility",          "keyed": "-"},
    {"item": 20, "text": "When people shout at me, I shout back.",                             "domain": "Aggression-Hostility",          "keyed": "+"},
    # Impulsive Sensation Seeking (10 items)
    {"item": 21, "text": "I often do things on impulse.",                                      "domain": "Impulsive_Sensation_Seeking",   "keyed": "+"},
    {"item": 22, "text": "I would like to take off on a trip with no preplanned or definite routes or timetables.", "domain": "Impulsive_Sensation_Seeking", "keyed": "+"},
    {"item": 23, "text": "I enjoy getting into new situations where you can't predict how things will turn out.", "domain": "Impulsive_Sensation_Seeking", "keyed": "+"},
    {"item": 24, "text": "I sometimes like to do things that are a little frightening.",       "domain": "Impulsive_Sensation_Seeking",   "keyed": "+"},
    {"item": 25, "text": "I'll try anything once.",                                            "domain": "Impulsive_Sensation_Seeking",   "keyed": "+"},
    {"item": 26, "text": "I would like the kind of life where one is on the move and travelling a lot, with lots of change and excitement.", "domain": "Impulsive_Sensation_Seeking", "keyed": "+"},
    {"item": 27, "text": "I sometimes do \"crazy\" things just for fun.",                      "domain": "Impulsive_Sensation_Seeking",   "keyed": "+"},
    {"item": 28, "text": "I prefer friends who are excitingly unpredictable.",                 "domain": "Impulsive_Sensation_Seeking",   "keyed": "+"},
    {"item": 29, "text": "I often get so carried away by new and exciting things and ideas that I never think of possible complications.", "domain": "Impulsive_Sensation_Seeking", "keyed": "+"},
    {"item": 30, "text": "I like \"wild\" uninhibited parties.",                               "domain": "Impulsive_Sensation_Seeking",   "keyed": "+"},
    # Neuroticism-Anxiety (10 items)
    {"item": 31, "text": "My body often feels all tightened up for no apparent reason.",       "domain": "Neuroticism-Anxiety",           "keyed": "+"},
    {"item": 32, "text": "I frequently get emotionally upset.",                                "domain": "Neuroticism-Anxiety",           "keyed": "+"},
    {"item": 33, "text": "I tend to be oversensitive and easily hurt by thoughtless remarks and actions of others.", "domain": "Neuroticism-Anxiety", "keyed": "+"},
    {"item": 34, "text": "I am easily frightened.",                                            "domain": "Neuroticism-Anxiety",           "keyed": "+"},
    {"item": 35, "text": "I sometimes feel panicky.",                                          "domain": "Neuroticism-Anxiety",           "keyed": "+"},
    {"item": 36, "text": "I often feel unsure of myself.",                                     "domain": "Neuroticism-Anxiety",           "keyed": "+"},
    {"item": 37, "text": "I often worry about things that other people think are unimportant.","domain": "Neuroticism-Anxiety",           "keyed": "+"},
    {"item": 38, "text": "I often feel like crying sometimes without a reason.",               "domain": "Neuroticism-Anxiety",           "keyed": "+"},
    {"item": 39, "text": "I don't let a lot of trivial things irritate me.",                   "domain": "Neuroticism-Anxiety",           "keyed": "-"},
    {"item": 40, "text": "I often feel uncomfortable and ill at ease for no real reason.",     "domain": "Neuroticism-Anxiety",           "keyed": "+"},
    # Sociability (10 items)
    {"item": 41, "text": "I do not mind going out alone and usually prefer it to being out in a large group.", "domain": "Sociability",    "keyed": "-"},
    {"item": 42, "text": "I spend as much time with my friends as I can.",                     "domain": "Sociability",                   "keyed": "+"},
    {"item": 43, "text": "I do not need a large number of casual friends.",                    "domain": "Sociability",                   "keyed": "-"},
    {"item": 44, "text": "I tend to be uncomfortable at big parties.",                         "domain": "Sociability",                   "keyed": "-"},
    {"item": 45, "text": "At parties, I enjoy mingling with many people whether I already know them or not.", "domain": "Sociability",   "keyed": "+"},
    {"item": 46, "text": "I would not mind being socially isolated in some place for some period of time.", "domain": "Sociability",    "keyed": "-"},
    {"item": 47, "text": "Generally, I like to be alone so I can do things I want to do without social distractions.", "domain": "Sociability", "keyed": "-"},
    {"item": 48, "text": "I am a very sociable person.",                                       "domain": "Sociability",                   "keyed": "+"},
    {"item": 49, "text": "I usually prefer to do things alone.",                               "domain": "Sociability",                   "keyed": "-"},
    {"item": 50, "text": "I probably spend more time than I should socializing with friends.", "domain": "Sociability",                   "keyed": "+"},
]

# =============================================================================
# EPQR-A - Eysenck Personality Questionnaire Revised-Abbreviated
# Source: Francis, Brown, & Philipchalk (1992)
# Items from EPQ-R (Eysenck, Eysenck & Barrett, 1985) 100-item version
# =============================================================================

EPQR_A_ITEMS = [
    # Psychoticism (6 items)
    {"item": 1,  "text": "Would you take drugs which may have strange or dangerous effects?",      "domain": "Psychoticism", "keyed": "+"},
    {"item": 2,  "text": "Do you prefer to go your own way rather than act by the rules?",         "domain": "Psychoticism", "keyed": "+"},
    {"item": 3,  "text": "Do you enjoy hurting people you love?",                                 "domain": "Psychoticism", "keyed": "+"},
    {"item": 4,  "text": "Do you enjoy practical jokes that can sometimes really hurt people?",    "domain": "Psychoticism", "keyed": "+"},
    {"item": 5,  "text": "Do you sometimes talk about things you know nothing about?",             "domain": "Psychoticism", "keyed": "+"},
    {"item": 6,  "text": "Do you think marriage is old-fashioned and should be done away with?",   "domain": "Psychoticism", "keyed": "+"},
    # Extraversion (6 items)
    {"item": 7,  "text": "Are you a talkative person?",                                            "domain": "Extraversion", "keyed": "+"},
    {"item": 8,  "text": "Are you rather lively?",                                                 "domain": "Extraversion", "keyed": "+"},
    {"item": 9,  "text": "Can you usually let yourself go and enjoy yourself at a lively party?",  "domain": "Extraversion", "keyed": "+"},
    {"item": 10, "text": "Do you tend to keep in the background on social occasions?",             "domain": "Extraversion", "keyed": "-"},
    {"item": 11, "text": "Do you prefer reading to meeting people?",                               "domain": "Extraversion", "keyed": "-"},
    {"item": 12, "text": "Are you mostly quiet when you are with other people?",                   "domain": "Extraversion", "keyed": "-"},
    # Neuroticism (6 items)
    {"item": 13, "text": "Does your mood often go up and down?",                                   "domain": "Neuroticism",  "keyed": "+"},
    {"item": 14, "text": "Do you ever feel 'just miserable' for no reason?",                       "domain": "Neuroticism",  "keyed": "+"},
    {"item": 15, "text": "Are your feelings easily hurt?",                                         "domain": "Neuroticism",  "keyed": "+"},
    {"item": 16, "text": "Are you often troubled about feelings of guilt?",                        "domain": "Neuroticism",  "keyed": "+"},
    {"item": 17, "text": "Would you call yourself tense or 'highly-strung'?",                      "domain": "Neuroticism",  "keyed": "+"},
    {"item": 18, "text": "Do you worry about awful things that might happen?",                     "domain": "Neuroticism",  "keyed": "+"},
    # Lie (6 items)
    {"item": 19, "text": "If you say you will do something, do you always keep your promise no matter how inconvenient it might be?", "domain": "Lie", "keyed": "+"},
    {"item": 20, "text": "Were you ever greedy by helping yourself to more than your share of anything?",                           "domain": "Lie", "keyed": "-"},
    {"item": 21, "text": "Have you ever blamed someone for doing something you knew was really your fault?",                        "domain": "Lie", "keyed": "-"},
    {"item": 22, "text": "As a child did you do as you were told immediately and without grumbling?",                               "domain": "Lie", "keyed": "+"},
    {"item": 23, "text": "Do you always practice what you preach?",                                                                 "domain": "Lie", "keyed": "+"},
    {"item": 24, "text": "Are all your habits good and desirable ones?",                                                            "domain": "Lie", "keyed": "+"},
]


def build_battery():
    all_items = []

    # Add scale prefix to each item
    for i in IPIP_120_ITEMS:
        all_items.append({
            "id": f"ipip_{i['item']:03d}",
            "scale": "IPIP-NEO-120",
            "domain": i["domain"],
            "facet": i["facet"],
            "text": i["text"],
            "keyed": i["keyed"],
            "response_format": "likert_5",
        })

    for i in SD3_ITEMS:
        all_items.append({
            "id": f"sd3_{i['item']:03d}",
            "scale": "SD3",
            "domain": i["domain"],
            "facet": None,
            "text": i["text"],
            "keyed": i["keyed"],
            "response_format": "likert_5",
        })

    for i in ZKPQ_ITEMS:
        all_items.append({
            "id": f"zkpq_{i['item']:03d}",
            "scale": "ZKPQ-50-CC",
            "domain": i["domain"],
            "facet": None,
            "text": i["text"],
            "keyed": i["keyed"],
            "response_format": "true_false",
        })

    for i in EPQR_A_ITEMS:
        all_items.append({
            "id": f"epqr_{i['item']:03d}",
            "scale": "EPQR-A",
            "domain": i["domain"],
            "facet": None,
            "text": i["text"],
            "keyed": i["keyed"],
            "response_format": "yes_no",
        })

    # Validate counts
    assert len(all_items) == 221, f"Expected 221 items, got {len(all_items)}"
    assert len([i for i in all_items if i["scale"] == "IPIP-NEO-120"]) == 120
    assert len([i for i in all_items if i["scale"] == "SD3"]) == 27
    assert len([i for i in all_items if i["scale"] == "ZKPQ-50-CC"]) == 50
    assert len([i for i in all_items if i["scale"] == "EPQR-A"]) == 24

    # Validate keyed directions
    for item in all_items:
        assert item["keyed"] in ["+", "-"], f"Invalid keyed: {item['keyed']} for {item['id']}"

    battery = {
        "description": "Unified psychometric battery for LLM response-style measurement.",
        "total_items": 221,
        "scales": {
            "IPIP-NEO-120": {
                "citation": "Johnson, J. A. (2014). Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory. Journal of Research in Personality, 51, 78-89.",
                "url": "https://ipip.ori.org/newNEOKey.htm",
                "response_format": "likert_5",
                "response_labels": {"1": "Very Inaccurate", "2": "Moderately Inaccurate", "3": "Neither Accurate Nor Inaccurate", "4": "Moderately Accurate", "5": "Very Accurate"},
                "n_items": 120,
                "domains": ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"],
                "items_per_domain": 24,
                "n_reverse": 41,
            },
            "SD3": {
                "citation": "Jones, D. N., & Paulhus, D. L. (2014). Introducing the Short Dark Triad (SD3). Assessment, 21(1), 28-41. DOI: 10.1177/1073191113514105",
                "url": "https://www2.psych.ubc.ca/~dpaulhus/research/DARK_TRAITS/MEASURES/SD3.1.1.doc",
                "response_format": "likert_5",
                "response_labels": {"1": "Strongly Disagree", "2": "Disagree", "3": "Neutral", "4": "Agree", "5": "Strongly Agree"},
                "n_items": 27,
                "domains": ["Machiavellianism", "Narcissism", "Psychopathy"],
                "items_per_domain": 9,
                "n_reverse": 5,
            },
            "ZKPQ-50-CC": {
                "citation": "Aluja, A., Rossier, J., Garcia, L. F., Angleitner, A., Kuhlman, M., & Zuckerman, M. (2006). A cross-cultural shortened form of the ZKPQ. Personality and Individual Differences, 41(4), 619-628.",
                "url": "https://www.psytoolkit.org/survey-library/zkpq-50-cc.html",
                "response_format": "true_false",
                "response_labels": {"true": "True", "false": "False"},
                "n_items": 50,
                "domains": ["Activity", "Aggression-Hostility", "Impulsive_Sensation_Seeking", "Neuroticism-Anxiety", "Sociability"],
                "items_per_domain": 10,
                "n_reverse": 12,
            },
            "EPQR-A": {
                "citation": "Francis, L. J., Brown, L. B., & Philipchalk, R. (1992). The development of an abbreviated form of the Revised Eysenck Personality Questionnaire (EPQR-A). Personality and Individual Differences, 13(4), 443-449.",
                "url": "https://doi.org/10.1016/0191-8869(92)90073-X",
                "response_format": "yes_no",
                "response_labels": {"yes": "Yes", "no": "No"},
                "n_items": 24,
                "domains": ["Psychoticism", "Extraversion", "Neuroticism", "Lie"],
                "items_per_domain": 6,
                "n_reverse": 5,
            },
        },
        "items": all_items,
    }

    out = Path(__file__).parent / "items_battery.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(battery, f, indent=2, ensure_ascii=False)

    print(f"Written {len(all_items)} items to {out}")
    for scale, meta in battery["scales"].items():
        n = len([i for i in all_items if i["scale"] == scale])
        n_rev = len([i for i in all_items if i["scale"] == scale and i["keyed"] == "-"])
        print(f"  {scale}: {n} items ({n_rev} reverse-scored)")


if __name__ == "__main__":
    build_battery()
