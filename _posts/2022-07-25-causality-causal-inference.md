---
title: "인과 | Causality 와 인과추론 | Causal inference"
date: 2022-07-25 17:00:00 +/-TTTT
categories: [Causal inference]
tags: [lg-aimers, causality, scm, back-door, do-calculus]
math: true
---



-----------------------

- 본 포스팅은 인과, 인과추론의 개념과 관련 이론 (Back-door, Do-calculus) 들을 소개하고 있습니다.
- Keyword : Causality, SCM, Back-door, Do-calculus


## **Causality**

- Influence by shich one event, process, state, or object a contributes to the production of another event, process, state, or object where the cause is partly responsible for the effect, and the effect is partly dependent on the cause.
- Causality in various academic disciplines
  - Physics, chemistry,biology, climate science
  - Psychology, social science, economics
  - Epidemiology, public health
- Relation to AI, ML, DS
  - AI : a rational agent performing actions to achieve a goal (reinforcement learning)
  - ML : currently focused on learning correlations
  - DS : capture, process, analyze, communicate with data



## **Structural causal model (SCM)**

- SCM $$M = <U,V,F,P(U)>$$ provides a formal framework.
- SCM induces observational, interventional, and counterfactual distributions.
- SCM induces a causal graph $$g$$, which implies conditional independencies testable via d-separation (blockage).
- The underlying model $$M$$ is unknown but the causal graph $$g$$ can be given from common sense or domain knowledge.
- Intervention do(X=x) as a submodel M<sub>x</sub>, which induces a manipulated causal graph $$g_\bar{x}$$.
- Causal effect of $$X=x$$ on $$Y=y$$ is defined as $$P(y\mid{do(x)})$$.



### **Remark**

- Identifiability : causal effect may be computable from existing observational data for some causal graphs.
- In a Markovian case an singleton X, a causal effect can be easily derivable by canceling output $$P(x\mid{pa_x})$$



--------------------

## **Back-door Criterion**

- **<u>Definition</u>**ㅣ**Back-door**

  - Find a set $$Z$$ s.t. it can sufficiently explain 'confounding' between $$X$$ and $$Y$$. Then, 

  $$
  P(y|do(x))=\sum_Z{P(y|x,z)P(z)}
  $$

- **<u>Definition</u>ㅣBack-door criterion**
  
  - A set $$Z$$ satisfies the back-door criterion with respect to a pair of variables $$X, Y$$ in causal diagram $$g$$ if;
    - (i) no node in $$Z$$ is a descendant of $$X$$; and
    - (ii) $Z$ blocks every path between X ∈ $$X$$ and Y ∈ $$Y$$ that contains an arrow into X.
- A back-door adjustment formula is simple and widely used but limited.



### **Back-door sets as substitutes of the direct parents of X**

- Rain satisfies the back-door criterion relative to Sprinkler ans Wet:
  - (i) Rain is not descendant of Sprinkler, and
  - (ii) Rain blocks the only back-door path from Sprinkler to Wet.
- Adjusting for the direct parents of Sprinkler, we have:
  
$$
P(\text{wt}|do(\text{sp}))=\sum_\text{sn}P(\text{wt}|\text{sp,sn})P(\text{sn})=\cdots=\sum_\text{rn}P(\text{wt}|\text{sp,rn})P(\text{rn})
$$


---------------------

## **Rules of Do-calculus**

- Backdoor criterion results in a very specific form of indentification formula.

- Do-calculus (Pearl, 1995) provides general machinery to manipulate observational and interventional distributions.

- **<u>Theorem</u>ㅣRules of Do-calculus (simplified)**

  - Rule 1 : Adding/removing observations

  $$
  P(y|do(x),z)=P(y|do(x))\,\,\,\text{if}\,\,(Z\perp{Y|X})\,\,\text{in}\,\,g_{\bar{X}}
  $$

  - Rule 2 : Action/observation exchange

  $$
  P(y|do(x),do(z))=P(y|do(x),z)\,\,\,\text{if}\,\,(Z\perp{Y|X})\,\,\text{in}\,\,g_{\bar{X}\underline{Z}}
  $$

  - Rule 3 : Adding/removing actions

  $$
  P(y|do(x),do(z))=P(y|do(x))\,\,\,\text{if}\,\,(Z\perp{Y|X})\,\,\text{in}\,\,g_{\bar{X}\bar{Z}}
  $$

  

- Do-calculus is sound and complete but it has no algorithmic insight
- A graphical condition and an efficient algorithmic procedure have developed for identifiability.

- Do-calculus is a set of rules to manipulate observational or interventional probabilites. (Do-calculus is complete)



---------------------------------------------

## **Modern identification tasks**

- Experimental conditions ➔ **Generalized identification**

  - Combining datasets of different experimental conditions

  - The identifiability of any expression of the form $$P(y\mid{do(x), z})$$ can be determined given any causal graph $$g$$ and an arbitrary combination of observational and experimental studies.
  - If the query is identifiable, then its estimand can be derived in polynomial time.

- Environmental conditions ➔ **Transportability**

  - Combining datasets from different sources

  - Non-parametric transportability can be determined provided that the problem instance is encoded in selection diagrams.
  - When transportability is feasible, the transport formula can be derived in polynomial time.
  - The causal calculus and the corresponding transportation algorithm are complete.

- Sampling conditons ➔ Recovering from **selection bias**

  - Nonparametric recoverability of selection bias from causal and statistical settings can be determined provided that an augmented causal graph is available.
  - When recoverability is feasible, the estimated can be derived in polynomial time.
  - The result is complete for pure recoverability, and sufficient for recoverability with external information.

- Responding conditons ➔ Recovering from **missingness**



----

#### **References**
- 본 포스팅은 `LG Aimers` 프로그램에 참가하여 학습한 내용을 기반으로 작성된것입니다. (전체내용 X)

➔ [`LG Aimers` 바로가기](https://www.lgaimers.ai/)

```
[1] LG Aimers AI Essential Course Module 5.인과추론, 서울대학교 이상학 교수 
```

