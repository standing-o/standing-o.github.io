---
title: "인과와 인과추론 | Causality and Causal Inference"
date: 2022-07-25 17:00:00 +/-TTTT
categories: [인공지능 | AI, AI 이론]
tags: [lg-aimers, causality, causal-inference, machine-learning]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: ❓ 인과추론(Causal Inference)의 기본 이론과 Back-door, Do-calculus와 같은 방법론들을 소개합니다.
---

-----------------------

> **<u>KEYWORDS</u>**     
> 인과, 인과추론, 인과추론 머신러닝, 인과추론 예시, 인과추론 AI, 인과추론 모델, Causality, Causal Inference, SCM, Back-door, Do-calculus
{: .prompt-info }

-----------------------

&nbsp;
&nbsp;
&nbsp;


## **Causality**
- Causality는 한 사건, 과정, 상태, 또는 객체가 다른 사건, 과정, 상태, 또는 객체의 발생에 기여하는 영향을 의미하며, 원인은 결과에 일부 책임이 있고 결과는 원인에 일부 의존하다는 특징을 보여줍니다.
- **다양한 학문 분야에서의 인과관계**
  - **자연과학** ㅣ 물리학, 화학, 생물학, 기후과학
  - **사회과학** ㅣ 심리학, 사회학, 경제학
  - **보건학** ㅣ 역학, 공중보건

- **AI, ML, Data Science와의 관련성**
  - **AI** ㅣ 목표를 달성하기 위해 행동을 수행하는 합리적인 에이전트 (강화학습)
  - **ML** ㅣ 현재는 주로 상관관계 학습에 초점
  - **DS** ㅣ 데이터를 수집, 처리, 분석, 커뮤니케이션하는 과정


&nbsp;
&nbsp;
&nbsp;


## **구조적 인과 모델 <sup>Structural Causal Model, SCM</sup>**

- SCM $$M = <U,V,F,P(U)>$$는 인과관계를 형식적으로 설명할 수 있으며, 관찰(Observation), 개입(Intervention), 반사실 분포(Counteractual Distribution)를 유도합니다.
- SCM은 인과 그래프 $$g$$를 유도하며, 이는 **d-분리(d-separation)**를 통해 조건부 독립성을 테스트할 수 있게 합니다.
  - 모델 $$M$$ 자체는 알 수 없지만, 인과 그래프 $$g$$는 상식 또는 도메인 지식으로부터 주어질 수 있습니다.
- 개입(Intervention) $$do(X=x)$$는 서브모델 $$M_x$$으로 표현되며, 조작된 인과 그래프 $$g_{\bar{x}}$$를 유도합니다.
- $$X=x$$가 $$Y=y$$에 미치는 인과 효과(Causal Effect)는 $$P(y\mid{do(x)})$$로 정의합니다.



#### **Remark**

- **Identifiability** ㅣ 일부 인과 그래프에서는 기존 관찰 데이터로부터 인과 효과를 계산할 수 있습니다.
- Markovian 경우 단일 $$X$$에서 인과 효과는 $$P(x\mid{pa_x})$$를 제거하여 쉽게 도출할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


## **Back-door Criterion**

- **<u>Definition</u>**ㅣ**Back-door**

  - 변수 $$X$$와 $$Y$$사이의 혼란(Confounding)을 충분히 설명할 수 있는 집합 $$Z$$를 찾는 것을 의미합니다.
  - 이때, $$Z$$가 Back-door 기준을 만족하면:

  $$
  P(y|do(x))=\sum_Z{P(y|x,z)P(z)}
  $$
  - $$do(x)$$는 $$X$$에 대한 개입(Intervention)을 나타냅니다.


- **<u>Definition</u>ㅣBack-door criterion**
  - 인과 그래프 $$g$$에서 변수 $$X$$와 $$Y$$의 쌍에 대하여, 집합 $$Z$$가 Back-door Criterion을 만족하려면
    - (i) $$Z$$의 노드는 $$X$$의 하위 노드(Descendant)가 아니어야 하며,
    - (ii) $$Z$$는 $$X$$로 들어오는 화살표를 포함한 모든 경로를 차단(Block)해야 합니다. 
- 이 공식은 간단하고 널리 사용되지만, 모든 인과 그래프에 적용할 수는 없는 한계가 있습니다.
  - Back-door Criterion이 만족되지 않는 복잡한 인과 그래프의 경우에는 Do-calculus 같은 추가적인 도구가 필요합니다.


### **Back-door sets as substitutes of the direct parents of $$X$$**

- ex. Rain은 Sprinkler와 Wet 사이에서 Back-door Criterion을 만족합니다.
  - (i) Rain은 Sprinkler의 하위 노드가 아니며,
  - (ii) Rain은 Sprinkler에서 Wet으로 향하는 유일한 Back-door 경로를 차단합니다.
- 따라서, Sprinkler의 직접 부모(Direct Parent)를 조정하여 다음과 같이 계산이 가능합니다.
  
$$
P(\text{wt}|do(\text{sp}))=\sum_\text{sn}P(\text{wt}|\text{sp,sn})P(\text{sn})=\cdots=\sum_\text{rn}P(\text{wt}|\text{sp,rn})P(\text{rn})
$$



&nbsp;
&nbsp;
&nbsp;



## **Rules of Do-calculus**

- Back-door Criterion은 특정 상황에서 매우 구체적인 형태의 인과 효과 계산 공식을 보여줍니다.

- Do-calculus(Pearl, 1995)는 관찰 분포와 개입 분포를 변환하거나 조작할 수 있는 일반화된 수학 공식입니다.

- **<u>Theorem</u>ㅣRules of Do-calculus (simplified)**

  - **Rule 1** ㅣ Adding/Removing Observations

  $$
  P(y|do(x),z)=P(y|do(x))\,\,\,\text{if}\,\,(Z\perp{Y|X})\,\,\text{in}\,\,g_{\bar{X}}
  $$

  - **Rule 2** ㅣ Action/Observation Exchange

  $$
  P(y|do(x),do(z))=P(y|do(x),z)\,\,\,\text{if}\,\,(Z\perp{Y|X})\,\,\text{in}\,\,g_{\bar{X}\underline{Z}}
  $$

  - **Rule 3** ㅣ Adding/Removing Actions

  $$
  P(y|do(x),do(z))=P(y|do(x))\,\,\,\text{if}\,\,(Z\perp{Y|X})\,\,\text{in}\,\,g_{\bar{X}\bar{Z}}
  $$


- Do-calculus는 **정확성(Soundness)**과 **완전성(Completeness)**을 가지지만, 알고리즘적인 통찰력을 보여주지는 않습니다.

- 식별 가능성(Identifiability)을 위해 그래프 조건과 효율적인 알고리즘 절차가 개발되었습니다.

- Do-calculus는 관찰적 확률(Observational Probability)과 개입적 확률(Interventional Probability)을 조작하기 위한 규칙의 집합입니다. (Do-calculus는 완전성을 보장합니다.)


&nbsp;
&nbsp;
&nbsp;

## **현대의 Identification 과제**
- **실험적 조건 ➔ 일반화된 식별(Generalized Identification)**
  - 서로 다른 실험 조건에서 얻게된 데이터셋을 결합합니다.
  - $$P(y\mid{do(x), z})$$형태의 표현이 식별 가능한지 여부는, 주어진 인과 그래프 $$g$$와 관찰적 및 실험적 연구의 임의 조합을 통해 결정할 수 있습니다.
  - 쿼리가 식별 가능하다면, 해당 추정량(Estimand)을 다항 시간(Polynomial Time) 내에 도출할 수 있습니다.

- **환경적 조건 ➔ 전달 가능성(Transportability)**
  - 서로 다른 소스에서 얻어진 데이터셋을 결합합니다.
  - 비모수(Non-parametric) 전달 가능성은 문제 인스턴스가 선택 다이어그램(Selection Diagrams)으로 인코딩된 경우 결정할 수 있습니다.
  - 전달 가능성이 가능하다면, 전달 공식(Transport Formula)을 다항 시간 내에 도출할 수 있습니다.
  - 인과 계산법(Causal Calculus)과 해당 전달 알고리즘은 완전성을 보장합니다.

- **샘플링 조건 ➔ 선택 편향(Selection Bias)으로부터의 복구**
  - 인과 및 통계적 설정에서 선택 편향의 비모수적 복구 가능성은 확장된 인과 그래프가 제공된 경우 결정할 수 있습니다.
  - 복구 가능성을 활용할 수 있다면, 추정값을 다항 시간 내에 도출할 수 있습니다.
  - 결과는 순수 복구 가능성(Pure Recoverability)에서는 완전하며, 외부 정보를 포함한 복구 가능성에 대해서는 충분합니다.

- **응답 조건(Responding Conditons) ➔ 결측(Missingness)으로부터의 복구**


&nbsp;
&nbsp;
&nbsp;


-----------
## Reference
> 본 포스팅은 LG Aimers 프로그램에서 학습한 내용을 기반으로 작성된것입니다. (전체 내용 X)
{: .prompt-warning }

1. LG Aimers AI Essential Course Module 5. 인과추론, 서울대학교 이상학 교수 


