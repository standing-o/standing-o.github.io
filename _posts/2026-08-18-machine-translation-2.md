---
title: "기계번역 톺아보기 (2) | LLM을 번역기로 쓰기"
date: 2026-08-18 00:00:00 +/-TTTT
categories: [인공지능 | AI, 자연어 | NLP]
tags: [machine-translation, llm, generative-ai, foundation-model, deep-learning]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
pin: false
description: 💬 LLM을 번역에 활용하는 LoRA와 QLoRA, 인간 선호 방법으로 학습하는 RLHF와 DPO, 그리고 그 평가지표를 공부합니다.
---


--------------------
> **<u>KEYWORDS</u>**         
> LLM 번역, LoRA, QLoRA, PEFT, 양자화, Continued Pretraining, SFT, 보상 모형, Bradley-Terry, PPO, GRPO, DPO, CPO, WPO, COMET, BLEURT, MQM, 메타평가
{: .prompt-info }
--------------------

<br>


> [(1)편 ㅣ 통계 모델과 Transformer](https://standing-o.github.io/posts/machine-translation-1/)
{: .prompt-tip }


<br>


## **Introduction**

- (1)편에서 통계 기계번역과 Transformer까지를 정리했습니다. 
  - 번역을 두 모델의 결합(SMT) 또는 하나의 미분 가능한 조건부 분포(NMT)로 두는 두 가지 접근을 다뤘습니다.
- 본 글은 LLM 시대의 기계번역을 세 단계로 정리합니다. 
  - 첫째, 디코더 전용 LLM을 번역에 활용하는 방법(LoRA, QLoRA, 2단계 미세조정). 
  - 둘째, 사람 선호로 학습하는 방법(RLHF, PPO, DPO). 
  - 셋째, 뉴럴 평가지표와 메타평가 이론입니다.


<br>


## **파라미터 효율 미세조정 <sup>Parameter-Efficient Fine-Tuning</sup>**

- LLM을 번역에 활용하려면 수십억 개의 파라미터를 업데이트해야 합니다. 
  - 전량 미세조정(Full Fine-Tuning)은 파라미터 전부를 학습하고 그만큼의 옵티마이저 상태를 메모리에 유지해야 하므로 비용이 매우 큽니다.
- **PEFT(Parameter-Efficient Fine-Tuning)** 는 갱신을 저차 부분공간으로 제한하거나 별도 모듈로 분리해서 그 비용을 줄입니다. 
  - 가장 널리 쓰이는 방법이 LoRA와 QLoRA입니다.


### **LoRA**

- **LoRA(Low-Rank Adaptation)** 는 가중치 갱신 $$\Delta W$$ 가 낮은 내재 차원(Intrinsic Dimension)을 갖는다 [^ref2]고 가정하고, 두 행렬의 곱으로 제약합니다 [^ref1].

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}\,B A
$$

- 여기서 각 요소는 다음과 같습니다.
  - $$W_0 \in \mathbb{R}^{d \times k}$$: 사전학습 가중치, 학습 중 **고정**
  - $$B \in \mathbb{R}^{d \times r}$$, $$A \in \mathbb{R}^{r \times k}$$: 학습 대상, $$r \ll \min(d, k)$$
  - $$\alpha$$: 축척 계수(Scaling Factor)
- 파라미터 수가 $$dk$$ 에서 $$r(d + k)$$ 로 줄어듭니다. $$d = k = 4096$$, $$r = 8$$ 이면 약 $$16.8M$$ 에서 $$65K$$ 로 감소합니다.


### **초기화의 의미**

- LoRA의 초기화 전략이 중요합니다. 
  - **$$A$$ 는 정규분포로, $$B$$ 는 0으로 초기화**합니다. 
  - 그러면 학습 시작 시점에 $$\Delta W = BA = 0$$ 이므로 모형이 **사전학습 모형과 정확히 같습니다**.
- 이는 어댑터가 $$W_{\text{up}}$$ 을 0으로 두는 설계와 같은 이유입니다. 
  - 학습 초기에 사전학습 성능을 보존하면서 점진적으로 갱신 방향을 학습하도록 만듭니다.


### **스케일링 계수 $$\alpha/r$$ 의 역할**

- $$\alpha/r$$ 로 나누는 이유는 $$r$$ 을 바꿀 때 갱신의 크기가 함께 변하지 않도록 보정하기 위함입니다.
- $$r$$ 이 커지면 $$BA$$ 의 노름(norm)도 통계적으로 커지는데, $$1/r$$ 로 나누면 그 효과가 상쇄됩니다. 
  - 결과적으로 $$r$$ 을 조정해도 학습률을 다시 맞출 필요가 줄어듭니다.


### **추론 비용 영역**

- LoRA의 중요한 성질은 **학습 후 $$W_0 + \frac{\alpha}{r}BA$$ 를 하나의 행렬로 합칠 수 있다**는 점입니다. 
  - 이를 **merge** 라고 부릅니다.
- 추가 모듈이 없으므로 **추론 비용이 전량 미세조정과 동일**합니다. 
  - 어댑터처럼 층 사이에 모듈을 추가하는 접근은 추론이 느려지지만, LoRA는 이 비용이 0입니다.
- 다만 merge를 적용하면 어댑터를 바꿔 끼우는 유연성을 잃습니다. 
  - 다른 과업으로 전환하려면 LoRA 가중치를 다시 분리해야 합니다.


### **어느 모듈에 적용할지**

- Transformer 안에서 LoRA를 적용할 후보는 다음과 같습니다.
  - 주의 사영 행렬: $$W^Q$$, $$W^K$$, $$W^V$$, $$W^O$$
  - 순방향망: $$W_1$$, $$W_2$$
  - 출력 임베딩: $$W_{\text{lm}}$$
- 인코더와 디코더 중 어느 쪽에 넣는지도 선택 요소입니다. 
  - 같은 파라미터 예산에서 **여러 모듈에 낮은 $$r$$ 로 나눠 넣는 것**과 한 모듈에 높은 $$r$$ 로 넣는 것이 자주 비교 연구됩니다.


<br>


## **양자화와 QLoRA <sup>Quantization and QLoRA</sup>**

- LoRA가 학습 메모리를 줄였다면, **양자화(Quantization)** 는 추론과 저장 메모리를 더 줄입니다. 
  - 가중치의 수치 정밀도를 낮춰 메모리를 절감합니다.


### **4비트 NormalFloat (NF4)**

- **NF4(4-bit NormalFloat)** 는 가중치가 정규분포를 따른다는 가정 아래, 각 구간의 확률 질량이 같아지도록 양자화 수준을 배치합니다. 
  - 정보 이론적으로 최적인 분위 양자화(Quantile Quantization)입니다.

$$
q_i = \tfrac{1}{2}\Big(\Phi^{-1}\big(\tfrac{i}{2^b + 1}\big) + \Phi^{-1}\big(\tfrac{i+1}{2^b + 1}\big)\Big)
$$

- $$\Phi^{-1}$$ 은 표준 정규분포의 역누적분포함수입니다. 
  - $$2^b$$ 개의 양자화 수준이 정규분포 분위에 맞춰 배치되므로, 가중치 분포에 가장 가까운 양자화가 됩니다.


### **이중 양자화와 페이지 옵티마이저**

- 양자화는 블록 단위로 축척 상수(Scale)를 둬야 합니다. 
  - 이 축척 상수까지 다시 양자화하는 **이중 양자화(Double Quantization)** 로 추가 메모리 절감이 가능합니다.
- 옵티마이저 상태를 CPU 메모리로 내렸다가 필요할 때 다시 올리는 **페이지 옵티마이저(Paged Optimizer)** 가 함께 사용됩니다.


### **QLoRA: 양자화와 LoRA의 결합**

- **QLoRA** 는 본체 가중치를 4비트 NF4로 얼리고, 어댑터만 고정밀(보통 bf16)로 학습합니다 [^ref3]. 
  - 순전파에서 필요한 순간에만 역양자화(Dequantize)를 수행합니다.

$$
Y = X\,\text{dequant}\big(W_0^{\text{NF4}}\big) + X\,\frac{\alpha}{r}BA
$$

- 중요한 성질은 **기울기가 $$A$$, $$B$$ 로만 흐르므로 양자화 오차가 학습을 방해하지 않는다**는 점입니다. 
  - 본체는 함수의 일부일 뿐이고 최적화 대상이 아닙니다.
- 이 구조가 **단일 소비자 GPU에서 수십억 파라미터 모형을 미세조정**할 수 있게 만든 지점입니다. 
  - 70B 모델도 48GB VRAM 하나에서 QLoRA로 적응시킬 수 있습니다.


<br>


## **LLM을 번역에 활용하기**

- 디코더 전용 LLM을 번역에 사용하는 접근은 Encoder-Decoder 번역 모형과 전제가 다릅니다. 
  - LLM은 번역을 위해 설계되지 않았으므로, 무엇을 얼마나 학습시켜야 하는가가 중요한 설계 질문이 됩니다.


### **대량 병렬 데이터의 역설**

- 병렬 데이터를 늘리면서 LLM을 전량 미세조정하면 품질이 단조 증가하지 않습니다. 
  - 적은 양에서 정점을 찍고 이후 하락하며, 아주 많이 주면 무작위 초기화 모형과 비슷해지는 현상이 관측됩니다.
- 원인은 사전학습으로 얻은 표현이 지워진다는 점입니다. 
  - 번역 전용 데이터만 많이 주면 LLM이 가진 범용 언어 능력이 사라집니다.
- 이 관찰이 **2단계 적응 구성**의 근거가 됩니다 [^ref4].

$$
\theta_0 \;\xrightarrow{\;\text{단일어 CPT}\;}\; \theta_1 \;\xrightarrow{\;\text{소량 고품질 병렬 SFT}\;}\; \theta_2
$$


### **1. 계속 사전학습(Continued Pretraining)**

- **계속 사전학습(Continued Pretraining, CPT)** 은 비영어 언어의 단일어 텍스트로 다음 토큰 예측(Causal Language Modeling, CLM)을 계속 수행합니다. 
  - 목적은 번역을 학습하는 것이 아니라 **목표 언어의 표현을 늘리는 것**입니다.

$$
\mathcal{L}_{\text{CPT}} = -\sum_{t} \log P_\theta(x_t \mid x_{<t}), \qquad x \sim D^{\text{mono}}
$$

- 영어 텍스트를 섞는 이유는 영어 능력의 망각을 방지하기 위함입니다.
- 목적 함수의 선택이 중요합니다. 
  - 사전학습과 **같은 형태**(CLM)를 유지하는 것이 유리합니다. 
  - Prefix LM이나 Mixture-of-Denoisers처럼 형태를 바꾸면 이미 학습된 것을 재적응시켜야 하는 비용이 발생합니다.


### **2. 지도 미세조정(Supervised Fine-Tuning)**

- **지도 미세조정(Supervised Fine-Tuning, SFT)** 은 번역 지시문 형식으로 병렬 데이터를 학습합니다. 
  - 핵심 관찰은 **양보다 품질이 지배적**이라는 점입니다. 
  - 사람이 쓴 소량(방향당 수천에서 수만)이 기계로 걸러낸 대량보다 성능이 좋습니다 [^ref5].
- 손실은 **완성부(Completion)에만** 걸립니다. 지시문과 원문에는 손실을 걸지 않고 목표 문장에만 겁니다.

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{completion}} \log P_\theta\big(y_t \mid y_{<t},\, \text{prompt}\big)
$$

- 조건부 분포를 학습하는 것이 목적이므로 조건 부분의 우도는 손실에 포함할 필요가 없습니다.


### **표면 정렬 가설**

- **표면 정렬 가설(Superficial Alignment Hypothesis)** 은 번역 능력이 사전학습에 이미 존재하고, 미세조정은 **형식(Format)만 가르친다**는 가설입니다 [^ref6].
- 이 가설이 예측하는 네 가지 현상이 있습니다.
  1. 아주 적은 예제로도 기준선에 도달합니다.
  2. 한 방향만 학습해도 다른 방향이 함께 개선됩니다.
  3. 사전학습에 없던 언어는 개선되지 않습니다.
  4. 목표 문장이 잡음이어도 형식만 맞으면 작동합니다.
- 세 번째가 가설의 검증 지점입니다. 형식만 학습하는 것이라면 언어 능력 자체는 미세조정으로 생기지 않아야 합니다.


### **목표 언어 이탈과 완화**

- **목표 언어 이탈(Off-Target Translation)** 은 지시를 무시하고 다른 언어로 출력하는 현상입니다. 
  - 디코더 전용 모형에서 특히 두드러집니다. 원인이 두 가지입니다.
    - 사전학습 데이터가 영어 중심이라 비영어 생성이 약합니다.
    - 지시문이 언어를 지정하는 신호가 되지 못할 때가 있습니다.
- 완화 방법은 목표 언어를 prompt에 명시하는 것, 예시를 넣는 것, 그리고 2단계 미세조정 자체입니다.


<br>


## **사람 선호로 학습하기 <sup>Alignment from Preferences</sup>**

- 보상 신호를 자동 지표에 두는 접근은 지표의 편향을 학습이 그대로 흡수한다는 한계가 있습니다. 
  - 그 자리에 **사람의 상대적 판단**을 놓는 것이 RLHF와 DPO 계열의 선호 학습입니다.
- 절대 점수 대신 상대 비교를 사용하는 이유는 두 가지입니다.
  - 사람은 "이 번역이 몇 점인가"보다 "둘 중 어느 것이 나은가"를 더 일관되게 답합니다.
  - 상대 비교는 평가자 간 척도 차이를 보정할 필요가 없습니다.


### **Bradley-Terry 보상 모형**

- Pair 별 비교를 확률 모형으로 두는 가장 널리 쓰이는 방법이 **Bradley-Terry 모형**입니다 [^ref7]. 
  - 각 후보에 잠재 강도(Latent Strength)가 있고, 비교 결과가 그 차이의 Logistic 함수를 따른다고 가정합니다.

$$
P(e_w \succ e_l \mid f) = \frac{\exp r(f, e_w)}{\exp r(f, e_w) + \exp r(f, e_l)} = \sigma\big(r(f, e_w) - r(f, e_l)\big)
$$

- $$\sigma$$ 는 Logistic Sigmoid입니다. 
  - 중요한 성질은 **$$r$$ 이 상수만큼 평행이동해도 같은 확률을 산출**한다는 점입니다. 
  - 따라서 보상은 차이만 의미를 가집니다.
- 선호 쌍 데이터 $$\mathcal{D} = \{(f, e_w, e_l)\}$$ 로 $$r$$ 을 최대우도 추정합니다.

$$
\mathcal{L}_R(\psi) = -\mathbb{E}_{(f, e_w, e_l) \sim \mathcal{D}}\Big[\log \sigma\big(r_\psi(f, e_w) - r_\psi(f, e_l)\big)\Big]
$$


<br>


## **강화학습 경로: PPO <sup>Reinforcement Learning Path</sup>**

- 보상 모형이 있다면 그것을 최대화하도록 정책 $$\pi_\theta$$ 를 학습할 수 있습니다 [^ref9]. 
  - 그러나 보상을 그대로 최대화하면 모형이 보상 모형의 허점을 찾아 언어 능력을 버리는 현상(**보상 해킹, Reward Hacking**)이 발생합니다.


### **KL 제약 보상 최대화**

- 보상 해킹을 완화하기 위해 참조 정책에서 멀어지는 것을 벌하는 KL 제약 항을 추가합니다 [^ref8].

$$
\max_{\pi_\theta}\;\mathbb{E}_{f \sim \mathcal{D},\; e \sim \pi_\theta(\cdot \mid f)}\big[r(f, e)\big] \;-\; \beta\,\mathbb{D}_{\text{KL}}\big(\pi_\theta(\cdot \mid f) \,\|\, \pi_{\text{ref}}(\cdot \mid f)\big)
$$

- $$\pi_{\text{ref}}$$ 는 보통 지도 미세조정을 마친 모형입니다. $$\beta$$ 가 크면 참조에 가까이 머물고, 작으면 보상을 좇습니다.
- 이 제약은 파라미터 공간이 아니라 **출력 분포 공간**에서 거리를 재는 정규화입니다.


### **PPO의 clip 메커니즘**

- 정책 기울기를 그대로 적용하면 갱신이 커질 때 분포가 급격히 변해 발산할 수 있습니다. 
  - **PPO(Proximal Policy Optimization)** 는 갱신 비율을 자르는 방식으로 이 문제를 다룹니다 [^ref10].

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat{A}_t,\; \text{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\big)\Big]
$$

$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

- $$\hat{A}_t$$ 는 이점 추정(Advantage Estimate)이고, 보통 가치 함수(Value Function)를 따로 학습해서 계산합니다.
- PPO의 부담은 구현 비용입니다. 
  - 정책, 참조, 보상, 가치 네 모형을 동시에 메모리에 유지해야 하고 온라인 샘플링 루프가 필요해서 학습 코드가 복잡해집니다.


### **GRPO: 가치 모형을 없애는 변형**

- **GRPO(Group Relative Policy Optimization)** 는 가치 모형을 제거합니다 [^ref11]. 
  - 같은 입력에 대해 여러 출력을 샘플링해서 **그 집단 안에서 상대 이점을 계산**합니다.

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

- 가치 함수 학습이 사라지므로 메모리와 구현 복잡도가 줄어듭니다. 
  - 대신 입력마다 $$G$$ 개의 출력을 샘플링해야 하는 비용이 추가됩니다.
- DeepSeekMath에서 GRPO가 PPO의 실용적 대안으로 제시된 이후 널리 사용되고 있습니다.


<br>


## **DPO: 참조 모델 없이 선호 학습하기 <sup>Direct Preference Optimization</sup>**

- DPO는 Bradley-Terry 선호 모형과 KL 제약 보상 최대화 문제의 닫힌 해(Closed-form Solution)를 결합해서, **보상 모델과 강화학습 루프를 모두 제거**하는 접근입니다 [^ref12]. 


### **1. 닫힌 해의 유도**

- KL 제약 보상 최대화 문제에는 닫힌 형태의 최적해가 존재합니다.

$$
\pi^{*}(e \mid f) = \frac{1}{Z(f)}\,\pi_{\text{ref}}(e \mid f)\,\exp\Big(\frac{1}{\beta}r(f, e)\Big)
$$

- $$Z(f) = \sum_{e'} \pi_{\text{ref}}(e' \mid f)\exp(r(f, e')/\beta)$$ 는 정규화를 위한 분배 함수입니다.
- 이 식은 라그랑지안 유도로 얻어집니다. KL 제약을 라그랑지안 승수와 함께 두고 변분 최적화로 풀면 지수 가중된 참조 분포라는 형태가 나옵니다.


### **2. 보상을 정책의 함수로 변환**

- 위 식을 $$r$$ 에 대해 풀면 보상이 정책의 함수로 표현됩니다.

$$
r(f, e) = \beta \log \frac{\pi^{*}(e \mid f)}{\pi_{\text{ref}}(e \mid f)} + \beta \log Z(f)
$$

- 즉, 최적 정책의 **참조 대비 로그 비율**이 보상의 선형 변환이라는 관계가 성립합니다. 


### **3. Bradley-Terry 대입과 $$Z(f)$$ 의 소거**

- 위 보상 식을 Bradley-Terry 우도에 대입합니다. 
  - Bradley-Terry 우도는 두 보상의 차이에만 의존합니다.

$$
P(e_w \succ e_l \mid f) = \sigma\big(r(f, e_w) - r(f, e_l)\big)
$$

- 보상 식을 대입하면 다음과 같은 식이 됩니다.

$$
r(f, e_w) - r(f, e_l) = \beta\log\frac{\pi^{*}(e_w \mid f)}{\pi_{\text{ref}}(e_w \mid f)} - \beta\log\frac{\pi^{*}(e_l \mid f)}{\pi_{\text{ref}}(e_l \mid f)} + \beta\log Z(f) - \beta\log Z(f)
$$

- **$$\beta \log Z(f)$$ 가 두 항에 같은 값으로 들어가 소거**됩니다. 
  - 이 소거가 DPO의 수학적 핵심입니다. 
  - 쉽게 계산할 수 없었던 분배 함수가 선호 쌍 설정에서 사라지므로, 보상 모형 없이도 선호 데이터를 학습 신호로 사용할 수 있습니다.


### **4. DPO 손실 함수**

- 소거 결과로 얻어지는 DPO 손실은 다음과 같습니다.

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(e_w \mid f)}{\pi_{\text{ref}}(e_w \mid f)} - \beta \log \frac{\pi_\theta(e_l \mid f)}{\pi_{\text{ref}}(e_l \mid f)}\Big)\Big]$$

- 지도 학습 형태의 손실 하나로 전체 학습이 끝납니다. 
- 필요한 것은 선호 쌍과 참조 정책뿐이며, 보상 모형과 PPO의 샘플링 루프가 모두 사라집니다.


### **5. 기울기의 의미**

- DPO 손실의 기울기를 보면 이 손실이 수행하는 작업이 드러납니다.

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\sigma\big(\hat{r}_l - \hat{r}_w\big)\Big(\nabla_\theta \log \pi_\theta(e_w) - \nabla_\theta \log \pi_\theta(e_l)\Big)\Big]
$$

- 여기서 $$\hat{r} = \beta\log(\pi_\theta/\pi_{\text{ref}})$$ 는 **암묵적 보상(Implicit Reward)** 입니다.
- 가중치 $$\sigma(\hat{r}_l - \hat{r}_w)$$ 는 **모형이 순서를 아직 맞히지 못하는 쌍에서 크고, 이미 맞히는 쌍에서 작습니다**. 
  - 즉 DPO는 **틀리는 쌍에 집중**하는 구조를 자연스럽게 형성합니다.


### **CPO와 WPO 변형**

- DPO는 참조 정책 $$\pi_{\text{ref}}$$ 를 메모리에 함께 유지해야 합니다. 
  - 이 비용을 제거하거나 완화하는 변형이 있습니다.

- **CPO(Contrastive Preference Optimization)** 는 참조 정책을 균등 분포로 두어 $$\pi_{\text{ref}}$$ 항을 소거합니다 [^ref13]. 
  - 다만 참조가 사라지면 선호되는 쪽의 절대 우도를 유지할 장치가 없어지므로, 음의 로그우도(NLL) 항을 더합니다.

$$
\mathcal{L}_{\text{CPO}} = \mathcal{L}_{\text{prefer}} + \alpha\,\big[-\log \pi_\theta(e_w \mid f)\big]
$$

- 두 항의 역할이 다릅니다. 
  - 앞 항은 **$$e_w$$ 를 $$e_l$$ 보다 높게** 하라는 요구이고, 뒤 항은 **$$e_w$$ 자체를 높게** 하라는 요구입니다. 
  - 앞 항만 쓰면 두 쪽을 함께 낮추면서 차이만 벌리는 해가 허용됩니다.

- **WPO(Weighted Preference Optimization)** 는 정책 외(Off-Policy) 쌍을 사용할 때의 분포 불일치를 가중치로 보정합니다 [^ref14]. 
  - 현재 정책이 각 쪽을 생성할 확률로 쌍에 무게를 둡니다.

$$
\mathcal{L}_{\text{WPO}} = -\mathbb{E}\Big[w(f, e_w)\,w(f, e_l)\,\log\sigma\big(\hat{r}_w - \hat{r}_l\big)\Big]
$$

- 가중치는 기울기를 흘리지 않습니다(detached). 
  - 가중치 자체를 최적화하면 모형이 가중치를 낮춰 손실을 줄이는 해를 찾기 때문입니다.


### **좋은 것만 모방하는 접근이 부족한 이유**

- 선호 데이터를 쓰는 가장 단순한 접근은 선호되는 쪽만 모아 지도 미세조정하는 것(SFT on preferred)입니다. 

$$
\mathcal{L}_{\text{SFT-pref}} = -\log \pi_\theta(e_w \mid f)
$$

- 이 손실은 $$e_l$$ 의 확률을 낮추지 않습니다. 
  - 모델은 **무엇이 좋은가**만 학습하고 **무엇이 나쁜가**는 학습하지 않습니다. 
  - 거의 맞는 번역을 거부하는 능력이 이 차이에서 나옵니다.
- 참조 문장을 모방하는 지도 학습이 참조의 품질을 성능 상한으로 만드는 것과 같은 구조적 한계입니다.


<br>


## **뉴럴 평가지표 <sup>Learned Evaluation Metrics</sup>**

- (1)편에서 BLEU, chrF, TER 같은 문자열 겹침 지표가 **바꿔 쓴 표현을 벌한다**는 한계를 다뤘습니다. 
  - 이 한계를 넘기 위해 **뉴럴 평가지표(Learned Metrics)** 가 도입되었습니다.
- 뉴럴 평가지표는 사전학습된 언어 모델 위에 사람 점수로 미세조정한 회귀 모델입니다. 
- 대표적으로 **BLEURT**(참조 기반) [^ref16]와 **COMET**(참조 기반 또는 참조 없음) [^ref15]이 있습니다.


### **뉴럴 평가지표의 세 가지 실패 유형**

- 뉴럴 평가지표는 신경망이기 때문에 학습 분포 밖의 입력에서 예측 불가능하게 동작합니다. 
  - 실패 유형을 세 갈래로 나눌 수 있습니다.

**Technical**

| 요인 | 내용 |
|---|---|
| 소프트웨어 버전 | 같은 체크포인트라도 구현이 바뀌면 점수가 달라짐 |
| 수치 정밀도 | FP32, FP16, INT8 에서 값이 다르고, 낮은 정밀도에서 사람 상관이 무너질 수 있음 |
| 체크포인트 미표기 | 논문이 `COMET`이라고만 적으면 재현이 불가능함 |

- BLEU에서 토큰화 설정이 같은 문제를 일으켰고 표준 구현(sacreBLEU)으로 해결되었습니다. 
  - 뉴럴 평가지표에서는 **환경 서명(Environment Signature)** 을 함께 보고하는 관행이 그 역할을 합니다.

**Data**

| 실패 | 내용 |
|---|---|
| 빈 출력 | 아무것도 출력하지 않아도 0이 아닌 점수를 받음 |
| 언어 불일치 | 목표 언어가 아닌 출력이 무작위 정답 언어 출력보다 높은 점수를 받을 수 있음 |
| 원문 복사 | 원문을 그대로 낸 것이 실제 번역보다 높은 점수를 받을 수 있음 |
| 점수 분포 편향 | 지표가 특정 구간에 점수를 몰아줌 |
| 도메인 편향 | 학습 데이터에 흔한 도메인에 유리함 |
| Translationese | 참조가 번역문인지 원문인지에 따라 값이 달라짐 |

- **번역이라 할 수 없는 출력을 걸러내지 못한다는 뜻**이므로, 지표를 학습 신호로 사용하는 선호 학습에서 그대로 위험이 됩니다.
  - 참조 없는 지표는 원문과 출력만 보므로 원문 복사에 특히 취약합니다.

- **Usage**: 지표를 사용하는 방식의 문제입니다. 
  - 다중 참조를 지원하지 않는 지표에 참조를 여럿 주는 방식, 문장 수준 값을 평균해서 체계 수준 값으로 쓰는 방식 등이 여기에 속합니다.


### **언어 범위의 두 층**

- 뉴럴 평가지표는 두 층의 언어 범위를 가집니다.

| 층 | 범위 |
|---|---|
| 인코더 사전학습 | 넓음 (백 개 규모의 언어) |
| 지표 미세조정 (사람 점수) | 좁음 (수십 개 언어쌍) |

- 뒤의 범위가 실질적 제약입니다. 
  - 인코더가 특정 언어를 지원해도, 사람 점수로 미세조정한 언어쌍에 포함되지 않으면 지표가 품질을 재지 못합니다. 
  - 이 경우 사람 상관이 0 근처로 떨어집니다.


### **지표를 학습 신호로 사용할 때의 주의**

- 지표를 학습 또는 선택의 신호로 쓰면 **지표가 보지 않는 축이 희생됩니다**. 
  - 선호 쌍 생성, 능동 학습, 순차 수준 학습이 모두 이 위험을 가집니다.
- 원인은 굿하트 법칙(Goodhart's Law)의 구조입니다. 
  - 대리 지표를 목표로 삼으면 그 지표와 진짜 목표의 상관이 최적화 과정에서 깨집니다.
- 대응 방법은 세 가지입니다.
  1. **지표와 평가를 분리**합니다. 학습에 사용한 지표로 결과를 평가하지 않습니다.
  2. **지표가 보지 않는 축을 따로 측정**합니다. 형식 보존, 언어 정합성, 길이 분포 같은 규칙 기반 검사가 여기에 해당합니다.
  3. **여러 지표의 합의**를 요구합니다. 평균하면 한 지표의 오류가 그대로 들어오므로, 방향이 일치할 때만 신호로 사용합니다.


<br>


## **메타평가 <sup>Meta-Evaluation</sup>**

- 지표를 사람 판정과 견주는 층이 **메타평가**입니다. 
  - 지표의 신뢰성을 수치로 검증하는 과정입니다.


### **체계 수준과 문장 수준**

- 메타평가는 두 층위를 구분합니다.

| 층위 | 보는 대상 | 통계량 |
|---|---|---|
| 체계 수준(System-level) | 체계들의 순위가 사람과 같은가 | 쌍별 정확도(Pairwise Accuracy) |
| 문장 수준(Segment-level) | 문장별 점수가 사람과 상관하는가 | 켄달 타우 $$\tau$$, 피어슨 $$r$$ |

- 체계 수준에서 잘 맞는 지표가 문장 수준에서 약할 수 있습니다. 
  - 체계 수준은 많은 문장의 평균이라 개별 오차가 상쇄되기 때문입니다. 용도가 다르므로 둘을 함께 봐야 합니다.
- 체계 수준 쌍별 정확도는 다음과 같이 정의됩니다.

$$
\text{Acc} = \frac{\big|\{(i,j) : \text{sign}(m_i - m_j) = \text{sign}(h_i - h_j)\}\big|}{\binom{N}{2}}
$$

- $$m_i, m_j$$ 는 두 체계의 지표 점수, $$h_i, h_j$$ 는 두 체계의 사람 점수입니다. 
  - 지표와 사람이 순위를 같은 방향으로 매긴 쌍의 비율을 계산합니다.


### **동점 보정**

- 문장 수준 비교에서 **동점(Tie)을 어떻게 처리하는가가 결과를 바꿉니다**. 
  - 사람 점수에는 동점이 많고 지표 점수에는 거의 없는 비대칭이 있습니다. 지표가 억지로 순서를 매기면 불리하게 평가되고, 동점을 무시하면 유리해집니다.
- 동점 보정은 지표에 동점 구간 $$\epsilon$$ 을 허용하고, 그 값을 정확도가 최대가 되도록 지표마다 따로 추정합니다 [^ref18]. 
  - 결과로 **이 지표는 얼마나 차이가 나야 구분이 가능하다고 볼 수 있는가**가 함께 얻어집니다.


### **사람 평가의 두 규격**

- 사람 평가에는 두 규격이 널리 쓰입니다.

| 규격 | 방식 | 성질 |
|---|---|---|
| 직접 평가(Direct Assessment, DA) | 연속 척도로 점수 부여 | 평가자 간 척도 차이를 표준화해야 함 |
| MQM(Multidimensional Quality Metrics) [^ref17] | 오류 구간에 유형과 심각도 표시, 가중 합산 | 재현성이 높고 오류 유형별 분석 가능 |

- MQM 점수는 다음과 같이 정의됩니다.

$$
\text{MQM score} = -\sum_{i \in \text{errors}} w_{\text{severity}(i)}
$$

- MQM은 전문가 주석이 필요해서 비용이 큽니다. 
  - 그래서 **언어 모형으로 MQM을 모사**하는 방법이 제안됩니다 [^ref19]. 
  - 오류를 항목별로 나열하게 하고 그 개수를 세어 점수화합니다.

$$
\hat{s} = -\big(w_{\text{major}}\,n_{\text{major}} + n_{\text{minor}}\big)
$$

- 점수를 직접 묻는 방식보다 문장 수준 상관이 개선되고, 설명도 함께 얻을 수 있습니다. 
  - 오류 탐지와 집계를 두 단계로 분리하는 것, 서술형이 아니라 항목식으로 나열하게 하는 것이 설계 요소입니다.


<br>


## **Conclusion**

- **LLM을 번역에 활용하는 방법**은 LoRA와 QLoRA로 메모리 비용을 줄이고, CPT와 SFT의 2단계 미세조정으로 사전학습 능력을 보존하는 방식입니다. 
  - LoRA는 저차 가정과 스케일링 계수 $$\alpha/r$$ 로 학습을 안정화하고, QLoRA는 NF4 양자화와 이중 양자화로 수십억 파라미터 모델을 단일 GPU에 올립니다.
- **사람 선호로 학습하는 방법**은 Bradley-Terry 보상 모형을 바탕으로 PPO 또는 DPO를 적용합니다. 
  - DPO의 유도는 KL 제약 보상 최대화의 닫힌 해에서 분배 함수 $$Z(f)$$ 가 선호 쌍 설정에서 소거된다는 수학적 성질에 기반합니다. 
  - 보상 모델과 강화학습 루프 없이 지도 학습 형태의 단일 손실로 전체 과정을 끝낼 수 있습니다.
- **뉴럴 평가지표**는 BLEU 같은 문자열 겹침 지표의 한계를 넘기 위해 도입되었지만, 세 층의 실패 유형(기술적, 데이터, 사용)을 가집니다. 
  - 메타평가에서는 체계 수준과 문장 수준을 구분하고 동점 보정이 결과를 바꿉니다. 
  - MQM은 사람 평가의 재현성을 높이는 규격이며, LLM으로 MQM을 모사하는 접근이 LLM 기반 자동 평가의 설계 지점입니다.
- (1)편의 Classical 번역 기법과 이번 (2)편의 LLM이 공유하는 구조가 있습니다. 
  - SMT의 로그선형 결합이 로그 선형 가중 평균이고, DPO의 암묵적 보상이 로그 비율이며, BLEU의 n-gram 정밀도가 비율 통계라는 공통점은 번역이라는 과업이 **"무엇이 좋은 번역인가"를 비율과 로그 공간에서 다룬다**는 특성에서 비롯됩니다. 
  - LLM이 번역을 포함한 범용 능력을 갖추게 된 지금도 이 구조는 그대로 사용되고 있습니다.


<br>


-----------------------

## References

[^ref1]: Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR* (2022).
[^ref2]: Aghajanyan, Armen, Luke Zettlemoyer, and Sonal Gupta. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." *ACL* (2021).
[^ref3]: Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS* (2023).
[^ref4]: Xu, Haoran, et al. "A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models." *ICLR* (2024). (ALMA)
[^ref5]: Alves, Duarte, et al. "Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning." *EMNLP Findings* (2023).
[^ref6]: Zhou, Chunting, et al. "LIMA: Less Is More for Alignment." *NeurIPS* (2023).
[^ref7]: Bradley, Ralph Allan, and Milton E. Terry. "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." *Biometrika* 39.3/4 (1952): 324-345.
[^ref8]: Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences." *NeurIPS* (2017).
[^ref9]: Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS* (2022). (InstructGPT)
[^ref10]: Schulman, John, et al. "Proximal Policy Optimization Algorithms." *arXiv:1707.06347* (2017).
[^ref11]: Shao, Zhihong, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv:2402.03300* (2024). (GRPO)
[^ref12]: Rafailov, Rafael, et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS* (2023).
[^ref13]: Xu, Haoran, et al. "Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation." *ICML* (2024).
[^ref14]: Zhou, Wenxuan, et al. "WPO: Enhancing RLHF with Weighted Preference Optimization." *EMNLP* (2024).
[^ref15]: Rei, Ricardo, et al. "COMET: A Neural Framework for MT Evaluation." *EMNLP* (2020).
[^ref16]: Sellam, Thibault, Dipanjan Das, and Ankur P. Parikh. "BLEURT: Learning Robust Metrics for Text Generation." *ACL* (2020).
[^ref17]: Freitag, Markus, et al. "Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation." *TACL* (2021). (MQM)
[^ref18]: Deutsch, Daniel, Rotem Dror, and Markus Freitag. "Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and Tie Calibration." *EMNLP* (2023).
[^ref19]: Kocmi, Tom, and Christian Federmann. "Large Language Models Are State-of-the-Art Evaluators of Translation Quality." *EAMT* (2023). (GEMBA)
