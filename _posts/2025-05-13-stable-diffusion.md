---
title: "Stable Diffusion 원리로 이해하기 (1)"
date: 2025-05-13 00:00:00 +/-TTTT
categories: [인공지능 | AI, AI 이론]
tags: [deep-learning, generative-ai, llm, neural-network, auto-encoder, vae, stable-diffusion]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
pin: false
description: 🎨 생성 모델의 기본 개념과 다양한 학습 방법을 공부하고, 데이터 생성 원리를 알아봅시다.
---


--------------------
> **<u>KEYWORDS</u>**         
> Stable Diffusion, Stable Diffusion 원리, Stable Diffusion AI, Stable Diffusion Model, Stable Diffusion 수학
{: .prompt-info }
--------------------

&nbsp;
&nbsp;
&nbsp;



## **생성 모델 <sup>Generative Models</sup>**

### **Introduction**

- 생성모델이란 목표 도메인의 데이터를 생성하는 모델을 뜻합니다.
  - 즉, 실제 데이터와 유사한 새 데이터를 만들어내는 것이 목적입니다.
- N개의 학습 데이터 $$D = \{ \mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)} \}$$가 미지의 확률분포 $$p(\mathbf{x})$$로 부터 서로 독립적으로 추출된 것이라고 가정할 때, 생성 모델은 $$q_{\theta}(\mathbf{x})$$라는 확률분포를 정의하고, 이에 따라 데이터를 샘플링합니다.
  - $$\mathbf{x} \sim q_{\theta}(\mathbf{x})$$
  - $$\theta$$는 신경망의 매개변수와 같은, 모델의 확률분포를 결정짓는 요소를 의미합니다.
  - **목표** ㅣ 실제 데이터의 확률 분포 $$p(\mathbf{x})$$와 최대한 유사한 분포 $$q_{\theta}(\mathbf{x})$$를 갖는 생성 모델을 학습하는 것입니다.
    - 이때 두 분포의 유사도를 평가하기 위해 **KL-Divergence**와 **Wasserstein Distance**와 같은 지표를 사용합니다.

- ex. 텍스트와 같은 조건 정보 $$\mathbf{c}$$가 주어졌을 때, 이에 대응하는 이미지 $$\mathbf{x}$$를 생성하는 문제는 Joint Probability $$p(\mathbf{x}, \mathbf{c})$$ 또는 Conditional Probability $$p(\mathbf{x} \vert \mathbf{c})$$를 모델링하는 생성 모델로 설명할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


- 데이터 $$\mathbf{x} \in X$$에 대한 생성 모델이 정의하는 확률 분포 $$q_{\theta}(\mathbf{x})$$는 다음과 같습니다.

  $$
  q_\theta(\mathbf{x}) = \gamma_{\theta}(\mathbf{x})/Z(\theta),
  $$
  
  $$
  Z(\theta) = \int_{\mathbf{x}' \in X} \gamma_{\theta}(\mathbf{x}') d \mathbf{x}'.
  $$

  - where $$\gamma_{\theta}(\mathbf{x}) \geq 0$$ is an unnormalized probability density function(PDF) and $$Z(\theta) > 0$$ is a partition function(normalization constant).
    - Partition Function $$Z(\theta)$$는 전체 데이터 공간을 적분함으로써 계산되며, 모델의 확률 분포가 정규화되도록 합니다.
    - i.e., 전체 데이터 공간에 대한 모든 정보를 포함하므로, 데이터 전체의 다양한 통계량을 계산할 수 있으며, $$\int q_\theta(\mathbf{x}) d \mathbf{x}= 1$$이 되도록 보장합니다.


&nbsp;
&nbsp;
&nbsp;


- **<u>Energy-based model(EBM)</u>**

  - Unnormalized PDF $$\gamma_{\theta}(\mathbf{x}) = \exp(-f_{\theta}(\mathbf{x}))$$

  - i.e.,
    $$
    q_\theta(\mathbf{x}) = \exp(-f_{\theta}(\mathbf{x}))/Z(\theta),
    $$
    
    $$
    Z(\theta) = \int_{\mathbf{x}' \in X} \exp(-f_{\theta}(\mathbf{x}')) d\mathbf{x}'.
    $$

  - 에너지 함수 $$f_{\theta}(\mathbf{x})$$는 데이터 포인트 $$\mathbf{x}$$의 에너지를 정의하며, 이 값이 작을수록 해당 데이터가 더 자주 등장할 가능성이 높다고 간주합니다.
  - 이러한 Energy-based 모델의 장점은 확률분포로서 엄격한 제약 없이 자유롭게 모델링 할 수 있다는 점입니다. 
    - 그러나 Normalization Constant $$Z(\theta)$$의 계산이 필수적이며, 이는 고차원 공간에서 계산 비용이 큽니다.

&nbsp;
&nbsp;
&nbsp;


### **Training**

#### 0. KL Divergence

- **확률 분포 $$p(\mathbf{x})$$로 부터 $$q(\mathbf{x})$$로의 KL Divergence는 다음과 같이 정의됩니다.**
  $$
  D_{KL}(p \| q) := \int_x p(x) \log \frac{p(x)}{q(x)} dx.
  $$

  - If $$p(x) = q(x),$$ then $$D_{KL}(p \| q) = 0$$.
  - else,  $$D_{KL}(p \| q) > 0$$.

- KL Divergence는 두 분포가 서로 얼마나 다른지를 측정하며, 차이가 클수록 그 값은 큰 양의 값을 가집니다.


&nbsp;
&nbsp;
&nbsp;


#### 1. Likelihood-based Model

- **<u>Likelihood-based Model</u>**
  - Likelihood-based 모델은 데이터 $$\mathbf{x}$$의 Likelihood $$q_\theta(\mathbf{x})$$를 명시적(Explicit)으로 정의하고, 그 확률 분포의 Log Likelihood를 최대화하여 파라미터 $$\theta$$를 추정합니다.

  - N개의 학습 데이터 $$D = \{ \mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)} \}$$는 서로 독립적으로 추출되었기에, $$D$$의 Likelihood는 다음과 같이 각 데이터 Likelihood의 곱으로 정의됩니다.
    
    $$
    q_\theta(D) = \Pi_i q_\theta(\mathbf{x}^{(i)}).
    $$

  - Log Likelihood는 다음과 같습니다.
    
    $$
    L(\theta) = \frac{1}{N} \log q_\theta(D)
    $$
    
    $$
    = \frac{1}{N} \sum_i \log q_\theta(\mathbf{x}^{(i)}).
    $$

  - 대표적인 Likelihood-based 생성모델로는 VAE, Auto-regressive Model, Energy-based 모델이 있습니다.


&nbsp;
&nbsp;
&nbsp;

- **<u>Maximum Likelihood Estimation(MLE)</u>**

  - MLE에서는 아래와 같이 Log Likelhood를 최대화하는 파라미터 $$\theta^{*}_{ML}$$를 구합니다.
    $$
    \theta^{*}_{ML} := arg\max_\theta L(\theta)
    $$

  - **ex. MLE of Energy-based Model**

    - 첫번째 Term은 학습 데이터 $$\mathbf{x}^{(i)}$$의 에너지를 낮추는 항이며, 두번째 Term은 전체 데이터 공간에서의 에너지 합을 높이는 방향으로 작용합니다.
      - 하지만 고차원 데이터 $$\mathbf{x}$$의 경우, 학습 데이터 주변에서만 에너지를 낮추면, 실제로 존재하지 않는 다른 위치에서도 에너지가 낮아지는 문제가 발생합니다.
      - 이로 인해, 전체 에너지를 높이는 작업이 어려워집니다.
      - i.e., 학습 데이터 외의 위치에서도 잘못된 데이터가 생성될 수 있으며, 이는 모델의 품질을 저하시킬 수 있습니다.
    
    $$
    L(\theta) = \frac{1}{N} \sum^N _{i=1} \log q_\theta (\mathbf{x}^{(i)})
    $$
  
    $$
    = -\frac{1}{N} \sum^N _{i=1} \big[ f_\theta(\mathbf{x}^{(i)}) \big] - \log Z(\theta)
    $$
    
    $$
    = -\frac{1}{N} \sum^N _{i=1} \big[ f_\theta(\mathbf{x}^{(i)}) \big] - \log \int_{\mathbf{x}' \in X} \exp(-f_\theta(\mathbf{x}')) d \mathbf{x}'.
    $$

    - 학습 중에는 학습 데이터 혹은 검증 데이터의 Likelihood를 활용하여 모델의 성능을 평가할 수 있습니다.

    - **Log-likelihood의 기울기**는 아래와 같이 표현되며, 에너지 함수의 기울기를 기반으로 구성됩니다.
      - 하지만, 아래 기댓값은 모델 분포 $$q_\theta(\mathbf{x})$$에 대한 샘플링이 필요하며, 이는 매우 어렵고 계산비용이 큽니다.
      - 이를 근사하기 위해 Markov Chain Monte Carlo(MCMC) 기반의 샘플링 기법이 사용되기도 하지만, 수렴 속도가 느리고 분산이 큰 단점이 있습니다.
      
      $$
      \frac{\partial L(\theta)}{\partial \theta} = -\frac{1}{N} \sum^N _{i=1} \big[ \frac{\partial f_\theta(\mathbf{x}^{(i)})}{\partial \theta} \big] - \frac{\partial}{\partial \theta} \log Z(\theta) \\
      $$

      $$
      = -\frac{1}{N} \sum^N _{i=1} \big[ \frac{\partial f_\theta(\mathbf{x}^{(i)})}{\partial \theta} \big] + \mathbb{E}_{\mathbf{x} \sim q_\theta(\mathbf{x})} \big[ \frac{\partial f_\theta (\mathbf{x})}{\partial \theta} \big].
      $$


&nbsp;
&nbsp;
&nbsp;


- **<u>KL Divergence와 MLE</u>**
  - **MLE는 $$D_{KL}(p \| q)$$를 최소화하는 문제**로 해석될 수 있습니다.
  - 이때 $$p(\mathbf{x})>0$$인 위치에서 분모인 $$q(\mathbf{x})$$가 작을 경우, KL Divergence가 큰 Penalty를 유발하므로 모델은 모든 Mode를 포괄하도록 학습됩니다.
    - 반대로 잘못된 샘플에 대한 Penalty는 상대적으로 작기 때문에, 틀린 데이터를 명확히 배제하는 데는 한계가 있습니다.


&nbsp;
&nbsp;
&nbsp;


#### 2. Implicit Generative Model

-  명시적인(Explicit) 확률 분포 $$q_\theta(\mathbf{x})$$를 정의하지 않고, 데이터를 직접 생성하는 생성 함수만을 통해 데이터를 모델링 하는 방식입니다.
  - i.e., 샘플을 생성하는 과정은 존재하지만 그에 대한 명시적인 Likelihood 함수는 정의되지 않으며, 모델이 나타내는 분포는 암묵적(Implicit)으로 표현됩니다.
   - ex. 정규 분포로부터 추출된 잠재변수 $$\mathbf{z}$$를 신경망과 같은 결정론적 함수 $$g_\theta(\mathbf{z})$$에 통과시켜 $$\mathbf{x}$$를 생성하는 방식입니다.
     - 이와 같은 분포를 Push-forward Distribution이라 하며, GAN이 대표적인 사례입니다.
- 학습이 불안정해지기 쉬우며, 학습 중 모델의 성능을 수치적으로 평가하기 어렵다는 단점이 있습니다.
  - 그러나 Partition Function을 Explicit하게 계산할 필요가 없어, 고차원 데이터에 대해 더욱 유연하고 표현력이 높은 모델을 설계할 수 있다는 장점도 있습니다.

&nbsp;
&nbsp;
&nbsp;

- **<u>KL Divergence와 Implicit Generative Model</u>**

  - Implicit 모델은 일반적으로 **Inverse KL Divergence $$D_{KL}(q \| p)$$를 최소화하는 문제**로 이해됩니다.
  - 이는 종종 **Jensen-Shannon Divergence**를 최소화하는 문제로 대체되며, GAN은 이를 직접적으로 최적화합니다.

  $$
  D_{JS}(p \| q) = \frac{1}{2} D_{KL}(p \| \frac{1}{2}(p+q)) + \frac{1}{2} D_{KL}(q \| \frac{1}{2}(p+q)).
  $$

  - Inverse KL의 경우 $$q(\mathbf{x})>0$$인 위치에서, $$p(\mathbf{x})$$가 작으면 큰 Penalty가 발생하므로, 모델은 특정한 Mode에 집중하여 학습하게 됩니다.
  - 이로 인해 일부 Mode만 복원하고 다른 Mode를 놓치는 **Mode Collapse** 현상이 발생하기 쉽습니다.


&nbsp;
&nbsp;
&nbsp;


### **Score**

- **Score**

  $$
  s(\mathbf{x}) := \nabla_{\mathbf{x}} \log p(\mathbf{x}) : \mathbb{R}^d \rightarrow \mathbb{R}^d
  $$
  
  $$
  \nabla_{\mathbf{x}} \log p(\mathbf{x}) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})}.
  $$


  - $$\log p(\mathbf{x})$$에서의 $$\mathbf{x}$$에 대한 기울기를 뜻합니다.
  - $$\mathbf{x}$$와 동일한 차원을 가지는 벡터입니다.
  - 확률 분포는 임의의 입력에 대해 미분 가능하다고 가정되며,
    - Score는 입력 공간에서의 벡터장을 형성하고, 각 위치에서 해당 점의 Log Likelihood가 가장 급격하게 증가하는 방향과 그 크기를 나타냅니다.

- Since $$\nabla_\mathbf{x} \log Z(\theta) = 0$$,
  
  $$
  \nabla_{\mathbf{x}} \log q_\theta(\mathbf{x}) = - \nabla_\mathbf{x} f_\theta(\mathbf{x}) - \nabla_\mathbf{x} \log Z(\theta)
  $$
  
  $$
  = - \nabla_\mathbf{x} f_\theta(\mathbf{x})
  $$

  - i.e., Score는 에너지 함수의 입력에 대한 음의 기울기와 같습니다.
  - Score를 사용하면 현재 위치에서 어느 방향으로 이동해야 확률이 높은 영역에 도달할 수 있는지 파악할 수 있으므로, 고차원 공간에서 확률이 높은 영역을 효율적으로 탐색할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


#### Langevin Monte Carlo

- Score를 활용한 MCMC 기법으로, 반복 수행 시 $$p(\mathbf{x})$$로 부터 최종 표본을 얻을 수 있습니다.

  - **확률 분포의 Score가 계산되면, Langevin Monte Carlo 방법을 통해 해당 확률 분포로부터의 샘플링이 가능합니다.**

- 임의의 Prior Distribution $$\pi(\mathbf{x})$$로 부터 데이터 $$\mathbf{x}_0 \sim \pi(\mathbf{x})$$를 추출하여 각 위치에서의 Score에 따라 전이합니다.

  - 이때, 정규분포로부터 추출된 노이즈 $$\sqrt{2 \alpha} \mathbf{u}_{i+1}$$를 추가하며, 이 과정을 $$K$$번 반복합니다.

  $$
  \mathbf{x}_{i+1} = \mathbf{x}_i + \alpha \nabla_\mathbf{x} \log p(\mathbf{x}_i) + \sqrt{2 \alpha} \mathbf{u}_{i+1}.
  $$

  - 만약 $$\alpha \rightarrow 0 \,\, \text{and} \,\, K \rightarrow \inf$$ 이면, $$\mathbf{x}$$는 $$p(\mathbf{x})$$로 부터의 샘플에 수렴합니다.



- Score에 따라 Likelihood가 높은 방향으로 데이터가 전이되며, 노이즈를 추가하면 Local Extremum에 갇히는 현상을 방지하여 분포 전반을 탐색할 수 있게 해줍니다.
  - 고차원 공간에서 확률이 높은 영역을 효율적으로 탐색할 수 있습니다.

&nbsp;
&nbsp;
&nbsp;


- **Score-based Model(SBM)**
  - 확률 분포 자체를 학습하지 않고, 해당 분포의 Score를 학습하여 생성 모델을 구현하는 방식입니다.
- 확률 분포의 경우 일반 함수와 다르게 그 총합이 1이라는 제약이 있어, 모든 입력에서의 Score만 일치하면 동일한 확률 분포로 간주할 수 있습니다.
- **Score를 어떻게 학습할 것인가**에 대한 방법론들은 아래와 같이 세 가지로 구분됩니다.

&nbsp;
&nbsp;
&nbsp;

#### Score Matching 1 ㅣ Explicit Score Matching(ESM)
- 모델 $$s_\theta(\mathbf{x}):\mathbb{R}^d \rightarrow \mathbb{R}^d$$에 대하여, 실제 Score와 모델 출력 간의 제곱 오차를 최소화되는 파라미터를 학습합니다.

  - 목표 분포인 $$p(\mathbf{x})$$에 대한 Expectation을 계산한다면 다음과 같습니다.

  $$
  J_{ESM_p}(\theta) = \frac{1}{2} \mathbb{E}_{p(\mathbf{x})} \big[ \| \nabla_\mathbf{x} \log p(\mathbf{x}) - s_\theta(\mathbf{x}) \| ^2 \big].
  $$

- 그러나 대부분의 경우 $$\nabla_\mathbf{x} \log p(\mathbf{x})$$를 알 수 없기 떄문에, 위 목적 함수는 직접 사용할 수 없습니다.


&nbsp;
&nbsp;
&nbsp;


#### Score Matching 2 ㅣ Implicit Score Matching(ISM)
- Score $$\nabla_\mathbf{x} \log p(\mathbf{x})$$를 직접 사용하지 않고도 학습을 가능하게 하는 대체 목적 함수입니다.
    
  $$
  J_{ISM_p} (\theta) = \mathbb{E}_{p(\mathbf{x})} \big[ \frac{1}{2} \| s_\theta(\mathbf{x}) \|^2 + tr(\nabla_\mathbf{x} s_\theta(\mathbf{x})) \big], \\
  $$    

  $$
  tr(\nabla_\mathbf{x} s_\theta(\mathbf{x})) = \sum^d_{i=1} \frac{\partial s_\theta (\mathbf{x})_i}{\partial x_i}
  $$
    
  $$
  = - \sum^d _{i=1} \frac{\partial^2 f_\theta(\mathbf{x})}{\partial x^2 _i},
  $$

  - where $$s_\theta(\mathbf{x})$$ is estimated score by model.
  - 이는 Explicit Score Matching과 동일한 결과를 도출하는 것으로 알려져 있습니다.

- $$p(\mathbf{x})$$가 명시적으로 주어지지 않으므로, 이를 학습 데이터 집합 $$D$$를 이용한 평균으로 근사하여 아래와 같이 재정의합니다.
      
  $$
  J_{ISM_{discrete}}(\theta) = \frac{1}{N} \sum^N _{i=1} \big[ \frac{1}{2} \| s_\theta(\mathbf{x}^{(i)}) \|^2 + tr(\nabla_\mathbf{x} s_\theta (\mathbf{x}^{(i)})) \big].
  $$

  - 첫번째 Term은 학습 데이터 위치의 Score의 크기를 최소화하여, 해당 지점들이 $$\log q(\mathbf{x}; \theta)$$의 Critical Point가 되도록 합니다 (1).
  - 두번째 Term은 각 성분의 2차 미분의 합이 음수로 유지되도록 유도하며, 이는 (1) 조건과 함께 에너지 함수의 Extremum에 해당 위치가 오도록 만듭니다.

- **단점**
  - $$s_\theta(\mathbf{x})$$의 각 성분에 대해 2차 미분이 요구되므로, 입력이 고차원일 경우 계산량이 큽니다.
  - 2차 미분이 $$-\infty$$가 되는 모델은 학습되기 쉽고 Over-fitting이 일어나기 쉽습니다.
  - 이 단점들을 아래 **Denoising Score Matching** 기법으로 개선할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


#### Score Matching 3 ㅣ Denoising Score Matching(DSM)
- 원본 데이터 $$\mathbf{x}$$에 노이즈 $$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$$를 추가하여 Perturbed Sample $$\tilde{\mathbf{x}}$$을 아래와 같이 구성합니다.
    
  $$
  \tilde{\mathbf{x}} = \mathbf{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
  $$
  - $$\sigma$$ ㅣ 노이즈의 Scale
- 이는 평균이 $$\mathbf{x}$$, 분산이 $$\sigma^2 I$$인 정규분포에서 샘플을 생성하는 것과 같습니다.

- **Perturbed Distribution**
  
  $$
  p_\sigma (\tilde{\mathbf{x}}, \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)
  $$
  
  $$
  = \frac{1}{(2 \pi)^{d/2} \sigma^d} \exp(-\frac{1}{2 \sigma^2} \| \tilde{\mathbf{x}} - \mathbf{x}\|^2)
  $$

  $$
  p_\sigma(\tilde{\mathbf{x}}) = \int _{\mathbf{x} \in \mathbb{R}^d} p_\sigma (\tilde{\mathbf{x}} \vert \mathbf{x}) p(\mathbf{x}) d \mathbf{x}.
  $$

- Perturbated Distribution에 대해 Explicit 및 Implicit Score Matching을 적용할 수 있으며,

  $$
  J_{ESM_{p_\sigma}} (\theta) = \frac{1}{2} \mathbb{E}_{p_{\theta} (\tilde{\mathbf{x}})} \big[ \| \nabla_\tilde{\mathbf{x}} \log p_\sigma (\tilde{\mathbf{x}}) - \mathbf{s}_\theta (\tilde{\mathbf{x}}, \sigma) \|^2 \big].
  $$

  $$
  J_{ISM_{p_\sigma}} (\theta) =  \mathbb{E}_{p_{\theta} (\tilde{\mathbf{x}})} \big[ \frac{1}{2} \| \mathbf{s}_\theta (\tilde{\mathbf{x}}, \sigma) \|^2 + tr(\nabla _\tilde{\mathbf{x}} \mathbf{s}_\theta (\tilde{\mathbf{x}}, \sigma)) \big].
  $$

  - If $$\sigma > 0$$,
    $$
    J_{ESM_{p_\sigma}} (\theta) = J_{ISM_{p_\sigma}} (\theta) + C_1
    $$
  - 이를 통해 Perturbed Distribution의 Score를 계산할 수 있고, Over-fitting을 줄일 수 있지만 계산량 문제를 해결할 순 없습니다.

- 계산량 문제를 해결하기 위해, Denoising Score Matching 기법을 사용합니다.
    
  $$
  J_{DSM_{p_\theta}} (\theta) = \frac{1}{2} \mathbb{E}_{p_\theta (\tilde{\mathbf{x}} \vert \mathbf{x})} \big[ \|  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} \vert \mathbf{x}) - \mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma)\| \big]
  $$
    
  - 여기서 학습의 목적은 $$\nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} \vert \mathbf{x})$$를 모델이 잘 근사하도록 하는 것입니다.
    - 즉, Perturbed Sample의 조건부 확률 분포의 Score는 원본 데이터 방향으로의 Denoising 방향이며, 그 크기는 노이즈 수준에 반비례합니다.

  $$
  \nabla _\tilde{\mathbf{x}} \log p_\sigma(\tilde{\mathbf{x}} \vert \mathbf{x}) = \nabla_\tilde{\mathbf{x}} \log \big( \frac{1}{(2 \pi)^{d/2} \sigma^d} \exp (-\frac{1}{2 \sigma^2} \| \tilde{\mathbf{x}} - \mathbf{x} \|^2) \big)
  $$

  $$
  = \nabla_\tilde{\mathbf{x}} \log \frac{1}{(2 \pi)^{d/2} \sigma^d} + \nabla_\tilde{\mathbf{x}} \big( -\frac{1}{2 \sigma^2} \| \tilde{\mathbf{x}} - \mathbf{x} \|^2 \big)
  $$

  $$
  = 0 - \frac{1}{\sigma^2} (\tilde{\mathbf{x}} - \mathbf{x}) = -\frac{1}{\sigma^2} \epsilon
  $$

  - Perturbated Sample의 조건부 확률 분포의 Score는 Perturbation을 Denoising할 방향으로 Scaling한 값이 됩니다.

&nbsp;
&nbsp;
&nbsp;

- 최종적으로, **Denoising Score Matching**의 학습 목적 함수는 다음과 같습니다.
  
  $$
  J_{DSM_{p_\theta}} (\theta) = \frac{1}{2} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2 I), \mathbf{x} \sim p(\mathbf{x})} \big[ \| -\frac{1}{\sigma^2} \epsilon - \mathbf{s}_\theta (\mathbf{x} + \epsilon, \sigma) \|^2 \big].
  $$
  
  - $$J_{DSM_{p_{\sigma}}} (\theta)$$는 노이즈를 추가한 데이터로부터 추가한 노이즈를 제거하는 방향을 예측하도록 학습하는 것이며, 데이터 분포의 Score를 효과적으로 추정할 수 있습니다.
    - Perturbation을 통해 Overfitting을 방지하고, 계산 비용은 입력 차원에 선형적으로 증가합니다.

&nbsp;
&nbsp;
&nbsp;


--------------
## References
1. 오카노하라 다이스케. (2024). 확산 모델의 수학 (손민규 옮김).

2. Vincent, Pascal. "A connection between score matching and denoising autoencoders." Neural computation 23.7 (2011): 1661-1674.

3. Kingma, Durk P., and Yann Cun. "Regularized estimation of image statistics by score matching." Advances in neural information processing systems 23 (2010).
