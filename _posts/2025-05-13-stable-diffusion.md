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
- N개의 학습 데이터 $$D = \{ \mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)} \}$$가 미지의 확률분포 $$p(\mathbf{x})$$로 부터 서로 독립적으로 추출된 것이라고 가정할 때, 생성 모델은 확률분포 $$q_{\theta}(\mathbf{x})$$를 가지며 해당 분포에 따라 데이터를 추출합니다.
  - $$\mathbf{x} \sim q_{\theta}(\mathbf{x})$$
  - $$\theta$$는 신경망의 매개변수와 같은 확률분포의 특징을 나타냅니다.
  - **목표** ㅣ 목표 확률 분포 $$p(\mathbf{x})$$와 가능한 가까운 확률분포 $$q_{\theta}(\mathbf{x})$$를 가지는 생성 모델을 얻는 것.
    - 두 분포가 얼마나 가까운지를 확인하는 닮음 지표로 **KL-Divergence**와 **Wasserstein Distance**를 사용합니다.

- 텍스트와 같은 타 데이터에 대응하는 이미지 $$\mathbf{x}$$를 생성하는 문제는 Joint Probability $$p(\mathbf{x}, \mathbf{c})$$ 또는 Conditional Probability $$p(\mathbf{x} \vert \mathbf{c})$$를 이용하는 생성모델로 이해할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


- 데이터 $$x \in X$$에 대한 생성모델의 확률 분포 $$q_{\theta}(\mathbf{x})$$는 다음과 같습니다.

$$
q_\theta(\mathbf{x}) = \gamma_{\theta}(\mathbf{x})/Z(\theta), \\
Z(\theta) = \int_{\mathbf{x}' \in X} \gamma_{\theta}(\mathbf{x}' d \mathbf{x}').
$$

- - where $$\gamma_{\theta}(\mathbf{x}) \geq 0$$ is an unnormalized probability density function(PDF) and $$Z(\theta) > 0$$ is a partition function(normalization constant).
  - Partition function $$Z(\theta)$$는 데이터 공간의 모든 정보를 가지기 때문에 이를 활용하면 데이터 전체의 여러 통계량을 계산할 수 있으며, $$\int q_\theta(\mathbf{x}) d \mathbf{x}= 1$$ 즉 확률밀도가 되도록 합니다.


&nbsp;
&nbsp;
&nbsp;


- **<u>Energy-based model(EBM)</u>**

  - Unnormalized PDF $$\gamma_{\theta}(\mathbf{x}) = \exp(-f_{\theta}(\mathbf{x}))$$

  - i.e.,
    $$
    q_\theta(\mathbf{x}) = \exp(-f_{\theta}(\mathbf{x}))/Z(\theta), \\
    Z(\theta) = \int_{\mathbf{x}' \in X} \exp(-f_{\theta}(\mathbf{x}')) d\mathbf{x}'.
    $$

  - 에너지 $$f_{\theta}(\mathbf{x})$$가 작으면, $$\mathbf{x}$$ 데이터는 출현하기 쉬운 데이터로 이해할 수 있습니다.

  - 이러한 에너지 함수는 확률분포로서의 제약이 크게 없어 모델링이 자유롭지만, Partition function을 계산해야 하므로 고차원 입력 데이터를 다룰 경우 어려움이 있습니다.

&nbsp;
&nbsp;
&nbsp;


### **Training**

#### 0. KL Divergence

- **확률 분포 $$p(\mathbf{x})$$로 부터 $$q(\mathbf{x})$$로의 KL Divergence**
  $$
  D_{KL}(p \| q) := \int_x p(x) \log \frac{p(x)}{q(x)} dx.
  $$

  - If $$p(x) = q(x),$$ then $$D_{KL}(p \| q) = 0$$.
  - else,  $$D_{KL}(p \| q) > 0$$.

- 두 확률분포가 다를수록 KL Divergence는 큰 양의 값을 가집니다.


&nbsp;
&nbsp;
&nbsp;


#### 1. Likelihood-based Model

- **<u>Likelihood-based Model</u>**

  - 데이터 $$\mathbf{x}$$의 생성확률 또는 우도 $$q_\theta(\mathbf{x})$$를 명시적 확률분포로 정의하고, 그 분포의 **로그 우도(Log-likelihood)를 최대화**하는 방식으로 매개변수를 추정합니다.

  - N개의 학습 데이터 $$D = \{ \mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)} \}$$는 서로 독립적으로 추출되었기에, D의 우도는 다음과 같이 각 데이터 우도의 곱으로 정의됩니다.
    $$
    q_\theta(D) = \Pi_i q_\theta(\mathbf{x}^{(i)}).
    $$

  - 로그 우도는 다음과 같습니다.
    $$
    xL(\theta) = \frac{1}{N} \log q_\theta(D) = \frac{1}{N} \sum_i \log q_\theta(\mathbf{x}^{(i)}).
    $$

  - VAE, Auto-regressive Model, Energy-based Model과 같은 생성 모델이 Likelihood-based Model 입니다.


&nbsp;
&nbsp;
&nbsp;

- **<u>Maximum Likelihood Estimation(MLE)</u>**

  - $$\theta^{*}_{ML}$$를 통해 매개변수를 추정합니다.
    $$
    \theta^{*}_{ML} := arg\max_\theta L(\theta)
    $$

  - **ex. MLE of Energy-based Model**

    - 아래의 첫번째 Term을 통해 학습 데이터 위치의 에너지를 줄이고 (1), 두번째 Term을 통해 그 외 모든 위치의 에너지를 높이는 (2) Parameter를 구하는 과정을 의미합니다.

    $$
    L(\theta) = \frac{1}{N} \sum^N _{i=1} \log q_\theta (\mathbf{x}^{(i)})
    $$
  
    $$
    = -\frac{1}{N} \sum^N _{i=1} \big[ f_\theta(\mathbf{x}^{(i)}) \big] - \log Z(\theta)
    $$
    
    $$
    = -\frac{1}{N} \sum^N _{i=1} \big[ f_\theta(\mathbf{x}^{(i)}) \big] - \log \int_{\mathbf{x}' \in X} \exp(-f_\theta(\mathbf{x}')) d \mathbf{x}'.
    $$

    - 하지만, 고차원 데이터 $$\mathbf{x}$$에 대하여 (1)번 작업을 수행한다면, 학습 데이터 외에도 에너지가 낮아지는 위치가 무수히 많이 발생하게 되며, 이는 (2)번 작업 또한 어렵게 만듭니다.

      - 즉, 무수히 많은 위치에 실제로 존재하지 않는 데이터가 생성될 수 있습니다.

    - 학습 데이터나 정답 데이터의 우도를 활용하면 학습 진행 중 모델을 평가할 수 있습니다. 

    - **Log-likelihood의 기울기**는 학습 데이터와 모델 분포 간 에너지 기울기 차이로 표현될 수 있는데, Energy-based Model은 모델 분포에서의 기댓값 계산이 어려워 직접 추정하기 어렵습니다.
      
      $$
      \frac{\partial L(\theta)}{\partial \theta} = -\frac{1}{N} \sum^N _{i=1} \big[ \frac{\partial f_\theta(\mathbf{x}^{(i)})}{\partial \theta} \big] - \frac{\partial}{\partial \theta} \log Z(\theta) \\
      $$

      $$
      = -\frac{1}{N} \sum^N _{i=1} \big[ \frac{\partial f_\theta(\mathbf{x}^{(i)})}{\partial \theta} \big] + \mathbb{E}_{\mathbf{x} \sim q_\theta(\mathbf{x})} \big[ \frac{\partial f_\theta (\mathbf{x})}{\partial \theta} \big].
      $$


      - Markov Chain Monte Carlo(MCMC) 샘플링으로 Partition Function 계산 없이 근사할 수 있지만, 계산 비용이 크고 분산이 크다는 한계가 있습니다.


&nbsp;
&nbsp;
&nbsp;


- **<u>KL Divergence와 MLE</u>**
  - **MLE는 $$D_{KL}(p \| q)$$를 최소화하는 문제**입니다.
  - $$p(\mathbf{x})>0$$일 때, 분모인 $$q(\mathbf{x})$$가 작다면 큰 Penalty가 생길 수 있어, 모델은 가능한 모든 Mode를 포함하도록 학습합니다.
  - 틀린 데이터에 Penalty를 주기 어렵습니다.


&nbsp;
&nbsp;
&nbsp;


#### 2. Implicit Generative Model

-  명시적인(Explicit) 확률 분포 $$q_\theta(\mathbf{x}))$$를 정의하지 않고, 샘플을 직접 생성하는 생성 함수만을 통해 데이터를 모델링 하는 방식입니다.
   - 표본 추출 과정으로 확률 분포가 Implicit하게 표현되는 모델로 Likelihood가 Explicit하게 구해지지는 않습니다.
   - ex. 정규 분포로부터 추출된 잠재변수를 신경망과 같은 결정론적 함수로 변환하여 얻어진 분포로 확률 분포를 표현할 경우, 이를 결정론적 함수에 의한 Push-forward 분포라고 부릅니다 (GAN).
-  학습이 불안정해지기 쉬우며, 학습 진행 중 모델을 평가하기 어렵습니다.
-  Partition Function을 Explicit하게 계산할 필요가 없어, 높은 표현력을 가지는 모델 생성에 활용하기 용이합니다.

&nbsp;
&nbsp;
&nbsp;

- **<u>KL Divergence와 Implicit Generative Model</u>**

  - 이는 **Inverse KL Divergence $$D_{KL}(q \| p)$$를 최소화하는 문제**입니다.
  - 아래의 Jensen-Shannon Divergence를 최소화하는 문제로 볼 수 있습니다.

  $$
  D_{JS}(p \| q) = \frac{1}{2} D_{KL}(p \| \frac{1}{2}(p+q)) + \frac{1}{2} D_{KL}(q \| \frac{1}{2}(p+q)).
  $$

  - $$q(\mathbf{x})>0$$일 때, $$p(\mathbf{x})$$가 작으면 큰 Penalty가 생길 수 있어, 일부의 큰 Mode를 파악하도록 학습합니다.
  - Mode가 일부분에 집중되거나 놓치는 Mode Collapse가 일어나기 쉽습니다. 
    - ex. GAN


&nbsp;
&nbsp;
&nbsp;


### **Score**

- **Score**
  $$
  s(\mathbf{x}) := \nabla_{\mathbf{x}} \log p(\mathbf{x}) : \mathbb{R}^d \rightarrow \mathbb{R}^d, \\
  = \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})}.
  $$


  - $$\log p(\mathbf{x})$$에서의 $$\mathbf{x}$$에 대한 기울기를 뜻합니다.
  - $$\mathbf{x}$$와 같은 차원을 가지는 벡터입니다.
  - 임의의 입력에 대하여 미분 가능한 확률 분포입니다.
    - 입력 공간에서의 벡터장을 나타내며, 각 점의 벡터는 당연하게도 그 위치에서 로그 우도가 가장 급격히 커지는 방향과 그 크기를 나타냅니다.

- Since $$\nabla_\mathbf{x} \log Z(\theta) = 0$$,
  $$
  \nabla_{\mathbf{x}} \log q_\theta(\mathbf{x}) = - \nabla_\mathbf{x} f_\theta(\mathbf{x}) - \nabla_\mathbf{x} \log Z(\theta) \\= - \nabla_\mathbf{x} f_\theta(\mathbf{x})
  $$

  - Score는 에너지 함수의 입력에 대한 음의 기울기와 같습니다.
  - Score를 사용하면 현재 위치에서 어느 방향으로 학습을 진행하여야 확률이 높은 영역에 도달할 수 있는지 알 수 있어, 고차원 공간에서 확률이 높은 영역을 효율적으로 탐색할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


#### Langevin Monte Carlo

- Score를 사용하는 MCMC 방법이며, 해당 과정을 반복하면 $$p(\mathbf{x})$$로 부터 최종 표본을 얻을 수 있습니다.

  - **확률 분포의 점수가 계산되면, Langevin Monte Carlo 방법을 활용하여 해당 확률 분포로부터 표본을 추출할 수 있습니다.**

- 임의의 Prior Distribution $$\pi(\mathbf{x})$$로 부터 데이터를 $$\mathbf{x}_0 \sim \pi(\mathbf{x})$$로 추출하여 각 위치에서의 Score에 따라 전이합니다.

  - 이때, 정규분포로부터 추출된 노이즈 $$\sqrt{2 \alpha} \mathbf{u}_{i+1}$$를 추가하며, 해당 이동을 K번 반복합니다.

  $$
  \mathbf{x}_{i+1} = \mathbf{x}_i + \alpha \nabla_\mathbf{x} \log p(\mathbf{x}_i) + \sqrt{2 \alpha} \mathbf{u}_{i+1}.
  $$

  - 만약 $$\alpha \rightarrow 0 \,\, \text{and} \,\, K \rightarrow \inf$$ 이면, $$\mathbf{x}$$는 $$p(\mathbf{x})$$로 부터의 표본에 수렴합니다.



- 데이터는 Score에 따라 데이터의 우도가 큰 영역을 중심으로 전이하지만, 노이즈를 추가하면 극댓값으로부터 탈출할 수 있어 확률 분포 전체를 파악할 수 있습니다.
  - 고차원 공간에서 확률이 높은 영역을 효율적으로 탐색할 수 있습니다.

&nbsp;
&nbsp;
&nbsp;

#### Score Matching

- **Score-based Model(SBM)**
  - 확률분포를 직접 학습하지 않고, 확률 분포의 Score를 학습하여 해당 Score를 활용하여 생성 모델을 구현하는 모델을 뜻합니다.
- 확률 분포의 경우 일반 함수와 다르게 그 총합이 1이라는 제약이 있어, 모든 입력에서의 Score만 일치하면 동일한 확률 분포임을 알 수 있습니다.
- **Score를 어떻게 학습하는지**에 대한 방법론들은 다음과 같습니다.



- **1. Explicit Score Matching(ESM)**
  - 모델 $$s_\theta(\mathbf{x}):\mathbb{R}^d \rightarrow \mathbb{R}^d$$에 대하여, 학습 대상의 Score와 모델 출력간의 제곱 오차가 최소화되는 파라미터를 구해보겠습니다.

    - 목표 분포인 $$p(\mathbf{x})$$에 대한 Expectation을 계산한다면 다음과 같습니다.

    $$
    J_{ESM_p}(\theta) = \frac{1}{2} \mathbb{E}_{p(\mathbf{x})} \big[ \| \nabla_\mathbf{x} \log p(\mathbf{x}) - s_\theta(\mathbf{x}) \| ^2 \big].
    $$

  - $$J_{ESM_p}(\theta)$$는 직접 함수를 추정하는 방법이지만, 대부분의 확률 분포는 Score $$\nabla_\mathbf{x} \log p(\mathbf{x})$$를 알 수 없기에 그대로 적용할 수 없습니다.


&nbsp;
&nbsp;
&nbsp;


- **2. Implicit Score Matching(ISM)**
  - Score $$\nabla_\mathbf{x} \log p(\mathbf{x})$$를 사용하지 않고 학습 목표를 정의합니다.
    
    $$
    J_{ISM_p} (\theta) = \mathbb{E}_{p(\mathbf{x})} \big[ \frac{1}{2} \| s_\theta(\mathbf{x}) \|^2 + tr(\nabla_\mathbf{x} s_\theta(\mathbf{x})) \big], \\
    $$    

    $$
    tr(\nabla_\mathbf{x} s_\theta(\mathbf{x})) = \sum^d_{i=1} \frac{\partial s_\theta (\mathbf{x})_i}{\partial x_i}
    $$
    
    $$
    = - \sum^d _{i=1} \frac{\partial^2 f_\theta(\mathbf{x})}{\partial x^2 _i}
    $$

    - where $$s_\theta(\mathbf{x})$$ is estimated score by model.
    - 이는 Explicit Score Matching을 활용하여 학습한 결과와 일치하는 것으로 알려져 있습니다.

    - $$\mathbb{E}_{p(\mathbf{x})}$$에서 실제로 $$p(\mathbf{x})$$는 알 수 없기 때문에, 학습 데이터 $$D$$에 대한 평균값으로 기댓값을 치환하여 목적함수를 재 정의할 수 있습니다.
      
      $$
      J_{ISM_{discrete}}(\theta) = \frac{1}{N} \sum^N _{i=1} \big[ \frac{1}{2} \| s_\theta(\mathbf{x}^{(i)}) \|^2 + tr(\nabla_\mathbf{x} s_\theta (\mathbf{x}^{(i)})) \big].
      $$

      - 첫번째 Term은 학습 데이터 위치의 Score의 절댓값을 최소화하는 것이며, 학습데이터의 위치 $$\mathbf{x}^{(i)}$$가 Log Likelihood $$\log q(\mathbf{x}; \theta)$$의 Critical Point가 되도록 합니다.
      - 두번째 Term은 각 성분의 2차 미분의 합을 음수로 한다는 것이며, 첫번째 Term의 Critical Point가 되도록 하는 조건과 함께 한다면 학습 데이터의 위치가 에너지 함수의 극값이 되도록 한다는 의미입니다.

    - **단점**
      - $$\mathbb{E}_{p(d)} \big[ tr(\nabla_\mathbf{x} s_\theta (\mathbf{\mathbf{x}})) \big]$$을 계산하려면, $$s_\theta(\mathbf{x})$$의 각 성분마다 Error Back-propagation을 적용해야 하지만 이는 입력이 고차원일 경우 거의 불가능합니다.
      - 2차 미분이 $$-\infty$$가 되는 모델은 학습되기 쉽고 Over-fitting이 일어나기 쉽습니다.
      - 이 단점들을 아래 Denoising Score Matching 기법으로 해결할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


- **3. Denoising Score Matching(DSM)**
  - $$\tilde{\mathbf{x}}$$ ㅣ $$\mathbf{x}$$에 $$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$$ 노이즈를 추가한 변수
    
    $$
    \tilde{\mathbf{x}} = \mathbf{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
    $$
    - $$\sigma$$ ㅣ 노이즈의 Scale
  - 위 과정은 평균이 $$\mathbf{x}$$, 분산이 $$\sigma^2 I$$인 정규분포로부터 표본을 얻는 과정으로 이해할 수 있습니다.

  - **Perturbated Distribution $$p_\sigma(\tilde{\mathbf{x}})$$**
  
    $$
    p_\sigma (\tilde{\mathbf{x}}, \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)
    $$
  
    $$
    = \frac{1}{(2 \pi)^{d/2} \sigma^d} \exp(-\frac{1}{2 \sigma^2} \| \tilde{\mathbf{x}} - \mathbf{x}\|^2)
    $$

    $$
    p_\sigma(\tilde{\mathbf{x}}) = \int _{\mathbf{x} \in \mathbb{R}^d} p_\sigma (\tilde{\mathbf{x}} \vert \mathbf{x}) p(\mathbf{x}) d \mathbf{x}.
    $$

  - Explicit Score Matching with Perturbated Distribution
    $$
    J_{ESM_{p_\sigma}} (\theta) = \frac{1}{2} \mathbb{E}_{p_{\theta} (\tilde{\mathbf{x}})} \big[ \| \nabla_\tilde{\mathbf{x}} \log p_\sigma (\tilde{\mathbf{x}}) - \mathbf{s}_\theta (\tilde{\mathbf{x}}, \sigma) \|^2 \big]
    $$

  - Implicit Score Matching with Perturbated Distribution
    $$
    J_{ISM_{p_\sigma}} (\theta) =  \mathbb{E}_{p_{\theta} (\tilde{\mathbf{x}})} \big[ \frac{1}{2} \| \mathbf{s}_\theta (\tilde{\mathbf{x}}, \sigma) \|^2 + tr(\nabla _\tilde{\mathbf{x}} \mathbf{s}_\theta (\tilde{\mathbf{x}}, \sigma)) \big]
    $$

    - If $$\sigma > 0$$,
      $$
      J_{ESM_{p_\sigma}} (\theta) = J_{ISM_{p_\sigma}} (\theta) + C_1
      $$
    - 이를 통해 Perturbated Distribution의 Score를 계산할 수 있고, Over-fitting을 줄일 수 있지만 계산량 문제를 해결할 순 없습니다.

  - Denoising Score Matching은 직접 Score를 목표로 하여 학습하는 것이 아닌, Perturbation이 발생하였을 때의 조건부 확률에 대한 Score를 목표로 학습합니다.

    $$
    J_{DSM_{p_\theta}} (\theta) = \frac{1}{2} \mathbb{E}_{p_\theta (\tilde{\mathbf{x}} \vert \mathbf{x})} \big[ \|  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} \vert \mathbf{x}) - \mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma)\| \big]
    $$

    - 이는 Original Distribution과 Perturbated Distribution의 동시 확률로 Expectation을 계산하고 있으며, 그 목표가 $$ \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) $$ (Perturbated Distribution의 Score)가 아니라 $$ \nabla_\tilde{\mathbf{x}} \log p_\sigma(\tilde{\mathbf{x}} \vert \mathbf{x})$$ (조건부 확률의 Score) 임을 보여줍니다.
  
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

  - 결론적으로, **Denoising Score Matching**은 다음과 같습니다.
  
  $$
  J_{DSM_{p_\theta}} (\theta) = \frac{1}{2} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, sigma^2 I), \mathbf{x} \sim p(\mathbf{x})} \big[ \| -\frac{1}{\sigma^2} \epsilon - \mathbf{s}_\theta (\mathbf{x} + \epsilon, \sigma) \|^2 \big]
  $$
  
  - $$J_{DSM_{p_{\sigma}}} (\theta)$$는 노이즈를 추가한 데이터로부터 추가한 노이즈를 예측하는 문제를 푸는 것으로, 데이터 분포의 Score를 구할 수 있습니다.
    - Perturbation을 추가하여 Overfitting을 막을 수 있으며 계산량은 입력 차원 $$d$$에 선형적입니다.



--------------
## Reference
[^ref1]: 
