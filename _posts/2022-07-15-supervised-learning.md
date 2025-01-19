---
title: "지도학습이란? | Supervised Learning(Regression, Classification)"
date: 2022-07-15 17:00:00 +/-TTTT
categories: [AI Theory, Machine Learning]
tags: [lg-aimers, supervised-learning, deep-learning, neural-network, regression, classification, ensemble, optimization, svm]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: 지도학습 이해, 지도학습 개념, 머신러닝 지도학습, 인공지능 지도학습, 딥러닝 지도학습, 지도학습 알고리즘, 지도학습이란, 지도학습 예시
---



--------------------------

> 인공지능의 지도학습(Supervised Learning) 개념과 관련 모델을 정리하며, 모델의 편향과 분산, 오버피팅/언더피팅의 중요성을 강조합니다.
{: .prompt-info }

지도학습에 대한 데이터 크기, 모델 복잡성, 일반화 등의 주제를 다룹니다. 

또한 모델의 편향과 분산, 오버피팅/언더피팅에 대한 이해와 교차 검증의 중요성을 강조합니다.

&nbsp;
&nbsp;
&nbsp;

## **지도학습 <sup>Supervised Learning</sup>**

- Given a set of labeled examples $$(x^1, y^1),...,(x^N, y^N)$$, learn a mapping function $$g : X ➔ Y$$, s.t. given an unseen sample $$x'$$, associated output $$y'$$ is predicted.

- 지도학습은 데이터의 크기와 품질에 크게 의존합니다. 
  - **데이터가 부족한 경우** ㅣ 데이터 증강, 약한 지도학습(Weak Supervision)
  - **데이터 특성이 서로 다른 경우** ㅣ 도메인 적응(Domain Adaptation), 전이 학습(Transfer Learning)

  
### **Problem Formulation**

- $$X = R^d$$는 입력 공간을 의미합니다.
  - $$X = R^d$$ : d-차원의 유클리드 공간(Euclidean Space)
  - 입력 벡터 $$x ∈ X : x = (x_1,...,x_d)$$
- $$Y$$는 출력 공간을 의미합니다. (이진 공간, Binary Space)
- **목표** ㅣ 최적의 목표 함수 $$f$$를 근사하는 것
  - $$f : X ➔ Y$$ (Unknown Ideal Function)
  - **데이터 셋** ㅣ $$(x^1, y^1),...,(x^N, y^N)$$; where $$y^N = f(X^N)$$
  - **가설(Hypothesis)** ㅣ $$g : X ➔ Y$$ (ML model to approximate $$f$$), $$g ∈ H$$



### **모델 일반화 <sup>Model Generalization</sup>**
- **학습(Learning)**은 본질적으로 불완전한 **Ill-posed Problem**이며, 데이터가 제한적이기 때문에 유일한 해(Unique Solution)을 찾기 어렵습니다.
- **일반화(Generalization)**는 보지 못한(Unseen) 데이터에 대해 잘 동작하도록 하는 것을 의미합니다.
  - 일반화 오류(Generalization Error) E<sub>gen</sub>; 해당 오류를 최소화하는 것이 목표이지만 Real World에서 이를 계산하기는 어렵습니다.
  - 대신, 학습(Train)/검증(Validation)/테스트(Test) 오류를 사용하여 근사합니다.



### **오류 <sup>Errors</sup>**

- 샘플 별 오류(Pointwise Error) $$e(h(x), y)$$ 는 각 입력 샘플에 대하여 측정됩니다.
- From a pointwise error to overall errors $$E[(h(x^i) - y^i)^2]$$
  - 입력 데이터가 학습 데이터셋(Train Dataset), 검증 데이터셋(Validation Dataset), 테스트 데이터셋(Test Dataset) 중 하나에서 선택된 경우, 각각 학습 오류($$E_{train}$$), 검증 오류($$E_{val}$$), 테스트 오류($$E_{test}$$)라고 부릅니다.
    - **학습 오류** $$E_{train}$$ ㅣ 학습 데이터셋에서 측정되며, 일반화 오류($$E_{gen}$$)를 반드시 대표하지는 않습니다.
    - **테스트 오류**  $$E_{test}$$ㅣ 학습에 사용되지 않은 데이터셋에서 측정되며, 일반화 오류($$E_{gen}$$)의 근사치로 사용될 수 있습니다.
- **목표** ㅣ $$E_{test} \sim E_{gen} \sim 0$$



### **과적합 <sup>Overfitting</sup>과 과소적합 <sup>Underfitting</sup>**

- **과적합(Overfitting)** ㅣ 실제 데이터 분포에 비해 지나치게 복잡한 모델을 사용하는 경우 발생합니다.
  - **높은 분산(High Variance)**
  - **과적합 방지**
    - 현대의 머신러닝 문제에서는 고차원 데이터를 처리하기 위해 복잡한 모델을 사용하는 경향이 있으며, 데이터 수가 상대적으로 부족하여 과적합 문제가 발생하기 쉽습니다.
    - **차원의 저주(Curse of Dimensionality)** ㅣ 성능을 개선하면서도 샘플의 밀도를 유지하기 위해 데이터의 차원을 늘릴 것인가? 그렇다면 데이터 수를 Exponentially하게 증가시켜야 합니다.
    - **해결책** ㅣ 데이터 증강(Data Augmentation), 정규화(Regularization), 앙상블(Ensemble)
- **과소적합(Underfitting)** ㅣ 실제 데이터 분포에 비해 지나치게 간단한 모델을 사용하는 경우 발생합니다.
  - **High Bias**
  - **과소적합 방지**
  - **해결책** ㅣ 모델을 더 복잡하게 (More Nodes, More Layers) 구성하기


### **편향 <sup>Bias</sup>과 분산 <sup>Variance</sup>**

- **편향(Bias)** ㅣ 모델이 개념을 제대로 표현하지 못해 발생하는 오류
- **분산(Variance)** ㅣ 모델이 학습 데이터의 작은 변화(노이즈)에 과민 반응하여 발생하는 오류

> **총 손실(Total Loss) = 편향(Bias) + 분산(Variance) (+ 노이즈)**

➔ **편향-분산 트레이드오프(Bias-Variance Trade-off)** ㅣ 모델 복잡도에 따라 근사와 일반화 성능 간의 트레이드-오프가 존재합니다.



### **교차 검증 <sup>Cross Validation, CV</sup>**

- 교차 검증(Cross Validation)은 과적합을 방지하면서 더 나은 모델을 선택할 수 있도록 도움을 줍니다. (하지만 더 복잡해 질 수 있음)



&nbsp;
&nbsp;
&nbsp;


## **회귀 <sup>Regression</sup>**
### **선형 회귀 <sup>Linear Regression</sup>**

- 가설 집합(Hypothesis Set) $$H$$, 모델 매개변수 $$\theta$$

$$
h_\theta(x)=\theta_0+\theta_1x_1+...+\theta_dx_d=\theta^Tx
$$

- 첫 시도에 적합하며, 단순성과 일반화에 유리합니다.
- L<sub>2</sub> 비용 함수(Cost Function) 
  - **목표** ㅣ 평균제곱오차(MSE) 최소화

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
$$

➔ $$\text{minimize}_{\theta_0, \theta_1} J(\theta_0, \theta_1)$$



#### **최적화 <sup>Optimization</sup>**

- 데이터 행렬 $$X \in \mathbb{R}^{N \times (d+1)}$$, 타겟 벡터 $$y \in \mathbb{R}^{N}$$, 가중치 벡터 $$\theta \in \mathbb{R}^{d+1}$$에 대하여,
- **In-sample Error** ㅣ $$\mid\mid{y - X\theta}\mid\mid_2$$
- **Normal Equation**
  - (최소 제곱) $$E$$는 연속적(Continuous)이고 미분 가능(Differentiable)하며 볼록(Convex)한 것으로 가정되어야 합니다.
  - 역행렬 계산이 포함되므로, 샘플 수가 매우 많을 경우 느리게 작동합니다.

$$
\theta^{*}=\text{argmin}_{\theta}E(\theta)
$$

$$
=\text{argmin}_{\theta}[\frac{1}{N}(\theta^TX^TX\theta-2\theta^TX^Ty+y^Ty)]
$$


$$
\therefore\theta^{*}=(X^TX)^{-1}X^Ty=X^{+}y
$$

➔ **문제** ㅣ 계산 복잡성 증가, 비가역 행렬 ➔ 반복 알고리즘(경사 하강법)이 필요합니다.

- **과적합 방지**
  - 특징(feature)이 너무 많으면 가설이 학습 데이터에는 잘 맞을 수 있으나, 새 데이터에 일반화되지 않을 수 있습니다.
  > **더 많은 특징 ➔ 더 많은 매개변수 ➔ 더 많은 데이터 필요**
  - 일반적으로, 데이터가 부족하면 과적합이 발생하며, 이때 특징 수를 줄이거나 **정규화**를 적용해야 합니다.



#### **경사 하강법 <sup>Gradient Descent</sup>**
- $$J$$는 최적화 하려는 목적 함수(Objective Function)이며, $$\alpha$$는 학습률(Learning Rate)을 나타냅니다.
  - $$\alpha$$가 너무 작으면 경사 하강법이 느리게 작동하며, 너무 크면 최소 값을 지나칠 수 있습니다 (Overshoot).

$$
\theta_{new}=\theta_{old}-\alpha\frac{\partial}{\partial\theta}J(\theta)
$$


- 일반적으로, 경사 하강법은 샘플 수(n)가 많아도 잘 작동합니다.
- 모든 샘플(배치)을 한 번에 사용하는 대신 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 또는 미니 배치(Mini-batch)를 사용합니다.
- **Advances** ㅣ AdaGrad, RMSProp, Adam
- **Limitation** ㅣ 국소 최적값(Local Optimum)
  - 국소 최적값에 빠질 가능성이 있으므로, 전역 최적값(Global Optimum)을 보장할 수는 없으나 좋은 국소 최적값을 찾으려고 노력합니다.
  - **<u>국소 최적값을 피하는 방법</u>**
    - **Momentum** ㅣ 고차원 및 작은 잡음이 있는 기울기에서 학습 속도를 높이도록 설계되었으며, 과거 기울기에 대하여 지수적 가중 이동 평균(Low-pass Filtering)을 사용합니다.
    - **SGD + Momentum** ㅣ 고차원 및 작은 잡음이 있는 기울기에서 학습 속도를 높일 수 있도록 설계되었습니다.
    - **Nesterov Momentum** ㅣ 기울기 를 선행 기울기 단계에서 평가한다는 점에서 표준 Momentum과의 차이를 보입니다.
    - **AdaGrad** ㅣ 각 방향에 대해 개별적으로 학습률을 조정합니다.
    - **RMSProp** ㅣ 누적 기울기가 커질 경우 학습률이 매우 작아져 더 이상 학습하지 못하는 문제를 해결합니다.
    - **Adam** ㅣ RMSProp + Momentum
- **학습률 스케줄링(Learning Rate Scheduling)**
  - 학습이 진행됨에 따라 학습률을 점차적으로 감소시키는 방법입니다.


&nbsp;
&nbsp;
&nbsp;

## **분류 <sup>Classification</sup>**
### **선형 분류 <sup>Linear Classification</sup>**
- 설명 변수(Explanatory Variable)의 선형 결합을 기반으로 데이터를 분류하기 위하여, 초평면(Hyperplane)을 결정 경계(Decision Boundary)로 사용합니다.
- 선형 공식 $$g \in H$$는 다음과 같이 표현됩니다.

$$
h(x)=sign((\sum^{d}_{i=1}w_ix_i)+w_0)
$$

➔ $$x_0 = 1, w_0$$ : a bias term, $$sign(x) = 1{\quad}if{\quad}x>0;0{\quad}if{\quad}x<0;$$

- **시그모이드 함수(Sigmoid Function)**
  - 점수 값을 확률 값으로 매핑하기 위해 사용됩니다.
  - 선형 함수의 출력을 압축합니다.

$$
\sigma(-w^Tx)=\frac{1}{1+e^{-w^Tx}}
$$

- **장점** ㅣ 단순성, 해석용이성


#### **손실 함수 <sup>Loss Function</sup>**

- **Hinge Loss**

$$
\text{Loss}_\text{hinge}(x,y,w)=\text{max}\{1-(w\cdot\phi(x))y, 0\}
$$

- **크로스 엔트로피 손실(Cross Entropy Loss)**
  - 두 확률 질량 함수(pmf) {p, 1-p}와 {q, 1-q}를 이진 결과를 기반으로 고려합니다.
  - 크로스 엔트로피 손실은 출력이 0과 1 사이 확률 값인 분류 모델의 성능을 측정합니다.

$$
CE(S,Y)=-\sum_{\forall{i}}Y_i\text{log}(S_i))
$$

&nbsp;
&nbsp;
&nbsp;

### **성능 평가 <sup>Performance Evaluation</sup>**

- **Accuracy** = (TP+TN)/ALL
- **Precision** = TP/(TP+FP)
- **Recall** = TP/(TP+FN)
- **F1** = PxR/(P+R)
- **TPR(True Positive Rate)** = R = TP/(TP+FN)
- **TNR(True Negative Rate)** = TN/(TN+FP)
- **False positive error** ㅣ 예측 = 양성(Positive), 실제 = 음성(Negative)
- **False negative error** ㅣ 예측 = 음성(Negative), 실제 = 양성(Positive)
- **ROC Curve** : TPR과 FPR 사이의 관계를 나타냅니다.

&nbsp;
&nbsp;
&nbsp;

## **서포트 벡터 머신 <sup>Support Vector Machine, SVM</sup>**
- 양쪽에서 가장 큰 마진(Margin)을 가지는 선형 분리기(Linear Separator)를 선택합니다.
  - 최대 마진 초평면(Hyperplane)과 서포트 벡터(Support Vector)를 활용합니다.
  - 이상치(Outlier)에 대한 Robustness를 가집니다.

### **서포트 벡터 <sup>Support Vector</sup>**

- 최소 마진을 가지며, 모델 성능에 가장 민감하게 영향을 미치는 데이터 포인트를 의미합니다.


### **마진 <sup>Margin</sup>**

- 초평면으로부터 양쪽에서 가장 가까운 데이터 포인트까지의 거리의 두 배를 뜻합니다.

- $$w$$ ㅣ 초평면에 수직인 벡터


### **최적화 <sup>Optimization</sup>**

- 최적의 가중치 $$w$$와 편향 $$b$$를 구하는 작업입니다.
- 데이터를 정확히 분류하면서 가능한 가장 큰 마진을 얻는 것이 목표입니다.
- **Hard margin SVM** ㅣ 선형적으로 분류 가능하다는 가정
- **Soft margin SVM** ㅣ 선형적으로 분류하지 못한다는 가정
  - 비선형 변환 및 커널 트릭(Kernel Trick) 활용할 수 있습니다.

- **제약 조건(Constraints)** ㅣ 선형적으로 구분 가능하며(Linearly Separable), Hard-Margin Linear SVM

$$
h(x)=w^Tx+b\geq1\text{ for }y=1
$$

$$
h(x)=w^Tx+b\leqq-1\text{ for }y=-1
$$

$$
y(w^Tx+b)\geq1\text{ for all samples}
$$

- **목적 함수(Objective function)** ㅣ 선형적으로 구분 가능하며(Linearly Separable), Hard-Margin Linear SVM
  - 서포트 벡터와 초평면 간의 거리:

$$
\frac{w^Tx+b}{\mid\mid{w}\mid\mid}=\frac{\pm1}{\mid\mid{w}\mid\mid}\longrightarrow\frac{2}{\mid\mid{w}\mid\mid}
$$


### **커널 트릭 <sup>Kernel Trick</sup>**

> 선형적으로 구분 가능하지 않은 경우, Not Linearly Separable

- **다항식 커널(Polynomial Kernel)**

$$
K(x,y)=(x\cdot{y}+1)^p
$$

- **가우시안 RBF(Gaussian Radial Basis Function) 커널**:

$$
K(x,y)=e^{-\mid\mid{x-y}\mid\mid^2/2\sigma^2}
$$

- **쌍곡 탄젠트 커널(Hyperbolic Tangent, MLP Kernel)**:

$$
K(x,y)=\text{tanh}(kx\cdot{y}-\delta)
$$


&nbsp;
&nbsp;
&nbsp;


## **인공 신경망 <sup>Artificial Neural Network, ANN</sup>**

- 성능을 향상시키기 위해 정교한 학습 방법이 필요합니다.

- **활성화 함수(Activation Functions)**
  - 시그모이드 뉴런은 총 입력값의 매끄럽고 유한한 실수값을 출력합니다.
  - 활성화 함수에 의한 비선형성을 부여합니다.
- 심층 신경망(Deep Neural Network)은 뉴런 수를 늘려 더 복잡한 비선형 경계를 표현할 수 있습니다.
- **다층 퍼셉트론(Multilayer Perceptron, MLP)**
  - XOR 문제 해결 가능합니다.
- **비선형 문제에서의 ANN**
  - 층(Layer) 수가 많아도 정확도가 낮은 경우 존재합니다.
  - ANN의 출력은 시그모이드 함수의 결과값이며, 이 출력값의 반복적인 곱셈으로 인해 결과가 0에 가까워질 수 있습니다.     
  ➔ **기울기 소실(Vanishing Gradient) 문제**


### **역전파 <sup>Back Propagation</sup>**

- 역전파로 인해 하위 계층의 매개변수가 거의 변화하지 않을 수 있습니다.
➔ **기울기 소실(Vanishing Gradient) 문제**

- **<u>Breakthrough</u>**
  - **사전 훈련(Pre-training) + 미세 조정(Fine-tuning)**
  - **합성곱 신경망(CNN)** ㅣ 중복된 매개변수 감소
  - **ReLU(Rectified Linear Unit)** ㅣ 일정한 기울기 전파
  - **Dropout** ㅣ 과적합 방지


&nbsp;
&nbsp;
&nbsp;


## **앙상블 학습 <sup>Ensemble Learning</sup>**
- 학습 데이터로 학습된 다양한 분류기(Classifier, Experts)들의 예측을 결합하여, 보지 않은(Unseen) 데이터에 대한 예측 라벨을 부여하는 방법입니다.
- **배깅(Bagging)과 부스팅(Boosting)**
  - **배깅** ㅣ 랜덤 포레스트(Random Forest) ➔ 내재적으로 Boosting
  - **부스팅**: 그래디언트 부스팅 머신(Gradient Boosting) ➔ 일반화된 AdaBoost
- **장점**
  - 예측 성능이 향상되며, 다른 유형의 분류기를 직접적으로 포함시킬 수 있습니다.
  - 구현이 간단하고 파라미터 튜닝이 적습니다.
- **Disadvantages**
  - 컴팩트한 표현(Compact Representation)을 제공하지 않습니다.


### **배깅 <sup>Bagging</sup>**

- **부트스트래핑 + 집합(Aggregating)**
- 여러 모델을 병렬로 학습합니다.
- 배깅은 투표/평균을 통해 분산을 줄여 과적합을 방지하므로 효과적입니다.
  - 학습 알고리즘이 불안정할 때 즉, 학습 데이터의 작은 변화가 학습된 분류기에서 큰 변화를 일으킬 경우에 도움을 줍니다.
  - 일반적으로, 분류기가 많을수록 더 나은 성능을 보입니다.

### **부스팅 <sup>Boosting</sup>**

- 약한 분류기들의 연쇄적 학습을 틍해, 여러 모델을 순차적으로 학습합니다.
  - **AdaBoost** ㅣ 가중치는 이전 분류기의 성능에 따라 조정되고, 최종 분류기를 결합합니다.
- 간단하고 유연하며, 비모수적(Non-parametric)이고 범용적으로 사용 가능합니다.
- 약한 학습기에 대한 사전 지식(Prior Knowledge)을 필요로하지 않습니다.

&nbsp;
&nbsp;
&nbsp;

---------------------
## Reference
> 본 포스팅은 LG Aimers 프로그램에서 학습한 내용을 기반으로 작성되었습니다. (전체 내용 X)
{: .prompt-warning }

1. LG Aimers AI Essential Course Module 2. 지도학습(분류/회귀), 이화여자대학교 강제원 교수 

