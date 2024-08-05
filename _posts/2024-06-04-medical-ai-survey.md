---
title: "의료 인공지능의 모든 것 | Medical AI"
date: 2024-06-04 00:00:00 +/-TTTT
categories: [AI Theory, Medical AI]
tags: [medical-ai, classification, detection, segmentation, registration, supervised-learning, svm, tree, xai]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
pin: false
image:
  path: 20240604-t.jpg
  alt: ""
description: 의료 인공지능, 헬스케어 인공지능, Medical AI Trend | 
---

> 의료 분야 인공지능에 대한 설명과 관련 기술 및 최신 동향을 설명합니다.
{: .prompt-info }

AI를 활용한 의료 이미지 분석 분야에서는 머신러닝 및 딥러닝을 활용한 이미지/영상 분류 (Classification), 객체 탐지 (Object Detection),  분할 (Segmentation), 정합 (Registration) 등의 방법론들이 널리 활용되고 있습니다.

머신러닝 기법들 중 특히 서포트 벡터 머신, 군집화, 결정 트리 및 랜덤 포레스트를 통해 암, 유전 데이터를 포함한 다양한 의학 데이터를 예측하는 연구가 많이 있습니다.

딥러닝을 기반으로 적은 데이터셋에 효과적인 모델 학습 방식을 적용하기 위해 Self-supervised Learning, Multi-instance Learning, Inductive Learning 및 Transfer Learning 방법들이 고려됩니다.

이처럼 다양한 AI 기법들이 신경 (Neruo), 망막 (Retinal), 폐 (Pulmonary), 병리학 (Pathology), 유방 (Breast), 심장 (Cardiac), 근골격계 (Musculoskeletal) 등의 의료 분야에 적용될 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


## **Introduction**
- 1970년부터 1990년 즈음에는 저수준의 픽셀 처리와 수학적 모델링을 기반으로 한 **Rule-based system** 및 전문가 시스템이 널리 활용됨
- 1990년대 후반부터는 지도학습과 같은 **기계 학습** 접근 방식이 활용되기 시작하였으며, 시스템은 사람이 디자인하고 예제 데이터를 사용하여 모델을 훈련함. 
특히 연구자가 직접 이미지로부터 추출한 특성들 (Handcrafted features) 을 토대로 최적의 결정 경계를 도출함.
- **딥러닝**이 도입된 이후 컴퓨터가 데이터를 통해 최적의 특성을 스스로 학습하게 됨.
  - 주로 이미지에 대하여 여러개의 층을 가진 CNN (Lenet, Alexnet 등)이 널리 사용되었으며, Imagenet 챌린지와 같은 성과를 토대로 의료 영상 분석 분야에서도 딥러닝 기법으로 연구 방향을 점진적으로 전환하게 됨. 
- 주요 기술적 발전으로는 주성분 분석 (PCA), 이미지 패치 클러스터링, 사전 접근법 (Dictionary approaches), CNN (Convolutional neural network) 등의 기법이 있음.


![fig1](20240604-1.png){: width="800"}
_Number of Papers in medical AI field [^ref1]_

&nbsp;
&nbsp;
&nbsp;


## **Dataset**
- **<u>원천 의료 데이터</u>**
  - 질병 예후, 공공 건강 기록, 생체 데이터, 의료 영상
  - **임상 데이터**: 환자 기록, 생활 습관, 병력 등
  - **생체 의학 데이터**: 유전체 정보, 약물 연구 등
- **<u>특징</u>**
  - 높은 복잡성, 많은 변수, 계층 불균형, 낮은 데이터 품질 (수집 과정의 누락 데이터)
- **<u>형태</u>**
  - **구조화 데이터**: 인구 통계, 생활 습관, 병력
  - **비구조화 데이터**: 의사-환자 대화, 의사의 조사 기록

&nbsp;
&nbsp;
&nbsp;


-----------------
## **Machine Learning**
- 의료에서의 머신러닝은 데이터를 통한 올바른 예측을 위해 적절한 **특성 (Feature)**을 찾는 것이 중요함
- 과거에는 전문 지식이 있는 의료 전문가가 의료 이미지를 해석하거나, 특정 결과를 설명하기 위한 의미 있는 **설명자 (Descriptor)**를 개발하였음
- 최근에는 매번 방대한 양의 데이터가 생성되며 여러 머신러닝 기법을 활용할 수 있게 되었고, 이는 전문가 만큼 정확한 예측을 할 수 있게됨.
  - **암 연구**: 임상 이미징 배열에서 관련된 특성 (Feature)을 정확히 찾아내어 암 치료의 초기 지표를 탐지
  - **임상 응용**: 머신러닝 예측은 예후, 진단, 이미지 검사 및 치료 과정에서 임상 의사의 업무를 보완함
  - 예측, 분류, 클러스터링, 회귀 등 여러 가지 작업을 처리하는 다양한 알고리즘이 존재함.

### **서포트 벡터 머신 <sup>Support Vector Machine, SVM</sup>**
- SVM은 데이터 포인트를 고차원 공간으로 매핑하여 최대한 멀리 떨어져있는 두 클래스 사이의 **초평면 (Hyperspace)**을 사용해 구분함.
- 초평면은 두 클래스 사이의 거리를 최대화하고, **커널 함수**를 사용하여 최적의 초평면을 구분하고 비선형 문제를 해결할 수 있음.
- 변수에 비해 작은 데이터 셋에서 효과적이므로, 질병 예측 및 진단과 같은 의학 분야에서 널리 적용됨.
- **<u>사례</u>**
  - Dallora et al. (2017) | 경도 인지 장애가 있는 환자가 알츠하이머 병으로 발전할지 여부를 예측함
  - Taylor et al. (2018) | 요로 감염을 예측하기 위해 사용함
  - Leha et al. (2019) | 폐 고혈압 진단 작업에 사용됨
  - Meiring et al. (2018) | 중환자실 (ICU) 환자들의 생존율 예측에 사용됨.

- **<u>Formulation</u>**
![fig2](20240604-2.png){: width="800"}
_Formulations of the SVM [^ref6]_

&nbsp;
&nbsp;
&nbsp;


### **군집화 <sup>Clustering</sup>**
- 데이터 내의 유사성을 찾아 그룹화 하는 방법이며, 라벨이 없는 데이터를 유의미한 **군집 (Cluster)**으로 분류함.
- **Proximity measure**: 유클리드 거리와 같은 measurement를 사용하여 데이터 간의 유사성을 계산함.
- **K-means clustering**: 데이터 포인트를 초기 무작위 군집으로 할당한 후, 각 군집의 중심에서 데이터 포인트들을 거리에 따라 재할당함.
- 유전적 발현 데이터와 단백질 도메인 같은 데이터에 광범위하게 사용됨.
- **<u>사례</u>**
  - Khan et al. (2017) | 유방암 및 간 질환 진단에서 타 알고리즘에 비해 좋은 성능을 도출함.

- **<u>Formulation</u>**
![fig3](20240604-3.png){: width="800"}
_Formulations of clustering methods [^ref6]_

&nbsp;
&nbsp;
&nbsp;

### **결정 트리 <sup>Decision Tree, DT</sup>**
- 학습된 지식을 if-then 규칙으로 구성된 트리에 포함하며, 트리의 각 노드는 학습 변수를 나타냄.
- 데이터를 분류하기 위해 **정보 이득 (Information gain)** 및 **Entropy**를 계산함.
  - 높은 정보 이득은 더 많은 정보를 포함하고 있다는 것을 의미함.
- 정보를 반복적으로 분할하여 가장 적절한 특징을 찾아내고, 더 이상 분할할 수 없을 때까지 과정을 반복함.
- **<u>Formulation</u>**
![fig4](20240604-4.png){: width="800"}
_Formulations of the decision tree [^ref6]_

&nbsp;
&nbsp;
&nbsp;

### **랜덤 포레스트 <sup>Random Forest, RF</sup>**
- 여러 개의 결정 트리를 모아 놓은 모델로, 각 트리가 분류 작업을 수행하고 다수결 투표로 최종 결과를 결정함.
- 개별 트리의 예측치를 평균 내어 최종 예측을 산출함.
- 깊은 트리일수록 과적합이 발생하기 쉬운데, RF는 다수의 트리를 사용함으로써 이를 방지함.
- **<u>사례</u>**
  - Wang et al. (2015) | 상관관계가 없는 여러 결정 트리를 사용하여 미국에서 진단된 당뇨병 환자의 의료비를 예측함.
  - Goto et al. (2018) | 병원 응급실에서 천식 또는 만성 폐쇄성 폐질환 환자의 중환자 치료 또는 입원 여부를 예측함.
  - Zhu et al. (2018) | 각 결정 트리의 투표에 가중치르 곱하여 클래스의 신뢰도를 강화하고 RF의 클래스 불균형 문제를 해결함.
- **<u>Formulation</u>**
![fig5](20240604-5.png){: width="800"}
_Formulations of the random forests [^ref6]_

&nbsp;
&nbsp;
&nbsp;

### **K-nearest Neighbor <sup>KNN</sup>**
- 분류되지 않은 데이터 포인트를 가장 가까운 K개의 데이터 포인트의 다수결에 따라 분류함.
- 유클리드 거리, 맨해튼 거리, 해밍 거리 등의 거리 계산 방법들이 활용됨.
- 모든 특징에 동등하게 가중치를 주기 때문에 많은 속성값을 가지는 데이터에는 부적합함.
- **Fuzzy KNN**: 데이터 포인트를 클래스 대신 샘플 벡터에 할당하여 추후 추가적으로 검토를 통해 분류함.
- **<u>사례</u>**
  - Zhang et al. (2016) | 의료 데이터에 KNN을 적용하기 위한 준비의 중요성을 강조하며 수학적 증거를 제공함.
  - Chen et al. (2017) | 구조화된 환자 데이터 (인구통계학적 정보, 생활 습관, 혈액 검사 결과, 질병 등) 를 통해 뇌경색증의 위험을 예측함.
- **<u>Formulation</u>**
![fig6](20240604-6.png){: width="800"}
_Formulations of the KNN [^ref6]_

&nbsp;
&nbsp;
&nbsp;


### **나이브 베이즈 <sup>Naive Bayes</sup>**
- 주어진 특징 집합을 기반으로 클래스를 예측하며, **Bayes' Theorem**을 기초로 사용함.
- 각 특징의 확률을 다른 특징들로부터 얻은 정보를 기반으로 계산하여 독립성을 확보함.
- 불필요한 특징을 제거하여 분류 정확도를 향상시키고, 특히 텍스트 분류에 탁월한 결과를 보임
- **<u>사례</u>**
  - Kukar et al. (2016) | 대퇴골 경부 골절 회복의 예후를 예측할때 타 분류 모델에 비해 더 높은 성능을 보임을 확인함.
- **<u>Formulation</u>**
![fig7](20240604-7.png){: width="800"}
_Formulations of the Naive bayes [^ref6]_

&nbsp;
&nbsp;
&nbsp;



## **Deep Learning**
### **신경망 <sup>Neural Netrwork, NN</sup>**
- 딥러닝의 기초이며 뉴런 (Neuron)으로 구성되고, 각 뉴런은 활성화 함수 (Activation function), 가중치 (Weight), 편향 (Bias) 파라미터로 구성됨.
- **역전파 (Backpropagation)** 알고리즘을 통해 뉴런의 가중치를 미세 조정함.
  - 경사 하강법 (Gradient descent)를 사용하여 Cost function을 최소화 하는 방향으로 모델을 최적화함.

### **합성곱 신경망 <sup>Convolutional Neural Network, CNN</sup>**
- 합성곱 (Convolution) layer, pooling layer, fully-connected layer로 구성됨
- 주로 이미지 데이터를 입력으로 복잡한 Feature map을 학습하여 정보 영역 선택, 특징 추출, 분류 등의 task를 수행함.

&nbsp;
&nbsp;
&nbsp;

![fig8](20240604-8.png){: width="800"}
_Formulations of the deep learning models [^ref6]_


&nbsp;
&nbsp;
&nbsp;


------------
## **Learning**
### **Learning Problem**

![fig9](20240604-9.png){: width="800"}
_Types of learning [^ref5]_

&nbsp;
&nbsp;
&nbsp;

#### **지도학습 <sup>Supervised Learning</sup>**
- 모델이 입력 데이터와 목표 변수 사이의 관계를 학습하는 문제이며, 클래스 라벨을 예측하는 **Classification** 문제와 수치 라벨을 예측해야하는 **Regression** 문제로 나뉘어짐.

#### **비지도학습 <sup>Unsupervised Learning</sup>**
- 입력 데이터만 사용하고 출력/정답 변수 없이 학습하는 방식
- Clustering은 데이터를 그룹화하며 Density estimation은 데이터 분포를 요약함.
- 추가적으로 시각화를 통해 패턴, 추세, 관계를 시각적으로 강조할 수 있으며, Projection을 통해 고차원 데이터를 저차원 데이터로 변환할 수 있음.

#### **강화학습 <sup>Reinforcement Learning</sup>**
- 어떤 맥락에서 작업을 수행하기 위해 피드백을 학습해야 하는 일련의 과제
- 피드백을 통해 작업을 개선하지만 피드백이 지연될 수 있기 때문에 인과 관계를 파악하는 것이 어려움.
- **Deep reinforcement learning**: 기존 강화 학습에 신경망을 결합함.
- **Q-learning**: 에이전트가 상태와 행동의 Q-값 (품질)을 학습해 최적의 행동을 선택하는 방법.
- **Temporal difference leanring**: 시간 차이를 이용해 예측을 업데이트하는 방법.


&nbsp;
&nbsp;
&nbsp;


### **Hybrid Learning Problem**
#### **반지도학습 <sup>Semi-supervised Learning</sup>**
- 일부 데이터만 라벨링 되어있고, 나머지 다수의 데이터는 라벨이 없는 상태로 학습하는 방법
- 라벨 유무와 상관없이 모든 사용가능한 데이터를 효율적으로 활용하여 모델을 개선하는 것. 
  - 이는 완전히 라벨이 있는 데이터만 사용하는 전통적인 지도 학습과 차별화 됨.

#### **자기 지도학습 <sup>Self-supervised Learning</sup>**
- 라벨이 없는 데이터를 활용하여 문맥 예측 및 이미지 회전 예측과 같은 목표 과제를 자율적으로 계산함.
- Autoencoder를 통해 입력 데이터를 압축-복원하여 특징을 학습할 수 있고, GAN을 활용하면 합성 이미지를 생성할 수 있음.

#### **다중 인스턴스 학습<sup>Multi-instance Learning</sup>**
- 한 그룹의 데이터 전체가 특정 클래스의 데이터를 포함하거나 포함하지 않는 것으로 라벨링되며, 그룹 내 개별 데이터들은 라벨링되지 않음.
- 개별 데이터 포인트에 대한 정보가 아닌, 그룹 전체에 대한 정보를 중심으로 이미지가 특정 객체를 포함하는지 여부를 학습.

&nbsp;
&nbsp;
&nbsp;


### **Statistical Inference**
- **추론 (Inference)**은 모델을 통해 데이터로부터 예측/결론을 도출하는 과정임.
- 모델의 특정 알고리즘이 어떻게 작동하는지 또는 학습 문제를 어떻게 해결하는지를 설명하는 추론 패러다임이 아래와 같이 다양하게 존재함.


#### **귀납 학습 <sup>Inductive Learning</sup>**
- 결과를 평가하기 위해 증거를 사용하는 학습 방식이며, 구체적 상황에서 일반적 결과를 도출함.
- **귀납 추론 (Inductive Reasoning)**: 과거의 특정 사례를 통해 일반적인 규칙 (모델)을 학습함, 가용한 증거를 이용해 결과를 도출
  - 여러 번의 데이터 (사례)를 통해 일반적인 패턴이나 규칙 (모델)을 추출함.
- 구체적인 예시를 바탕으로 훈련 데이터셋에서 모델 또는 가설을 만들며, 이 모델은 알려지지 않은 데이터에 대해 예측을 수행하는데 사용됨.


#### **연역 학습 <sup>Deinductive Learning</sup>**
- 일반 개념을 사용해 구체적인 결과를 평가하는 방법.
- Induction과 반대의 개념임
  - **귀납, Induction**: 구체적인 사례에서 일반적인 결론을 도출하는 과정, Bottom-up
  - **연역, Deduction**: 일반적인 개념에서 구체적인 결론을 도출하는 과정, Top-down
- **연역 추론 (Deduction Reasoning)**: 모든 전제를 충족시키고 결과를 결정하려는 방식.
- 머신러닝에서 알고리즘을 사용하여 예측을 할 수 있으며, Induction을 사용하여 모델을 학습하기 전에 훈련 데이터셋에 맞도록 한다는 점에서 유사함.
  - 이러한 모델은 Deductive method로 사용됨.


#### **Transductive Learning**
- 통계 학습 이론에서 특정 예제를 예측하는 과정이며, 일반적인 규칙을 학습하는 Induction과 달리 Transduction은 구체적인 예제를 바탕으로 예측함.
- 적용 예시로는 함수의 특정 지점에서 값을 추정하는 모델을 들 수 있으며, 제한된 지식에서 최상의 결과를 도출하려는 경우 활용되는 것이 원칙임.
- 대표 알고리즘으로는 **k-NN (k-nearest neighbors)** 가 있으며 이는 예측이 필요할 때마다 직접 알고리즘을 사용하며, 훈련 데이터로 모델링하지 않음.

&nbsp;
&nbsp;
&nbsp;


### **Learning Techniques**
- 아래는 모델 학습과정에서 활용될 수 있는 기법들임.

#### **Multi-task Learning**
- 서로 다른 task의 세부 사항을 결합하여 일반화 성능을 향상하는 기술임.
- 하나의 작업 (Task)에 대한 라벨링된 데이터가 풍부할 때 유용하며, 해당 데이터를 라벨링된 데이터가 적은 또 다른 task에 공유가능함.
- 같은 입력 패턴이 여러 다른 출력 또는 지도학습 문제에 사용할 수 있음.
- 각 출력은 모델의 다른 부분에 의해 예측되며, 동일한 입력을 각 작업에 대해 일반화 가능함.

#### **Active Learning**
- 모델이 학습 과정 중 사용자에게 질문을 던져 불확실성을 해결하려는 방법론
- 수동적인 기존의 지도학습보다 더 효율적인 데이터를 사용하면서 동일하거나 더 나은 결과를 도출하는 것을 목표로 함.
- **Central Principle**: Active learning 알고리즘이 학습할 데이터를 선택하게 함으로써, 적은 학습 레이블로 더 높은 정확도를 달성할 수 있음.
- Active learner는 질문을 던지며, 이는 주로 라벨이 없는 정보 인스턴스로 나타나고 이를 Oracle (Human annotator)이 라벨을 붙임.
- 데이터가 적고 라벨링 비용이 많이 들 때 유용하며, Domain sampling을 조정하여 샘플 수를 줄이는 동시에 모델의 효율성을 높임.


#### **Online Learning**
- 머신러닝은 주로 오프라인에서 수행되며, 이는 일정한 데이터 배치에서 학습한다는 의미임.
  - Data stream이 있을 경우, 새로운 데이터 포인트가 도착할 때마다 추정치를 업데이트해야 하므로 온라인 학습이 필요함.
- 이는 시간이 지남에 따라 데이터가 빠르게 변화할 때 유용하며, 데이터셋이 점진적으로 증가하는 경우에도 유용함.
- 일반적으로 온라인 학습은 모델 성능이 모든 지식을 일괄적으로 사용 가능 했을 때와 얼마나 잘 수행되는지를 비교하여 불일치를 제거하는 것을 목표로 함.
- **Online Gradient Descent**: 주로 Stochastic gradient descent를 사용하며 이는 일반화 오류를 최소화한다는 것이 명확히 드러남.


#### **Transfer Learning**
- 기존 Task를 해결하면서 학습한 모델을 다른 새 Task 문제 해결에 사용함.
- 주된 Task와 유사한 과정이 있을 때 효과적이며, 데이터가 많이 필요한 관련 작업에서 유용함.
- Multi-task learning과 달리 Transfer learning은 작업을 순차적으로 학습하며, Multi-task learning은 하나의 모델이 동시에 여러 작업에서 좋은 성능을 내도록 학습함.
- 큰 작업에서 학습한 패턴 추출 기능이 또 다른 작업에 도움이 됨.



#### **Ensemble Learning**
- 두 개 이상의 모델을 결합하여 예측 성능을 향상시키는 방법이며, 개별 모델보다 더 나은 성능을 달성하는 것을 목표로 함.
- 불확실성을 줄이고 예측 능력을 향상시키기 위한 중요한 방법임
- **<u>주요 방법</u>**
  - **Bootstrap**: 샘플링을 통해 여러 데이터셋을 만든 후 개별 모델을 학습
  - **Weighted Average**: 각각의 모델 예측에 가중치를 부여하여 최종 예측 도출
  - **Stacking (Stacked Speculation)**: 여러 모델의 예측 결과를 다시 입력값으로 사용하는 메타 모델을 학습
- **<u>계산 방법</u>** 
  - **Bagging**: 다수의 데이터 서브셋을 이용해 다수의 모델을 학습, 개별 모델의 예측을 평균 계산하여 최종 예측 도출
  - **Boosting**: 예측 오류를 줄이기 위해 순차적으로 모델을 학습, 각 모델이 이전 모델의 오류를 보정함




&nbsp;
&nbsp;
&nbsp;


------------
## **Task Definition in Medical Imaging**
### **분류 <sup>Classification</sup>**
- **Image Classification**는 주로 질병/암 진단과 같은 이진 분류에 대한 것이며, 
의료 데이터의 경우 보통 일반 컴퓨터 비전 데이터 셋의 양보다 적기 때문에 (~수백/수천), Transfer learning 기법을 활용해 작은 데이터 셋 문제를 해결함.
- Classification Task에서는 주로 **특징 추출 (Feature Engineering)** 및 **미세 조정 (Fine-tuning)** 이 많이 활용되며 최근에는 특히 CNN을 기반으로한 여러 연구들이 진행됨:
  - 미세 조정된 Google Inception v3, 3D CNN, Graph-based CNN.

- **Object/Lesion Classification**
  - 객체나 병변을 분류하는 Task이며 CT의 결절 분류와 같이 소규모 영역에 대한 분류를 예로 들 수 있음.
  - 정확한 분류를 위해 Local 정보와 전체적인 Context 정보를 필요로 함
  - 예로 Multi-stream architecture, Multi-scale 이미지를 결합한 특징 벡터, CNN-RNN, 3D Information integration, End-to-end training, RBMs, SAEs, CSAEs, Multiple Instance Learning (MIL).
- **Exam Classification**
  - 진단 검사 사진을 질병 유무 또는 정상/비정상으로 분류
  - 사전 학습된 CNN이 주로 활용됨
- **Essential Terminology**
  - Binary classification, multi-class classification, multi-label classification



&nbsp;
&nbsp;
&nbsp;


### **Detection**
- Manual Detection은 환자에게 심각한 결과를 초래할 수 있는 여러 문제를 겪고 있어, 이러한 문제를 최소화 하기 위해 자동 감지가 필요함.
- **Detection Task**는 주로 장기, 병변 및 랜드마크 위치를 지정해주는데 쓰이며, 특히 3D 데이터 파싱을 위해 다양한 접근 방식이 제안됨:
  - 3D 영역을 2D 직교 평면의 조합으로 처리함, 3개의 독립적인 2D MRI 슬라이스로 원거리 대퇴 표면의 랜드마크 식별, 2D CT 볼륨 파싱 후 3D 경계 상자를 식별하여 심장, 대동맥 궁, 하행 대동맥 근처의 ROI 식별.
- 객체 또는 병변 탐지는 의료 진단 영역의 주요 부분이며, 임상의에게 많은 시간을 소모하게 함. 이를 AI를 통해 이미지 내 작은 병변의 위치를 지정하고 식별하는 것으로 해결.
- 대부분의 탐지 시스템은 CNN을 사용하여 픽셀을 분류 후 후처리를 수행하며, 문맥적 또는 3D 정보를 포함하기에 Multi-stream CNN을 사용함.
- 또한, 학습 데이터 생성 부담을 줄이기 위해 Weakly supervised learning을 활용할 수 있음.

#### **False Positive Detection**
- **False Positive (FP)**는 정상이지만 비정상으로 간주되는 픽셀로, CAD 시스템의 민감도를 줄이면서 잘못된 의료적 개입을 야기함.
- 기존에는 FP를 감소시키기위해 통계 분석을 기반으로하나 후처리 필터를 활용하거나 수동적인 방법을 사용하였지만, 딥러닝을 통해 이를 개선함.
  - CNN으로 다양한 뷰와 스케일의 2D 슬라이스를 학습하여 FP와 TP를 구분함.

&nbsp;
&nbsp;
&nbsp;


### **Segmentation**
- **Segmentation Task**는 장기 및 하위 구조를 분할하여 부피 및 형태와 관련된 임상 파라미터의 정량적 분석을 가능하게 함.
  - 주요 접근법은 객체의 윤곽 또는 내부를 구성하는 폭셀(voxel)을 식별하는 것이고 일반적으로 U-net, V-net, RNN 등이 활용될 수 있음.
- 3D 이미지를 직접적으로 처리하여 2D 이미지에 비해 더 정확한 Segmentation을 가능케 하는 3D CNN-based Segmentation 기법이 있음.
  - **Efficient dense training scheme**: 전체 이미지를 사용하는 대신 인접한 이미지 패치를 사용하여 학습
  - Deeper and more discriminative 3D-CNNs
  - **Dual pathway architecture**: 여러 스케일에서 병렬로 처리.
- Pereira et al. | 작은 CNN Kernel을 활용하여 글리오마 (가장 공격적인 뇌종양)를 분할하였음.
- Avendi et al. | 딥러닝 구조를 결합한 심장 좌심실 분할을 통해 임상 평가에 중요한 지표를 제공.


&nbsp;
&nbsp;
&nbsp;


### **Registration**
- **정합 (Registration)**은 한 의료영상을 다른 의료영상에 공간적으로 맞추는 과정이며, 일반적으로 두 영상을 비교하여 유사도를 계산하고 이를 최적화하여 정합을 수행함.
- 여러 이미지 데이터를 하나의 일치된 좌표 시스템으로 변환하고 중요한 의료적 함의가 있는 일치된 이미징 내용을 생성
  - 동일한 항목 (MRI와 CT스캔) 또는 다른 시간 및 위치에서 촬영된 두 스캔이미지 정렬을 위한 좌표 찾기에 활용 가능
- 주로 **Deep regression networks**와 **Deep learning networks**를 사용함 
  - Wu et al. (2013) | 독립적 공간 분석과 CNN을 결합하여 HAMMER 정합 알고리즘을 보강.
  - Simonovsky et al. (2016) | 서로 다른 모달리티의 패치들 사이의 유사도 비용 추정.
  - Cheng et al. (2015) | CT와 MRI 이미지 패치의 유사도 평가를 위한 오토인코더 사용.
- 변환 파라미터 예측을 위해 딥러닝을 활용하여 입력 이미지로부터 바로 변환 파라미터를 예측 가능함.
  - Miao et al. (2016) | CNN을 활용하여 3D 모델과 2D X-ray 정합을 수행하여 수술 중 임플란트 객체의 위치와 자세 평가.
  - Yang et al. (2016d) | U-net과 같은 아키텍처를 사용하여 LDDMM 기법의 현재와 이전 뇌 MRI 정합.
- ANT나 Simple ITK와 같은 툴을 활용할 수 있음. 


&nbsp;
&nbsp;
&nbsp;


### **Localization**
- 2D 및 3D 공간, 그리고 시간 (4D)에서 장기나 다른 기관의 **위치를 인식**함.
- ConvNet을 사용하여 관심 부위의 해부학적 구조를 3D 의료 이미지에서 자동으로 Localization함.
- 수동 및 자동으로 지정된 중심점과 Bounding box의 거리를 계산하여 분석
- 이미지를 얻는 과정의 차이, 구조적 차이, 환자 간의 병리 차이 등 때문에 딥러닝 네트워크가 변동에 민감할 수 있음.




&nbsp;
&nbsp;
&nbsp;


### **Content-based image retrieval**
- **Content-based image retrieval (CBIR)**은 방대한 데이터베이스에서 지식 발견을 위한 기술임.
- 유사한 사례 기록 식별, 희귀 장애 이해, 환자 치료 개선을 위해 수행됨.
- 주로  픽셀 수준 정보에서 효과적인 특징 표현 추출하거나, 이를 의미 있는 개념과 연관시키기 위해 활용됨.
- 현재 접근법은 (사전 학습된) CNN을 사용해 의료 영상에서 Feature descriptor 추출하는 것임
  - Anavi et al. (2016), Liu et al. (2016b) | X-ray 이미지 데이터베이스를 대상으로 5-레이어 CNN 사용, Fully-connected layers에서 특징 추출
  - Anavi et al. (2016) | 마지막 레이어와 사전 학습된 네트워크 사용  특징을 one-vs-all SVM 분류기에 피드해 거리 메트릭(distance metric) 얻음.
  - Shah et al. (2016) | CNN 특징 기술자와 해싱-포레스트 사용, 1000개의 특징을 중첩 패치로부터 추출


&nbsp;
&nbsp;
&nbsp;


### **Image Generation and Enhancement**
- **이미지 생성 및 향상 기법**은 장애 요소 제거, 이미지 정규화, 이미지 품질 향상, 데이터 완성, 패턴 발견 등의 사례에 활용될 수 있음.
- 주로 2D 또는 3D CNN를 사용하며 분류 네트워크에서 사용되는 Pooling layer가 없고 입력 이미지와 원하는 출력이 모두 포함된 데이터셋으로 학습함.
- 손실함수는 생성된 이미지와 원하는 출력 간의 차이로 정의됨.
  - Yang et al. (2016c) | 일반 X-ray와 bone-suppressed X-ray 생성
  - Bahrami et al. (2016) | 3T와 7T 뇌 MRI 생성
  - Li et al. (2014) | MRI로부터 PET 생성
  - Nie et al. (2016a) | MRI로부터 CT 생성
- 다중 저해상도 입력에서 고해상도 이미지 생성하는 목적으로도 활용됨.
  - Oktay et al. (2016) | 다중 저해상도 입력 MRI에서 고해상도 심장 MRI 생성
  - Golkov et al. (2016) | 제한된 데이터로부터 고급 MRI 확산 매개 변수 추론 가능
- 이미지 향상에도 응용될 수 있음.
  - Janowczyk et al. (2016a) | H&E 염색된 조직 병리학 이미지 정규화
  - Benou et al. (2016) | DCE-MRI 시리즈에서 denoising 수행

&nbsp;
&nbsp;
&nbsp;

### **Text Report**
- 주로 리포트를 텍스트 라벨로 사용하여 **텍스트 설명과 이미지를 같이 학습**하여, 테스트 시에는 의미있는 클래스 라벨 예측을 가능케 함.
  - Schlegl et al. (2015) | 리포트를 활용한 이미지 분류 정확도 개선
  - Kisilev et al. (2016) | 유방 병변에 대해 BI-RADS 기술어 예측.
  - Shin et al. (2015, 2016a) | PACS 시스템에서 추출한 대규모 데이터 세트의 방사선 보고서 및 이미지 간의 의미적 상호작용을 추출.
- 잠재 디리클레 할당(LDA) 사용
  - Wang et al. (2016e) | LDA 사용
  - Shin et al. (2016a) | CNN을 이용하여 이미지의 레이블을 한 번에 하나씩 생성하고, 이를 사용하여 RNN을 훈련시켜 MeSH 키워드 시퀀스 생성.

&nbsp;
&nbsp;
&nbsp;


---------
## **Application Areas**
![fig10](20240604-10.png){: width="800"}
_Medical Image Analysis [^ref5]_

&nbsp;
&nbsp;
&nbsp;

### **Chest**
- **흉부 영상 분석**의 주요 응용 분야는 아래와 같음.
  - **결절 탐지 및 분류**: 결절의 탐지, 특성화, 및 분류가 흉부 영상 분석에서 가장 일반적으로 다루어지는 응용 분야임
  - **기존 시스템에 추가된 딥러닝 특징**: 많은 연구가 딥러닝에서 도출된 특징을 기존의 특징 집합에 추가하거나 비교함
  - **흉부 X-선**: 여러 그룹이 단일 시스템으로 여러 질병을 탐지함
  - **CT 스캔**: 간질성 폐 질환을 나타내는 질감 패턴의 탐지도 일반적인 연구 주제임
- 텍스트와 이미지 분석 결합 연구들이 대규모 이미지와 텍스트 보고서를 사용하여 이미지 분석을 위한 CNN과 텍스트 분석을 위한 RNN을 결합한 시스템을 훈련함.
- 폐 질환은 100개 이상의 만성 질환으로 구성되며, 폐 조직의 염증으로 특징지을 수 있음.
  - 전통적인 진단은 환자 인터뷰, 신체 검사, X선 또는 CT 스캔으로 이루어지며, 이는 잘못된 진단을 야기할 수 있음.
  - CT 슬라이드를 통해 패치 이미지를 생성하고 VGG 및 Alexnet을 활용하여 약 86% 정도의 폐 질환 분류 정확도 성능을 달성함.


![fig11](20240604-11.png){: width="600"}
_Generating image patches from one CT slide [^ref2], [^ref3]_



&nbsp;
&nbsp;
&nbsp;


### **Brain**
- 뇌 영상 분석에서 딥러닝을 활용하여 **알츠하이머병 (Alzheimer's Disease, AD) 분류**, 뇌 조직 및 해부학적 구조의 세분화, 병변 감지 등을 수행하고 있음.
  - 뇌 이미지에서 추출된 특성을 분류하여 해부학적 뇌 구조 변화를 모니터링함 (뇌실 크기, 모양, 조직 두께, 뇌 부피)
  - 3D-CNN을 활용하여 AD 바이오마커를 인식하고 학습한 일반화 특징을 통해 AD를 예측하고 해부학적 모양 변화를 감지함
  - Multi-modal RBM을 통해 MRI와 PET 이미지의 High-level hidden feature을 찾고, 3D 패치에서 특징을 추출하여 다층 RBM으로 분류함.
  - AD 분류에서 RBM이 CNN보다 뛰어난 성능을 보여준다는 연구 결과가 있음.
- **로컬 패치**를 활용하는 학습 기법은 로컬 패치에서 Representation으로, Representation에서 라벨로의 매핑을 학습함
  - Ghafoorian et al.(2016b) | 로컬 패치가 해부학적 맥락 정보를 놓칠 수 있기에, 패치 측면에서 샘플링 비율을 점진적으로 낮춰 더 넓은 맥락을 포괄함.
- 또 다른 접근법은 Multi-scale 분석 및 Fully-connected layer에서의 표현 융합
- 뇌 종양을 분할하기위해 딥러닝 기반 Segmentation 모델이 활용됨.
  - 종양 패치 이미지를 통해 Two-pathway 또는 Cascaded 구조 기반 CNN을 학습하여 비종양, 괴사, 부종, 증강 종양 등으로 세분화함.

![fig12](20240604-12.png){: width="800"}
_Two pathways and cascaded architectures [^ref2], [^ref4]_

- 대부분의 방법은 2D에서 3D 볼륨을 슬라이스로 분석하는 것이며, 이는 계산 요구 사항 감소 또는 데이터 셋 내 평면 해상도에 비해 두꺼운 슬라이스 때문
  - 최근에는 3D 네트워크를 사용하는 연구도 증가하는 실정임.
- 대다수가 뇌 MRI 영상을 중점적으로 다루고 있으며, 다른 뇌 영상 Modality (CT 및 초음파) 또한 딥러닝 분석에서 혜택을 받을 것으로 예상


&nbsp;
&nbsp;
&nbsp;



## **Pathology and Microscopy**
- **<u>주요 도전과제</u>**
  - 핵 탐지, 세분화 및 분류
  - 대형 기관 세분화
  - 병변 또는 전체 슬라이드 이미지 수준에서 질병 탐지 및 분류
  - 색상 정규화

- **<u>사례: 조직병리학 (Histopathology)</u>**
  - Janowczyk et al. (2016a) | H&E 염색 이미지를 대상으로한 Stain normalization를 위한 Deep sparse AE 기법을 소개.
  - Sethi et al. (2016) | CNN 기반 조직 분류에 색상 정규화의 중요성을 강조.
- 유방암 병리 이미지 분석에 DeepMitosis라고 불리는 50개 층의 CNN을 활용하여 TP/FP에 대한 확률 점수를 추정한 연구가 진행됨.



&nbsp;
&nbsp;
&nbsp;

### **Breast**
- 대부분의 Breast Imaging 기술이 2차원이므로 기존의 이미지 기술을 쉽게 적용할 수 있음.
- 주로 종양 같은 병변의 검출 및 분류, 미세 석회화의 검출 및 분류, 유방암 위험 점수 계산 등이 있음.
  - **Modality**: 유방조영술이 가장 일반적으로 사용됨.
    - Tomosynthesis, 초음파(US), 전단파 탄성 측정 기법 등의 연구는 아직 적음. 
    - 유방 MRI에 대한 연구는 거의 없음.
- 유방암 대규모 공개 디지털 데이터베이스 부재로 인해, 오래된 필름 데이터셋이 많이 사용됨.
- 적은 데이터셋 문제는 Semi-supervised learning, Self-supervised learning, Weakly supervised learning 등으로 해결하려고 함.
- 유방 촬영술 이미지 분석 (Mammogram Image Analysis)은 유방암 조기 발견을 위한 가장 안전한 방법이며, CAD 시스템에서 CNN을 활용한 유방 병변 감지, 분할, 분류 등의 자동화 과정을 거침.

&nbsp;
&nbsp;
&nbsp;


### **Cardiac**
- **심장 이미지 분석 (Cardiac image analysis)** 에는 주로 MRI가 많이 쓰이며, **좌심실 분할 (Left ventricle segmentation)** 같은 작업이 주로 포함됨.
- 아래의 방법들이 적용될 수 있음.
  - Segmentation, Tracking, Slice classification, Image quality assessment, Super-resolution
  - Automated calcium scoring (자동 칼슘 점수화), Coronary centerline tracking (관상 동맥 중심선 추적)

- 대부분의 논문에서 단순한 2D CNN을 사용하여 3D 또는 4D 데이터를 슬라이스 별로 분석함.
- DBN (Deep Belief Networks)이 많이 활용되었으며 주로 특징 추출에 사용되고, 복합 분할 프레임워크에 통합됨.
- Poudel et al. (2016) | U-net 아키텍처 내에 순환 연결 (Recurrent connection)을 도입하여 좌심실을 슬라이스 별로 분할하고, 이전 슬라이스의 정보를 기억하여 다음 슬라이스 분할에 활용
- Kong et al. (2016) | 표준 2D CNN과 LSTM을 결합하여 Temporal regression를 수행하고, 특정 프레임과 심장 시퀀스 식별


&nbsp;
&nbsp;
&nbsp;


### **Musculoskeletal**
- **Segmentation**: 근골격계 (Musculoskeletal) 이미지를 분석하여 뼈, 관절 및 관련 연조직의 이상 여부를 분할함.
- **Detection**: 관절과 뼈의 세부 구조, 그리고 이상 유무를 식별함.
- MRI, CT, X-ray, 초음파 등의 다양한 이미징 기법을 활용함.
  - Jamaludin et al. (2016) | 12,000개의 척추 디스크 데이터를 사용해 시스템을 학습시킴.


&nbsp;
&nbsp;
&nbsp;


### **Other**
- **피부병 진단 (Dermoscopic Image Analysis)**
  - 기존에 어려운 영역으로 알려졌던 피부암 진단에 대한 딥러닝의 급격한 성능 향상.
  - Esteva et al. (2017) | Google’s Inception v3 모델을 이용하여, 보통 사진과 dermoscopic 이미지로 훈련, 매우 큰 데이터셋 사용.
    - 제안된 시스템이 30명의 피부과 전문의와 동일한 수준의 성능을 보임.
- 초창기 연구는 pre-trained CNN을 특징 추출기로 사용했으나, 최근 연구는 완전한 end-to-end 교육된 CNN 사용이 증가.
  - 이러한 CNN은 기존의 수작업 특징 기반 머신러닝 방법을 대체하고, 의료영상 해석에서의 표준적인 접근 방식으로 자리잡음.

&nbsp;
&nbsp;
&nbsp;


## **Discussion**
- **기술적 확산**
  - 딥러닝이 의료 영상 분석의 모든 측면에 빠르게 확산
  - 242개 논문이 2016년 또는 2017년 초에 출판
  - 다양한 딥러닝 아키텍처가 사용됨.

- **초기 연구**
  - 초기 연구에서는 사전 학습된 CNN을 특징 추출기로 사용하였으며, 이러한 네트워크는 다운로드 후 바로 의료 이미지에 적용 가능
  - 기존 손수 설계된 Feature 기반 기계 학습 시스템을 쉽게 확장 가능

- **최신 연구 동향**
  - End-to-end로 학습된 CNN이 의료 영상 해석에서 선호되는 접근법이 됨.
  - 전통적인 Hand-crafted 기계 학습 방법을 대체하였으며, 이는 현재 표준적인 관행임.

- **성공적인 딥러닝 방법의 핵심 요소**
  - 특정 작업과 응용 분야에 대한 완벽한 딥러닝 방법과 아키텍처를 도출하기는 어려움.
  - CNN이 대부분의 의료 영상 분석 대회에서 최고 성능 제공하지만 정확한 아키텍처가 좋은 솔루션을 제공하는 가장 중요한 결정 요인은 아님.

- **전문 지식의 중요성**
  - 과제 해결에 대한 전문 지식을 활용하는 것이 단순히 CNN에 레이어를 추가하는 것보다 더 많은 이점을 제공
  - 성공적인 연구 그룹은 딥러닝 네트워크 외부의 새로운 데이터 전처리 또는 증강 기술에서 차별화를 보임
  - **사례**: CAMELYON16 챌린지에서 염색 정규화 전처리 단계 추가로 성능 크게 향상 (AUC 0.92에서 0.99로)
    - 데이터 증강 전략이 네트워크의 견고성을 향상시키는 데 필수적임.

- **특화된 모델 구조 디자인**
  - 고유한 작업 특정 특성을 통합하는 아키텍처 설계가 단순한 CNN보다 더 나은 결과를 제공
  - 여러 뷰를 활용한 네트워크와 다중 스케일 네트워크 등이 이러한 예에 포함

- **미래 가능성**
  - **비지도 학습**: 매우 큰 양의 라벨이 없는 데이터를 사용 가능.
  - Variational Auto Encoder (VAE)
  - 생성적 적대 신경망 (GANs)
  - **설명 가능성 (Explainable AI, XAI)**: '블랙 박스'로 설명되지 않는 딥러닝 문제를 해결하기 위한 다양한 기법들
    - 역전파, Captioning 작업, Bayesian 통계와의 결합을 통해 신뢰할 수 없는 예측을 할 때 이를 파악하는 능력

- **주요 연구 성과**
  - Esteva et al. (2017), Gulshan et al. (2016) | 딥러닝을 활용한 이미지 분류에서 의학 전문가를 능가하는 경우도 있음을 보여줌.
    - 2D RGB 이미지 분류에 초점을 맞췄으며, 이는 기존 컴퓨터 비전 (예: ImageNet) 과 유사한 작업임.
    - ResNet 및 VGG-Net 과 같은 잘 연구된 네트워크 아키텍처를 사용할 수 있어 성능이 우수함.


&nbsp;
&nbsp;
&nbsp;


## **Challenges**
- **대규모 데이터셋 부족**: 훈련 데이터셋이 부족
  - **PACS 시스템**: 서양 병원의 방사선과에서 사용된 PACS 시스템은 수백만 장의 이미지를 저장하고 있어 데이터 양의 상당한 축적이 가능
- **Labeling (Annotation) 어려움**: PACS 시스템은 방사선 의사가 작성한 자유 텍스트 보고서를 저장하고 있으나 이를 자동화된 방식으로 정확한 레이블로 전환하는 것은 매우 복잡
  - 전문가가 직접 이미지를 라벨링하는 경우, 비용과 시간이 많이 소모
  - 단층 촬영 이미지 분할을 위해 3D 슬라이스별 주석을 달아야 하는 경우가 많으며, 이는 매우 많은 시간을 필요로 함.
- **라벨 노이즈 문제**: 라벨링된 데이터가 존재한다 해도, 도메인 전문가 간에 의견이 일치하지 않는 경우가 있음.
  - LIDC-IDRI 데이터셋에서는 네 명의 방사선 의사가 독립적으로 단층 촬영 이미지를 주석 달았으나 완전히 일치하지 않는 단층 촬영 개수가 세 배 더 많았음.
- **이진 분류의 한계**: 의료이미지에서는 정상 대 비정상, 객체 대 배경과 같은 이진 분류가 자주 사용됨.
  - 그러나 이는 종종 비정상의 여러 하위 범주가 매우 드문 경우 이진 예측이 실패하게 됨.
- **클래스 불균형**: 특정 질병 이미지가 상대적으로 적은 경우, 클래스 불균형 문제가 발생
  - 유방암 검사 프로그램은 대부분 정상 유방 촬영 이미지를 가지고 있어, 이를 처리하는 데 어려움.
- **추가 정보 통합**: 의료진은 환자의 역사, 나이, 인구 통계 정보 등 이미지를 넘어 많은 데이터를 활용함.
  - 이를 딥 러닝 네트워크에 통합하는 연구가 있지만, 아직 충분한 성능을 보이지 못함.

- **의료 영상 분석의 도전 과제**
  - 3D 흑백이나 다중 채널 이미지와 같은 대부분의 의료 영상에서는 사전 훈련된 네트워크나 아키텍처가 존재하지 않아 새롭게 개발된 네트워크가 필요함.
  - 의료 영상에는 비등방성 복셀 크기, 다양한 채널 간의 작은 등록 오류, 다양한 강도 범위와 같은 고유한 문제가 있음.
  - 분류 문제로 제기할 수 없는 작업들도 종종 존재하며, 이 경우 비딥러닝 방법 (Counting, segmentation, regression) 으로 후처리가 필요함.





&nbsp;
&nbsp;
&nbsp;

#### **References**

[^ref1]: [Litjens, Geert, et al. "A survey on deep learning in medical image analysis." Medical image analysis 42 (2017): 60-88.](https://arxiv.org/pdf/1702.05747)
[^ref2]: [Abdou, Mohamed A. "Literature review: Efficient deep neural networks techniques for medical image analysis." Neural Computing and Applications 34.8 (2022): 5791-5812.](https://www.researchgate.net/profile/Mohamed-Abdou/publication/358274478_Literature_review_efficient_deep_neural_networks_techniques_for_medical_image_analysis/links/64d513a0d3e680065aac89e1/Literature-review-efficient-deep-neural-networks-techniques-for-medical-image-analysis.pdf)
[^ref3]: [Anthimopoulos, Marios, et al. "Lung pattern classification for interstitial lung diseases using a deep convolutional neural network." IEEE transactions on medical imaging 35.5 (2016): 1207-1216.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7422082)
[^ref4]: [Havaei, Mohammad, et al. "Brain tumor segmentation with deep neural networks." Medical image analysis 35 (2017): 18-31](https://arxiv.org/pdf/1505.03540) 
[^ref5]: [uganyadevi, S., V. Seethalakshmi, and Krishnasamy Balasamy. "A review on deep learning in medical image analysis." International Journal of Multimedia Information Retrieval 11.1 (2022): 19-38.](https://link.springer.com/article/10.1007/S13735-021-00218-1)
[^ref6]: [Garg, Arunim, and Vijay Mago. "Role of machine learning in medical research: A survey." Computer science review 40 (2021): 100370.](https://www.sciencedirect.com/science/article/abs/pii/S1574013721000101)
