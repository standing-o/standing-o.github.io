---
title: "비지도학습과 딥러닝 | Unsupervised learning"
date: 2022-07-17 17:00:00 +/-TTTT
categories: [AI Theory, Machine Learning]
tags: [lg-aimers, unsupervised-learning, deep-learning, neural-network, clustering, feature-engineering]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: 비지도학습 이해, 비지도학습 개념, 머신러닝 비지도학습, 인공지능 비지도학습, 딥러닝 비지도학습, 비지도학습 알고리즘, 비지도학습이란, 비지도학습 예시
---



----------------

> 인공지능 비지도 학습(Unsupervised learning)의 간단한 개념과 전반적인 견해를 이야기합니다.
{: .prompt-info }

전통적인 머신러닝과 딥러닝에서의 비지도 학습 방법을 소개하면서 K-means, 계층적 클러스터링, 밀도 추정 기법을 설명합니다.

딥러닝에서의 특성 공학(Feature Engineering)과 표현 학습의 차이, 고차원 데이터 개념을 설명하면서 딥러닝의 표현은 설명하기 어려운 경우가 많다는 점을 이야기합니다.

&nbsp;
&nbsp;
&nbsp;

## **전통적인 머신러닝 비지도학습**
- 전통적인 머신러닝의 비지도 학습은 주로 낮은 차원의 데이터를 다루고, 간단한 개념의 알고리즘으로 구성됩니다.
- **<u>종류</u>**
  - K-평균 군집화 (K-means Clustering)
  - 계층적 군집화 (Hierarchical Clustering)
  - 밀도 추정 (Density Estimation)
  - 주성분 분석 (Principal Component Analysis, PCA)


&nbsp;
&nbsp;
&nbsp;

## **딥러닝 비지도학습**

- **특성 공학(Feature Engineering)**
  - 인간에 의해 이루어집니다.
  - 데이터에 대한 도메인 지식(Domain Knowledge)과 창의성이 요구됩니다.
  - Brainstorming...

- **표현 학습(Representation Learning)**
  - 기계에 의해 이루어집니다.
  - 딥러닝 지식과 코딩 기술이 요구됩니다.
  - Trial and Error...


### **딥러닝에서의 표현 <sup>Representation in Deep Learning</sup>**
- 딥러닝에서의 표현은 제약이 적습니다.
  - 간단한 SGD로 유용한 네트워크를 찾을 수 있습니다.
  - 표현 특성은 필요에 따라 조정할 수 있습니다.
  - 학습된 표현은 이해하기 어렵습니다.

- 분리된(Disentangled) 표현
  - 정렬됨(Aligned), 독립적(Independent), 부분공간(Subspaces)
  - 제약이 많지 않아서 가능합니다.


### **각도 정보 <sup>Angle Information</sup>**

- $0 \sim 2\pi$
  - 알고리즘은...
    - $0$과 $2\pi$는 다르다고 생각하면서, $0$과 $1.9\pi$는 멀다고 생각합니다.
  - $(x_1, x_2) = (\cos \theta, \sin \theta)$
    - $0$과 $2\pi$는 같으며, $0$과 $1.9\pi$은 가깝습니다.


### **공간 정보 <sup>Spatial Information</sup>**

- **목표** ㅣ 수학적 객체로 표현하기



### **인간의 표현 문제**
- 인간은 이해할 수 있으며, 인간은 목표를 가지고 설계할 수 있습니다.

> **딥러닝에서 좋은 표현이란? 유용하지만 관련없는 정보**


### **잘 정의된 작업 <sup>A Well Defined Task</sup>**

- 일반적으로 관심 있는 속성만 $$y$$로 고려됩니다.
  - Imagenet - 클래스 라벨
  - $$y$$는 인간이 선택한 라벨로 단순히 정의되므로, Well Defined Task 라고 가정할 수 있습니다.

- 좋은 표현이라는 모호한 개념 ➔ **지도학습**
  - $$y$$가 잘 정의되어도 $$h_1$$과 $$h_2$$에서 무엇을 원할까?
  - "표현 학습이 성공적이다"고 말할 수 있을까?
  - "유용한 정보가 잘 정리되었다"고만 말할 수 있을 것입니다.
  - 일반적인 목적에서 좋은 표현이란 무엇일까?
  
  

### **정보 병목 <sup>Information Bottleneck</sup>**

- 잘 정의된 지도 학습 작업에서는 $$h_1$$과 $$h_2$$가 무엇을 만족해야 할까?
- 좋은 표현이라는 모호한 개념 ➔ **지도학습**
  - 일반적인 목적에서 좋은 표현이란 무엇일까?
  - 일반적인 목적은 후속 작업들의 목록으로 정의될 수 있습니다.
  - 관심 있는 작업의 성능이 좋다면 다시 돌아오게 될 것입니다.


### **표현 <sup>Representation</sup>**

- **우리가 원하는 것** ㅣ 표현에 대한 공식적인 정의와 평가 지표
- **현실** ㅣ 정의가 없고 작업 의존적인 평가 방법들

&nbsp;
&nbsp;
&nbsp;

## **비지도 표현 학습 <sup>Unsupervised Representation Learning</sup>**
- **비지도 학습 성능 ~ 지도 학습 성능**
  - 인스턴스 분별, 대조 손실, 공격적 증강 덕분..
- 지도 학습과 마찬가지로..
  - 성능 지표가 불분명할 수 있습니다.
  - 대체 손실 설계는 훌륭합니다. (일부는 원칙에 기반, 일부는 경험적)
  - 학습 기술 개발은 계속되고 있습니다. (하지만 증강 방법이 지배적)
- 자연어 처리(NLP)
  - 마스크된 언어 모델링
  - 그 다음은?
- 비지도 표현 학습
  - 아직 갈 길이..


&nbsp;
&nbsp;
&nbsp;


------------------
## Reference
> 본 포스팅은 LG Aimers 프로그램에서 학습한 내용을 기반으로 작성되었습니다. (전체 내용 X)
{: .prompt-warning }

1. LG Aimers AI Essential Course Module 3. 비지도학습, 서울대학교 이원종 교수

