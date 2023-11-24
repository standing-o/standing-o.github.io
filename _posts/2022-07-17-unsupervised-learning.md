---
title: "비지도학습과 딥러닝 | Unsupervised learning"
date: 2022-07-17 17:00:00 +/-TTTT
categories: [AI Theory, Machine Learning]
tags: [lg-aimers, unsupervised-learning]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: 비지도학습 | Unsupervised learning
---



----------------

> 인공지능의 비지도 학습 개념과 종류를 설명합니다.
{: .prompt-info }

전통적인 머신러닝과 딥러닝에서의 특징 및 차이를 소개하면서 K-means, 계층적 클러스터링,밀도 추정과 같은 방법을 정리합니다.

딥러닝에서는 특성 공학과 표현 학습의 차이, 고차원 데이터 개념을 설명하면서 딥러닝의 표현은 설명하기 어려운 경우가 많다는 점을 이야기합니다.

&nbsp;
&nbsp;
&nbsp;

## **In traditional machine learning**

- K-means clustering
- Hierarchical clustering
- Density estimation
- PCA

#### 특징

- Low dimensional data
- Simple concepts

&nbsp;
&nbsp;
&nbsp;

## **In Deep learing**

### Feature engineering vs. Representation learning

- **Feature engineering**
  - By human
  - Domain knowledge & Creactivity
  - Brainstorming
- **Representation learning**
  - By machine
  - Deep learning knowledge & coding skill
  - Trial and error



### Modern unsupervised learning

- High dimensional data
- Difficult concepts ➔ Not well understood, but surprisingly good performance
- Deep learning
- Unsupervised representation learning



### Representation in deep learning

- Deep learning representation is under constrained
  - Simple SGD can find one of the useful networks
  - Representation characteristics can be adjusted if needed
  - Learned representation becomes difficult to understand

- Disentangled representation
  - Alinged
  - Independent
  - Subspaces
  - Possible because severaly underconstrained



### Angle information

- 0 ~ 2&pi;
  - Algorithm thinks : 0 and 2&pi; are different / 0 and 1.9&pi; are far
- (x<sub>1</sub>, x<sub>2</sub>) = (cos(&theta;), sin(&theta;))
  - 0 and 2&pi; are the same
  - 0 and 1.9&pi; are close



### Spatial information

- Goal : Represent as mathematical object



### Human representation problems

- Human can understand
- Human can design with a goal

➔ Good representation in deep learning? : Useful and irrelevant



### A well defined task

- Typically, only on attribute of interest is considered as y
  - Imagenet - class
  - y is well defined because it is simply defined as human selected label
- Good representation - a vague concept (Supervised)
  - Even when y is well defined, what do we want for h<sub>i</sub> and h<sub>2</sub>?
  - Simply say "representation learning successful"  if good performance?
  - But then there is almost nothing we can sy about h<sub>i</sub> and h<sub>2</sub>
  - Other than saying "useful information has been well curated"
  - Is there anything we can say or pursue?
  - For a general purpose, what is a good representation?
  
  

### Information bottleneck

- For a well defined supervised task, what should h<sub>i</sub> and h<sub>2</sub> satisfy?
- Good representation - a vague concept (Unsupervised)
  - For a general purpose, whawt is a good representation?
  - General purpose often defined as a list of downstream tasks?
  - So, we go back to good performance for the tasks of interest?



### Representation

- What we want: a formal definition and evaluation metrics for representation
- Reality : No definition, task dependent evaluation methods

&nbsp;
&nbsp;
&nbsp;

## **Unsupervised representation learning**

- Unsupervised performance ≈ supervised performance
  - For linear evaluation
  - Thanks to instance discrimination, contrastive loss, and aggressive augmentation
- As in supervised learning
  - Performance metric can be unclear
  - Design of surrogate loss is an art (some principled; some hueristics based)
  - Training techinique development continuing (but augmentation methods are dominating)
- NLP
  - Masked language modeling
  - What next?
- Unsupervised representation learning
  - Still a long way to go...


&nbsp;
&nbsp;
&nbsp;

## Reference
> 본 포스팅은 LG Aimers 프로그램에서 학습한 내용을 기반으로 작성되었습니다. (전체 내용 X)
{: .prompt-warning }

1. LG Aimers AI Essential Course Module 3. 비지도학습, 서울대학교 이원종 교수

