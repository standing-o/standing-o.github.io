---
title: "비지도학습 | Unsupervised learning 과 딥러닝"
date: 2022-07-17 17:00:00 +/-TTTT
categories: [Machine learning, Basic]
tags: [lg-aimers, unsupervised-learning]
math: true
---



----------------

- 본 포스팅은 인공지능의 비지도 학습 개념과 종류를 설명하고 있습니다.
- Keyword :  Unsupervised learning



## **In traditional machine learning**

- K-means clustering
- Hierarchical clustering
- Density estimation
- PCA

#### **특징**

- Low dimensional data
- Simple concepts



## **In Deep learing**

### **Feature engineering vs. Representation learning**

- Feature engineering
  - By human
  - Domain knowledge & Creactivity
  - Brainstorming
- Representation learning
  - By machine
  - Deep learning knowledge & coding skill
  - Trial and error



### **Modern unsupervised learning**

- High dimensional data
- Difficult concepts ➔ Not well understood, but surprisingly good performance
- Deep learning
- Unsupervised representation learning



### **Representation in deep learning**

- Deep learning representation is under constrained
  - Simple SGD can find one of the useful networks
  - Representation characteristics can be adjusted if needed
  - Learned representation becomes difficult to understand

- Disentangled representation
  - Alinged
  - Independent
  - Subspaces
  - Possible because severaly underconstrained



### **Angle information**

- 0 ~ 2&pi;
  - Algorithm thinks : 0 and 2&pi; are different / 0 and 1.9&pi; are far
- (x<sub>1</sub>, x<sub>2</sub>) = (cos(&theta;), sin(&theta;))
  - 0 and 2&pi; are the same
  - 0 and 1.9&pi; are close



### **Spatial information**

- Goal : Represent as mathematical object



### **Human representation problems**

- Human can understand
- Human can design with a goal

➔ Good representation in deep learning? : Useful and irrelevant



### **A well defined task**

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
  
  

### **Information bottleneck**

- For a well defined supervised task, what should h<sub>i</sub> and h<sub>2</sub> satisfy?
- Good representation - a vague concept (Unsupervised)
  - For a general purpose, whawt is a good representation?
  - General purpose often defined as a list of downstream tasks?
  - So, we go back to good performance for the tasks of interest?



### **Representation**

- What we want: a formal definition and evaluation metrics for representation
- Reality : No definition, task dependent evaluation methods



----------------

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



----

#### **References**
- 본 포스팅은 `LG Aimers` 프로그램에 참가하여 학습한 내용을 기반으로 작성되었습니다. (전체내용 X)

➔ [`LG Aimers` 바로가기](https://www.lgaimers.ai/)


```
[1] LG Aimers AI Essential Course Module 3.비지도학습, 서울대학교 이원종 교수 
```

