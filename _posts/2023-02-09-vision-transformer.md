---
title: "Vision Transformer (ViT)"
date: 2023-02-09 13:00:00 +/-TTTT
categories: [Deep Learning, Computer Vision]
tags: [vision-transformer, vit, transformer, attention]
math: true
---


------------------------
- 이 게시글은 Vision Transformer의 기본 원리와 구조, 수식을 소개합니다.

`Original Paper Review` 
| [An image is worth 16x16 words: Transformers for image recognition at scale](https://github.com/standing-o/Machine_Learning_Paper_Review/issues/15)

# **Overview**
![VIT](/assets/img/for_post/20230209-1.png)

- Image Patch Sequence를 Input으로 하여 기존의 Transformer 구조를 거의 그대로 Vision에 활용함
- CNN에 비해 Inductive Bias가 부족함
  ➔  데이터 전체를 보고 Attention할 위치를 정하기 때문
- 적은 데이터로 학습할 경우, Resnet50보다 성능이 더 떨어짐
- Imagenet 21K (1400만 장), JFT-300M (3억 장) 으로 Pre-training, CIFAR-10 (6만 장) 으로 Transfer Learning

# **Formulation**
## **1. Input**
### 1-1 Image
- (C, H, W) = (Channel, Height, Width)
    
$$x\in\mathbb{R}^{C\times{H}\times{W}}$$
    

### 1-2. Flatten
- P is the patch size and N is the number of patches per image.
     
$$x_p \in \mathbb{R}^{N \times (P^2 C)} \quad \text{where} \quad N=\frac{HW}{P^2}$$

### 1-3. Linear Projection
- D is the dimension of the latent vector.
     
$$[x^1_p E; x^2_p E; \cdots ; x^N_p] \in \mathbb{R}^{N \times D} \quad \text{where} \quad E \in \mathbb{R}^{(P^2 C)\times D}$$

### 1-4. Insert the Class Token
$$z_0= [x_{cls}; x^1_p E; x^2_p E; \cdots ; x^N_p] \in \mathbb{R}^{(N+1)\times D}$$

### 1-5. Add the Positional Encoding
(in progress)
    
$$z_0= [x_{cls}; x^1_p E; x^2_p E; \cdots ; x^N_p] + E_{pos} \in \mathbb{R}^{(N+1)\times D} \quad \text{where} \quad E_{pos} \in \mathbb{R}^{(N+1)\times D}$$

## **2. Transformer Encoder**
### 2-1. Multi-Head Self Attention (MSA)
- Query, Key and Value
    
$$q = z \cdot w_q, \,\, k=z\cdot w_k, \,\, v = z \cdot w_v \quad \text{where} \quad w_q, w_k, w_v \in \mathbb{R}^{D \times D_h}$$
    
- Self-Attention 
  - A is the attention score matrix
  - Dividing by D<sub>h</sub><sup>1/2</sup> value is to prevent the softmax from having the small gradient.
    
  $$SA(z) = A \cdot v \in \mathbb{R}^{N \times D_h} \quad \text{where} \quad A=\text{softmax}(\frac{q \cdot k^T}{\sqrt{D_h}}) \in \mathbb{R}^{N \times N}$$

- Multi-head Self Attention
    
$$\text{MSA}(z) = [SA_1(z); SA_2(z); \cdots ; SA_k(z)]U_{msa} \quad \text{where} \quad U_{msa} \in (k, D_h, D)$$

### 2-2. Multi-Layer Perceptron (MLP)

(in progress)

### 2-3. Encoder with L layers
- `Layer Normalization ➔ Multi-head Self Attention ➔ Skip Connection`
    
$$z'_l = MSA(LN(z_{l-1})) + z_{l-1},$$
    
- `Layer Normalization ➔ Multi-layer Perceptron ➔ Skip Connection`
    
$$z_l = MLP(LN(z'_l)) + z'_l \quad \text{for} \quad l=1,2,\ldots, L$$
    
- L is the number of layers.
    
$$\text{where} \quad LN(z^j_i) = \gamma \frac{z^j_i - \mu_i}{\sqrt{\sigma^2_i + \epsilon}} + \beta$$
    
- &gamma;  and &beta; are trainable.

## **3. Output**
- Make the prediction using the class token.
    
$$\hat{y} = LN(z^0_L) \in \mathbb{R}^C \quad \text{where} \quad z^0_L \in \mathbb{R}^D \quad (z_L \in \mathbb{R}^{(N+1) \times D})$$
    
- C is the number of classes.

# **ViT Variants**
(in progress)
