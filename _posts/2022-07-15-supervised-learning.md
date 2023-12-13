---
title: "지도학습이란? (SVM, ANN, Ensemble) | Supervised learning"
date: 2022-07-15 17:00:00 +/-TTTT
categories: [AI Theory, Machine Learning]
tags: [lg-aimers, supervised-learning, svm, ann, ensemble]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: 지도학습 | Supervised learning, overfitting, underfitting, linear model, SVM, ANN, ensemble
---



--------------------------

> 인공지능의 지도학습 개념과 그 종류 (선형모델, SVM, ANN) 를 정리합니다.
{: .prompt-info }

지도학습에 대한 데이터 크기, 모델 복잡성, 일반화 등의 주제를 다룹니다. 

또한 모델의 편향과 분산, 오버피팅/언더피팅에 대한 이해와 교차 검증의 중요성을 강조합니다.

&nbsp;
&nbsp;
&nbsp;

## **지도학습 <sup>Supervised learning</sup>**

- Given a set of labeled examples $$(x^1, y^1),...,(x^N, y^N)$$, learn a mapping function g : X ➔ Y, s.t. given an unseen sample x', associated output y' is predicted.

- Supervised learning relies on the sizes of dataset; what if we have no sufficient data?
  - Data augmentation, learning from insufficient labels (weak supervision)
- What if the data properties are different between datasets?
  - Domain adaptation, transfer learning



### Problem Formulation

- $$X = R^d$$ is an input space
  - $$X = R^d$$ : a d-dimensional Euclidean space
  - Input vector $$x ∈ X : x = (x_1,...,x_d)$$
- Y is an output space (binary decision)
- We want to approximate a target function f
  - f : X ➔ Y (unknown ideal function)
  - Data $$(x^1, y^1),...,(x^N, y^N)$$; dataset where $$y^N = f(X^N)$$
  - Correct label is ready for a training set
  - Hypothesis $$g : X ➔ Y$$ (ML model to approximate $$f$$), $$g ∈ H$$
- Learning model : feature selection, model selection, optimization



### Model Generalization

- Learning is an ill-posed problem; data is limited to find a unique solution
- Generalization (goal) : a model needs to perform well on unseen data
  - Generalization error E<sub>gen</sub>; the goal is to minimize this error, but it is impractical to compute in the real world
  - Use training/validation/test set errors for the proxy



### Errors

- Pointwise error is measured on an each input sample : $$e(h(x), y)$$
- From a pointwise error to overall errors: $$E[(h(x^i) - y^i)^2]$$
  - If an input sample is chosen from training, validation, and testing datasets, the errors are called a training error (E<sub>train</sub>), a validation error (E<sub>val</sub>), and a testing error (E<sub>test</sub>).
- Training error E<sub>train</sub> measured on a training set, which may or may not represent E<sub>gen</sub>; used for fitting a model
- Testing error E<sub>test</sub> (not used in training), which can be used for a proxy of E<sub>gen</sub>.
- Goal :  E<sub>test</sub> **≈** E<sub>gen</sub> **≈** 0



### Overfitting and Underfitting

- **Underfitting** problem because of using too simpler model than actual data distribution 
  - **High Bias**
- **Overfitting** problem because of more complex model than actual data distribution 
  - **High Variance**
  - **Avoid overfitting**
    - Problem : In today's ML problems, a complex model tends to be used to handle high-dimensional data (and relatively insufficient number of data); prone to an overfitting problem
    - Curse of dimension : Will you increase the dimension of the data to improve the performance as well as maintain the density of the examples per bin? If so, you need to increase the data exponentially.
    - Remedy : Data augmentation, regularization, ensemble



### Bias and Variance

- **Bias** : error because the model can not represent the concept
- **Variance** : error because a model overreacts to small changes (noise) in the training data

➔ Total loss = Bias + Variance (+ noise)

➔ Bias-variance trade-off : the two objective have trade-off between approximation and generalization w.r.t model complexity



### Cross-validation (CV)

- CV allows a better model to avoid overfitting (but more complexity)



&nbsp;
&nbsp;
&nbsp;

## **Linear Regression**

- Hypothesis set H, model parameter &theta;

$$
h_\theta(x)=\theta_0+\theta_1x_1+...+\theta_dx_d=\theta^Tx
$$

- Good for a first try : simplicity, generalization
- L<sub>2</sub> cost function (Goal : minimizing MSE)

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
$$

➔ $$\text{minimize}_{\theta_0, \theta_1} J(\theta_0, \theta_1)$$



### Optimization

- Data matrix X ∈ R<sup>Nx(d+1)</sup>, target vector y ∈ R<sup>N</sup>, weight vector &theta; ∈ R<sup>d+1</sup>
- In-sample error : $$\mid\mid{y - X\theta}\mid\mid_2$$
- **Normal equation**
  - (Least square) E is continuous, differentiable, and convex
  - Need to compute an inverse matrix and slow if the number of samples is very large

$$
\theta^{*}=\text{argmin}_{\theta}E(\theta)
$$

$$
=\text{argmin}_{\theta}[\frac{1}{N}(\theta^TX^TX\theta-2\theta^TX^Ty+y^Ty)]
$$


$$
\therefore\theta^{*}=(X^TX)^{-1}X^Ty=X^{+}y
$$

➔ Problem : huge computational complexity, non-invertible matrix ➔ needs iterative algorithm (gradient descent)   

- **To avoid overfitting**
  - If we have too many features, the hypothesis may fit the training set very well. However, it may fail to generalize to new samples.
  - More features ➔ more parameters ➔ need more data; (in practive) less data ➔ overfitting
  ➔ Reduce the number of features, regularization



#### Gradient Descent Algorithm

$$
\theta_{new}=\theta_{old}-\alpha\frac{\partial}{\partial\theta}J(\theta)
$$

- J is the objective function that we want to optimize. &alpha; : the step size to control the rate to move down the error surface (hyper parameter).
  - If &alpha; is too small, gradient descent can be slow.
  - If &alpha; is too large, gradient descent can overshoot the minimum.
- Gradient descent works well even when n large
- All examples (batch) are examined at each iteration : Use stochastic gradient descent or mini batch.
- Advances : AdaGrad, RMSProp, Adam
- Limitation : local optimum ➔ cannot guarantee global minimum but attempt to find a good local minimum
  - **To avoid local minimum**
    - **Momentum** : designed to speed up learning in high curvature and small/noise gradients ➔ exponentially weighted moving average of past gradients (low pass filtering)
    - **SGD + Momentum** : use a velocity as a weighted moving average of previous gradients
    - **Nesterov Momentum** : difference from standard momentum where gradient g is evaluated (lookahead gradient step)
    - **AdaGrad** : adapts an indibidual learning rate of each direction
    - **RMSProp** : attempts to fix the drawbacks of AdaGrad, in which the learning rate becomes infinitesimally small and the algorithm is no longer able learning when the accumulated gradient is large.
    - **Adam** : RMSProp + momentum
- Learning rate scheduling

➔ Need to gradually decrease learning rate over time


&nbsp;
&nbsp;
&nbsp;

## **Linear Classification**

- Uses a hyperplane as a decision boundary to classify samples based on a linear combination of its explanatory variables
- The linear formula g ∈ H can be written as:

$$
h(x)=sign((\sum^{d}_{i=1}w_ix_i)+w_0)
$$

➔ $$x_0 = 1, w_0$$ : a bias term, $$sign(x) = 1{\quad}if{\quad}x>0;0{\quad}if{\quad}x<0;$$

- **Sigmoid function**
  - Used to map a score value into a probability value.
  - Squash the output of the linear function:

$$
\sigma(-w^Tx)=\frac{1}{1+e^{-w^Tx}}
$$

- Advantage : simplicity and interpretability

### Loss

- **Hinge Loss**

$$
\text{Loss}_\text{hinge}(x,y,w)=\text{max}\{1-(w\cdot\phi(x))y, 0\}
$$

- **Cross-entropy Loss**
  - Considers two probability mass functions (pmf) {p, 1-p} and {q, 1-q} with a binary outcomes:
  - Cross-entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1.

$$
CE(S,Y)=-\sum_{\forall{i}}Y_i\text{log}(S_i))
$$

&nbsp;
&nbsp;
&nbsp;

## **Support Vector Machine (SVM)**

- Choose the linear separator (hyperplane) with the largest margin on either side
  - Maximum margin hyperplane with support vectors
  - Robust to outliers



### Support Vector

- an instance with the minimum margin, which will be the most sensible data points to affect the performance



### Margin

- twice the distance from the hyperplane to the nearest instance on either side

- w : orthogonal to the hyperplane



### Optimization

- Optimal weight w and bias b
- Classifiers points correctly as well as achieves the largest possible margin
- Hard margin SVM assumes linear separability
- Soft margin SVM extends to non-separable cases

➔ Nonlinear transformation and kernel trick

- **Constraints** : linearly separable, hard-margin linear SVM

$$
h(x)=w^Tx+b\geq1\text{ for }y=1
$$

$$
h(x)=w^Tx+b\leqq-1\text{ for }y=-1
$$

$$
y(w^Tx+b)\geq1\text{ for all samples}
$$

- **Objective function** : linearly separable, hard-margin linear SVM
  - Distance from a support vector to the hyper plane:

$$
\frac{w^Tx+b}{\mid\mid{w}\mid\mid}=\frac{\pm1}{\mid\mid{w}\mid\mid}\longrightarrow\frac{2}{\mid\mid{w}\mid\mid}
$$


### Kernel Trick (not linearly separable)

- **Polynomials**:

$$
K(x,y)=(x\cdot{y}+1)^p
$$

- **Gaussian radial basis function (RBF)**:

$$
K(x,y)=e^{-\mid\mid{x-y}\mid\mid^2/2\sigma^2}
$$

- **Hyperbolic tangent (multilayer perceptron kernel)**:

$$
K(x,y)=\text{tanh}(kx\cdot{y}-\delta)
$$


&nbsp;
&nbsp;
&nbsp;

## **Artificial Neural Network (ANN)**

- Needs elaborated training schemes to improve performance

- **Activation functions**
  - Sigmoid neurons give a real-valued output that is a smooth and bounded function of their total input
  - Non-linearity due to the activation functions
- Deep neural network can represent more complex (non-linear) boundaries with increasing neurons
- **Multilayer perceptron (MLP)**
  - Can solve XOR problem
- ANN for non-linear problem
  - There exists cases when the accuracy is low even if the number of layers is high

➔ The result of one ANN is the result of sigmoid function

➔ The numerous multiplication of this result converges to near zero ➔ Gradient vanishing problem



### Back Propagation

- **Back propagation** barely changes lower-layer parameters (vanishing gradient)
- Breakthrough
  - Pre-training + fine tuning
  - CNN for reducing redundant parameters
  - Rectified linear unit (constant gradient propagation)
  - Dropout



&nbsp;
&nbsp;
&nbsp;

### Performance Evaluation

- **Accuracy** = (TP+TN)/ALL
- **Precision** = TP/(TP+FP)
- **Recall** = TP/(TP+FN)
- **F1** = PxR/(P+R)
- **TPR** = R = TP/(TP+FN)
- **TNR** = TN/(TN+FP)
- **False positive error** : predict = positive, actual = negative
- **False negative error** : predict = negative, actual = positive
- **ROC Curve** : performance comparisons between different classifiers in different true positive rates (TPR) and true negative rates (TNR).



### Error Measure

- The error measure should be specified by the user 

➔ Not always given but needs to be carefully considered

&nbsp;
&nbsp;
&nbsp;

## **Ensemble Learning**

- Predict class label for unseen data by aggregating a set of predictions : different classifiers (experts) learned from the training data
- Make a decision with a voting
- **Bagging and boosting** : improving decision tree
  - By bagging : random forest (inherently boosting)
  - By boosting : gradient boosting machine as generalized Adaboost
- Advantages
  - Improve predictive performance, Other types of classifiers can be directly included
  - Easy to implement, No too much parameter tuning
- Disadvantages
  - Not a compact representation

#### Bagging

- **Bootstrapping + aggregating** (for more robust performance; lower variance)
- Train several models in parallel
- Bagging works because it reduces variance by voting/averaging (robust to overfitting)
  - Learning algorithm is unstable; if small changes to the training set cause large changes in the learned classifier.
  - Usually, the more classifiers the better

#### Boosting

- **Cascading of week classifiers** ➔ training multiple models in sequence, adaboost
  - Adaboost : trained on weighted form of the training set, weight depends on the performance of the previous classifier, combined to give the final classifier
- Simple and easy, flexible, Versatile, non-parametric
- No prior knowledge needed about week learner

&nbsp;
&nbsp;
&nbsp;

## Reference
> 본 포스팅은 LG Aimers 프로그램에서 학습한 내용을 기반으로 작성되었습니다. (전체 내용 X)
{: .prompt-warning }

1. LG Aimers AI Essential Course Module 2. 지도학습(분류/회귀), 이화여자대학교 강제원 교수 

