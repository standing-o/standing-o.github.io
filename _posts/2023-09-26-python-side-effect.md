---
title: "파이썬의 부작용 | Side Effect in Python"
date: 2023-09-26 13:00:00 +/-TTTT
categories: [Programming, Python]
tags: [tabular, tree, random-forest, gradient-boosting, inductive-bias]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
description: 파이썬의 부작용 | Python Side Effect
---


------------------------

> 파이썬의 부작용을 알아봅시다.
{: .prompt-info }

파이썬의 부작용은 함수가 값을 리턴하는 대신 외부 세계의 어떤 state를 수정하거나 상호작용을 할 때 발생합니다.

&nbsp;
&nbsp;
&nbsp;

## **Side Effect in Python**
- 리턴 값, 함수의 state, global program state 에 대한 모든 변경사항
  - global/static 변수 및 original object의 수정, console output의 생성, 파일 및 데이터베이스 쓰기
- 추적 및 수정이 어려운 버그가 발생할 수 있음
- Side Effect가 있는 함수

```python
# Python side effect example
def add_element(data, element):
    data.append(element)
    return data
 
my_list = [1, 2, 3]
print(add_element(my_list, 4))  # Output: [1, 2, 3, 4]
print(my_list)  # Output: [1, 2, 3, 4]
```

&nbsp;
&nbsp;
&nbsp;


- Side Effect가 없는 함수: 동일한 입력에 대해 항상 동일한 출력을 생성
  - Pure Function: 동일한 입력에 대해 항상 동일한 출력을 생성

```python
# Python pure function example
def add_element_pure(data, element):
    new_list = data.copy()
    new_list.append(element)
    return new_list
 
my_list = [1, 2, 3]
print(add_element_pure(my_list, 4))  # Output: [1, 2, 3, 4]
print(my_list)  # Output: [1, 2, 3]
```

&nbsp;
&nbsp;
&nbsp;


## **Decorator로 Side Effect 제어하기**
- 함수/클래스를 래핑하면 decorator는 소스 코드를 변경하지 않고도 래핑된 함수가 실행되기 전/후에 코드를 실행
  - 원본 데이터를 그대로 유지하면서 데이터 복사본을 작업에 사용하도록 보장

```python
def no_side_effects_decorator(func):
    def wrapper(*args, **kwargs):
        data_copy = args[0].copy()  # create a copy of the data
        return func(data_copy, *args[1:], **kwargs)
    return wrapper

# 함수를 `no_side_effects_decorator` 래핑
@no_side_effects_decorator
def add_element(data, element):
    data.append(element)
    return data
 
my_list = [1, 2, 3]
print(add_element(my_list, 4))  # Output: [1, 2, 3, 4]
print(my_list)  # Output: [1, 2, 3]
```

&nbsp;
&nbsp;
&nbsp;

## **불변 데이터 구조**
- 불변 객체는 생성된 후에는 변경될 수 없으므로 Side Effect를 피하는데 효과적
  - 불변 데이터 구조를 사용하는 경우 데이터를 수정하는 모든 작업에서는 새 객체가 생성됨
  - tuples, strings, frozensets

&nbsp;
&nbsp;
&nbsp;

## **Example. Tensorflow `tf.function` Decorator**

```python
@tf.function
def get_MSE(y_true, y_pred):
  print("Calculating MSE!")
  sq_diff = tf.pow(y_true - y_pred, 2)
  return tf.reduce_mean(sq_diff)

error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
```

<pre>
    # 한 번만 출력됨
    Calculating MSE!
</pre>

&nbsp;
&nbsp;
&nbsp;

- `print` 문은 `Function`이 원래 코드를 실행할 때 실행되며 Tracing Process를 통해 Graph를 생성 
- 추적은 TensorFlow 연산을 Graph로 캡처하고 `print`는 Graph로 캡처되지 않음 
- 이 Graph는 세 번의 모든 호출시 실행되지만 Python 코드를 다시 실행하지는 않음

```python
# Globally set everything to run eagerly to force eager execution.
tf.config.run_functions_eagerly(True)
(...)
```

<pre>
    Calculating MSE!
    Calculating MSE!
    Calculating MSE!
</pre>

&nbsp;
&nbsp;
&nbsp;

- Eager/Graph Execution 모두에서 값을 인쇄하려면 `tf.print` 사용


&nbsp;
&nbsp;
&nbsp;


## References
1. [Tensorflow Guide - Performance](www.tensorflow.org/guide/)
2. [Side Effect in Python](https://ecoagi.ai/topics/Python/side-effect-in-python)

