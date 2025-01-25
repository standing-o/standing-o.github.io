---
title: "파이썬의 부작용 | Side Effect in Python"
date: 2023-09-26 13:00:00 +/-TTTT
categories: [Programming, Python]
tags: [python]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
description: 파이썬, Python, 파이썬의 부작용, Python Side Effect
---


------------------------

> 파이썬의 부작용(Side Effect in Python)을 알아봅시다.
{: .prompt-info }

파이썬의 부작용은 함수가 값을 리턴하는 대신 외부 세계의 어떤 State를 수정하거나 상호작용을 할 때 발생합니다.

이런 변화는 코드를 이해하기 어렵게 만들고 버그의 발생 가능성을 높일 수도 있습니다. 

&nbsp;
&nbsp;
&nbsp;

## **Side Effect in Python**
- 함수가 리턴 값, 함수의 상태, 또는 전역 프로그램 상태(Global Program State)를 변경하는 모든 작업을 의미합니다.
  - 전역/정적 변수 수정, 원본 객체의 변경, 콘솔 출력 생성, 파일 및 데이터베이스 쓰기가 포함됩니다.
- 이러한 동작은 추적 및 수정이 어려운 버그를 발생시킬 수 있습니다.

- **Side Effect가 있는 함수**

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


- **Side Effect가 없는 함수**
  - 동일한 입력에 대해 항상 동일한 출력을 생성해야 합니다.
  - **Pure Function** ㅣ Side Effect가 없는 함수로, 동일한 입력에 대해 항상 동일한 출력을 생성해야 합니다.

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
- 함수나 클래스를 래핑하여 소스 코드를 수정하지 않고도 실행 전후에 동작을 추가할 수 있어야 합니다.
  - 원본 데이터를 그대로 유지하며 데이터 복사본을 사용하도록 보장해야 합니다.

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
- 불변 객체는 생성된 후에는 변경이 불가능하므로 Side Effect를 방지할 수 있어야 합니다.
  - 불변 데이터 구조를 사용하면 데이터를 수정하는 작업 시 새로운 객체를 생성해야 합니다.
  - 불변 데이터 구조 예시 ㅣ Tuples, Strings, Frozensets

&nbsp;
&nbsp;
&nbsp;

## ex. Tensorflow `tf.function` Decorator

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

- `print` 문은 함수가 원래 코드를 실행하면서 Graph를 생성할 때 실행됩니다.
- Graph는 추적(Tracing) 과정에서 생성되며, 이후 실행되는 모든 호출에서는 Python 코드가 다시 실행되지 않습니다.

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

- Eager/Graph Execution 모두에서 값을 출력하려면 `tf.print`를 사용해야 합니다.


&nbsp;
&nbsp;
&nbsp;


---------------
## References
1. [Tensorflow Guide - Performance](https://www.tensorflow.org/guide/)
2. [Side Effect in Python](https://ecoagi.ai/topics/Python/side-effect-in-python)

