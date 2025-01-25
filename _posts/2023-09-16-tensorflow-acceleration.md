---
title: "텐서플로우 가속 | Tensorflow Acceleration"
date: 2023-09-16 13:00:00 +/-TTTT
categories: [AI Framework, Tensorflow]
tags: [tensorflow, python, distributed-training, gpu-acceleration]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: 텐서플로우, Tensorflow 가속기, 텐서플로우 가속기, Tensorflow Acceleration, Tensorflow Accelerate, Distributed Training, GPU Acceleration
---


------------------------
> GPU 환경에서 Tensorflow 가속기를 활용하는 방법들을 정리합니다.
{: .prompt-info }

TensorFlow는 기본적으로 CPU를 사용하여 수치 연산을 처리하지만 대규모 데이터나 복잡한 모델을 다룰 때는 GPU, TPU와 같은 가속기를 활용하여 연산을 가속할 수 있습니다.

&nbsp;
&nbsp;
&nbsp;

## **Distributed Training**
### `tf.distribute.Strategy`
- Multiple GPU, 머신, TPU를 활용한 분산 훈련이 가능하도록 지원합니다.
- Tensorflow 2 버전에서 `tf.function`을 사용하여 그래프 실행을 수행합니다.

#### Synchronous vs Asynchronous Training
- **Synchronous** ㅣ 모든 Trainer가 동기화된 입력 데이터의 다른 슬라이스에 대해 학습을 수행한 후 각 단계에서 Gradient를 집계합니다.
- **Asynchronous** ㅣ 모든 Trainer가 입력 데이터를 독립적으로 학습한 후, 파라미터를 비동기적으로 업데이트합니다.

### `tf.distribute.MirroredStrategy`
- 단일 머신에 여러 GPU를 사용하는 동기 분산 학습을 지원합니다.
- GPU당 하나의 복제본을 생성하며, 각 모델 변수는 모든 복제본에서 미러링됩니다.
  - 변수들은 `MirroredVariable` 이라는 단일 개념적 변수를 형성합니다.
  - 동일한 업데이트를 적용하며, 복제본 간 동기화를 유지합니다.
- 여러 장치에 변수의 변경사항을 전달하기 위해 **All Reduce** 알고리즘 사용합니다.
- 텐서플로우에 표시되는 모든 GPU를 사용하고 NCCL을 장치 간 통신 수단으로 사용하는 `MirroredStrategy` 인스턴스를 생성합니다.

```python
mirrored_strategy = tf.distribute.MirroredStrategy()
```

- 일부 GPU만 사용

```python
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
```

- 장치 간 통신을 재정의
  - Options : `tf.distribute.NcclAllReduce` (Default), `tf.distribute.HierarchicalCopyAllReduce`, `tf.distribute.ReductionToOneDevice`

```python
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
```

- Multiple Workers -> `tf.distribute.MultiWorkerMirroredStrategy`

&nbsp;
&nbsp;
&nbsp;

### `tf.distribute.Strategy` with Keras `model.fit`
- `tf.distribute.Strategy`는 `tf.keras`에 통합됩니다.
- `MirroredStrategy` `strategy.scope()`는 학습을 분산할 때 사용할 Strategy을 Keras에 표시합니다.     
  -> Model, Optimizer, Metric을 생성하면 일반 변수 대신 분산 변수를 생성할 수 있습니다.

```python
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])

model.compile(loss='mse', optimizer='sgd')

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)

# Numpy도 사용 가능
import numpy as np

inputs, targets = np.ones((100, 1)), np.ones((100, 1))
model.fit(inputs, targets, epochs=2, batch_size=10) # 각 배치가 여러 복제본에 균등하게 분할
```

&nbsp;
&nbsp;
&nbsp;

```python
# Compute a global batch size using a number of replicas.
BATCH_SIZE_PER_REPLICA = 5
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     mirrored_strategy.num_replicas_in_sync)
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100)
dataset = dataset.batch(global_batch_size)

LEARNING_RATES_BY_BATCH_SIZE = {5: 0.1, 10: 0.15, 20:0.175}
learning_rate = LEARNING_RATES_BY_BATCH_SIZE[global_batch_size]
```

### `tf.distribute.Strategy` with Custom Training Loops


```python
with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  optimizer = tf.keras.optimizers.SGD()

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(
  global_batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

loss_object = tf.keras.losses.BinaryCrossentropy(
  from_logits=True,
  reduction=tf.keras.losses.Reduction.NONE)

def compute_loss(labels, predictions):
  per_example_loss = loss_object(labels, predictions)
  return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

def train_step(inputs):
  features, labels = inputs

  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    loss = compute_loss(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

@tf.function
def distributed_train_step(dist_inputs):
  per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, 
                                  per_replica_losses, axis=None)

for dist_inputs in dist_dataset:
  print(distributed_train_step(dist_inputs))

# explicit
# iterator = iter(dist_dataset)
# for _ in range(10):
#   print(distributed_train_step(next(iterator)))
```
&nbsp;
&nbsp;
&nbsp;

### `TF_CONFIG` 환경 변수를 설정
```python
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}
})
```

&nbsp;
&nbsp;
&nbsp;

## **GPU Acceleration**
- TensorFlow가 GPU를 사용하고 있는지 확인합니다.

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

- `/device:CPU:0` ㅣ CPU
- `/GPU:0` ㅣ 첫번째 GPU에 대한 Short-hand Notation
- `/job:localhost/replica:0/task:0/device:GPU:1` ㅣ 두번째 GPU에 대한 Fully Qualified Name

### **GPU 장치 할당**
- 기본적으로 연산이 할당될 때 GPU 장치에 우선순위가 지정되어야 합니다.
  - `tf.matmul`연산은 CPU 및 GPU 커널을 모두 사용할 수 있으며, `GPU:0` 장치가 `tf.matmul` 연산을 실행하도록 선택되어야 합니다.
- 텐서플로우 작업에 해당 GPU 구현이 없는 경우 연산은 CPU 장치로 대체되어야 합니다.
  - `tf.cast`연산은 CPU 커널만 있기 때문에 `CPU:0` 장치가 `tf.cast` 연산을 실행하도록 선택되어야 합니다.
- 장치 할당 로깅이 활성화되면 MatMul 연산이 `GPU:0`에서 수행된다는 정보가 기록되어야 합니다.

```python
tf.debugging.set_log_device_placement(True)

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
```

<pre>
    Executing op _EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0
    Executing op _EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0
    Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
    tf.Tensor(
    [[22. 28.]
     [49. 64.]], shape=(2, 2), dtype=float32)
</pre>

- 장치 수동 할당

```python
tf.debugging.set_log_device_placement(True)

with tf.device('/CPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```

&nbsp;
&nbsp;
&nbsp;

### **GPU 메모리 제한**
- 텐서플로우는 모든 GPU의 거의 모든 메모리를 프로세스가 볼 수 있도록 매핑해야 합니다.
  - 이는 메모리 조각화(Memory Fragmentation)를 줄이고, 상대적으로 제한된 GPU 메모리 리소스를 효율적으로 사용하기 위함입니다.
    - **메모리 조각화(Memory Fragmentation)** ㅣ 메모리 공간이 작게 나뉘어져 있어 사용 가능한 메모리가 충분하더라도 할당이 불가능한 상태를 말합니다.
  - `tf.config.set_visible_devices` ㅣ 텐서플로우가 접근할 수 있는 GPU를 조정할 수 있어야 합니다.

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
```

- `tf.config.experimental.set_memory_growth` ㅣ 런타임 할당에 필요한 만큼의 GPU 메모리만 할당합니다.
  - 처음에는 매우 적은 메모리만 할당하고, GPU 메모리 영역이 점점 확장되면서, 메모리 조각화를 방지하기 위해 메모리 할당이 해제되지 않습니다.

```python
# 텐서를 할당하거나 연산을 실행하기 전 실행
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
```

&nbsp;
&nbsp;
&nbsp;


- `tf.config.set_logical_device_configuration` 으로 가상 GPU를 설정 후 GPU 할당 전체 메모리를 제한합니다.

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
      gpus[0],
      [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
```

&nbsp;
&nbsp;
&nbsp;

### **Multiple GPU**
- 단일 GPU가 있는 시스템에서 개발하는 경우, 가상 기기로 여러 GPU를 시뮬레이션할 수 있습니다.

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
```

&nbsp;
&nbsp;
&nbsp;

- `tf.distribute.Strategy`
  - **Data Parallelism** ㅣ 입력 데이터를 나누고 모델의 복사본을 각 GPU에서 실행합니다.

```python
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
  inputs = tf.keras.layers.Input(shape=(1,))
  predictions = tf.keras.layers.Dense(1)(inputs)
  model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
  model.compile(loss='mse',
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))
```

&nbsp;
&nbsp;
&nbsp;


---------------------
## Reference
1. [Tensorflow Guide - Accelerators](https://www.tensorflow.org/guide/)
