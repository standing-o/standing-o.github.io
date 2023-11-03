---
title: "파이썬 라이브러리로 간단하게 음성 데이터 전처리 하기"
date: 2023-11-03 17:00:00 +/-TTTT
categories: [AI Theory, Audio]
tags: [speech, audio, voice, mfcc, spectrogram, melspectrogram]
math: true
toc: true
---


------------------------
- 음성 데이터에 대한 Feature Extraction은 음성 관련 모델링에서 중요한 과정 중 하나입니다. 
- Feature Extraction 에는 여러 기법이 사용되며, 음성 신호를 효율적으로 표현하고 노이즈를 감소시켜 머신러닝/딥러닝 모델링에 적합한 형태로 변환시켜 줍니다.
- 파이썬 라이브러리를 통해 간단하게 음성의 특성을 추출하는 방법 및 여러 전처리 방식들을 소개하겠습니다.


### **0. 데이터 불러오기**

```python
import librosa

sr = 22050
data, sample_rate = librosa.load(path, sr=sr)
```

- `data`: 오디오 파일의 파형 데이터가 numpy 배열로 반환
- `sample_rate` : 오디오 데이터의 샘플링 속도, 초당 샘플링된 음성 데이터 포인트의 수, Hz단위

### **1. Melspectrogram**

```python
S = librosa.feature.melspectrogram(audio_array, sr=sr, n_mels=40)
log_S = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(log_S, sr=sr)  # x axis: Time, y axis: Frequency
```

- 음성데이터를 Frequency, Time의 2차원 도메인으로 변환합니다.
  - `n_mels`: Melspectrogram의 주파수 해상도 조절
  - `power_to_db`: Melspectrogram을 데시벨로 스케일링하여 특징을 강조
  - `ref=np.max`: 데시벨로 변환 시 사용되는 참조 값

### **2. MFCC**

```python
mfcc = librosa.feature.mfcc(audio_array, sr=sr, n_mels=40)
librosa.display.specshow(mfcc, sr=sr)  # x axis: Time, y axis: MFCC coefficients
```

### **3. Data Augmentation**

- 노이즈를 추가합니다.

```python
def noising(data,noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data
```

- 좌우로 이동시킵니다.

```python
def shifting(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max+1)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data
```

- 피치를 조절합니다.

```python
def change_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
```

### **4. 그 외 전처리 기법**
#### **Random Padding**
- 음성 데이터들의 길이가 서로 다른 경우 Melspectrogram, MFCC 모두 길이가 제각각 이므로, 고정 사이즈를 정하고 랜덤으로 앞과 뒤에 Padding을 해줍니다.
- 데이터마다 생기는 데시벨 차이를 고려하여 Min Max Scaling도 활용합니다.

```python
def random_pad(mels, pad_size, mfcc=True):

  pad_width = pad_size - mels.shape[1]
  rand = np.random.rand()
  left = int(pad_width * rand)
  right = pad_width - left
  
  if mfcc:
    mels = np.pad(mels, pad_width=((0,0), (left, right)), mode='constant')
    local_max, local_min = mels.max(), mels.min()
    mels = (mels - local_min)/(local_max - local_min)
  else:
    local_max, local_min = mels.max(), mels.min()
    mels = (mels - local_min)/(local_max - local_min)
    mels = np.pad(mels, pad_width=((0,0), (left, right)), mode='constant')

  return mels
```

#### **Trimming**
- 60Db 이하를 무음으로 trimming 해줍니다.
```python
def trimming(data, sampling_rate, top_db):
    frame_length = 0.03
    frame_stride = 0.01
    input_nfft = int(round(sampling_rate*frame_length))
    input_stride = int(round(sampling_rate*frame_stride))
    trim = data.apply(lambda x : librosa.effects.trim(x, top_db=top_db, frame_length=input_nfft, hop_length=input_stride)[0])
    return trim
```

#### **Specaugment Augmentation**
- MFCC 또는 Melspectrogram과 각각의 1,2차 미분을 stacking 하여 3차원 특성을 생성합니다.
- MFCC 벡터는 단일 프레임의 Power Spectrum 포락선만 설명하지만, 음성에도 역학 정보가 있을 수 있습니다. 
  - 즉 시간에 따른 MFCC의 궤도를 계산하고 이를 원래의 특징 벡터에 추가하여 성능을 향상시킬 수 있습니다.

```python
def preprocess_dataset(data, sampling_rate, frame_length, frame_stride, feature):
    nfft = int(round(sampling_rate*frame_length))
    stride = int(round(sampling_rate*frame_stride))

    features = []
    for i in data:
        if feature == "mfcc":
            n_feature = 40
            S = librosa.feature.mfcc(y=i, sr=sampling_rate, n_mfcc=n_feature, n_fft=nfft, hop_length=stride)
            S_delta = librosa.feature.delta(S)
            S_delta2 = librosa.feature.delta(S, order=2)
            
        elif feature == "melspec":
            n_feature = 128
            S = librosa.feature.melspectrogram(y=i, sr=sampling_rate, n_mels=n_feature, n_fft=nfft, hop_length=stride)
            S = librosa.power_to_db(S, ref=np.max)
            S_delta = librosa.feature.delta(S)
            S_delta2 = librosa.feature.delta(S, order=2)
            
        S = np.stack((S, S_delta, S_delta2), axis=2)
        features.append(S)
    return features
```
-------------------
- 이러한 기법들을 비교하면서 모델에 맞는 특성을 선택하여 사용한다면 예측 성능을 향상시킬 수 있습니다.
