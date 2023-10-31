---
title: "음성 처리 | Speech Processing 를 위한 전처리 방법"
date: 2023-10-31 17:00:00 +/-TTTT
categories: [AI Theory, Audio]
tags: [speech, audio, voice, filter-bank, mfcc, melspectrogram]
math: true
toc: true
---


------------------------
- 음성 데이터를 전처리하는 방법을 알아봅시다.
- Keyword: Speech, Filter Bank, MFCC, Melspectrogram
                          
                         
                     
                      
                  
## **Overview**
- Speech Emotion Recognition (SER)과 같은 task를 위해 머신러닝/딥러닝 모델을 개발하는 경우에, 일반적으로 음성 데이터를 MFCC와 같은 Feature로 변환하여 활용합니다.
- 음성 데이터에 대한 Feature Extraction 기법 중 주요 개념을 소개하겠습니다.

## **음성 신호란? (Audio Signal)**
- 음성 데이터는 샘플링 과정을 통해 소리로부터 수집되며 보통 sample rate와 sample data라는 두가지 요소로 구성됩니다. 
- Sample rate는 초당 샘플링 횟수를 의미하며, 높은 sample rate 일수록 더 높은 음질의 음성 파일이 저장됩니다. 
- Sample rate와 sample data로 아래의 feature들을 추출하여 음성 데이터 분석에 활용합니다.
  - **Zero Cross Rate**: 신호 내에서 0을 지나는 비율을 의미하며 신호의 주파수 분포를 알 수 있습니다.
  - **Spectral Centroid**: Spectrum의 중심을 나타내며 신호의 주파수 분포를 요약합니다.
  - **Spectral Spread**: 주파수 스펙트럼의 폭으로 주파수 특성을 설명합니다.
  - **Spectral Entropy**: 신호의 복잡성을 측정합니다.
  - **Spectral Flux**: 스펙트럼의 차이를 측정하여 신호 변화를 파악합니다.
  - **MFCC**: 신호의 스펙트럼을 인간 청각 특성에 가깝게 변환합니다.
  - **Chroma Vector**: 음의 음계에 대한 정보를 추출합니다.
  - **Chroma Deviation**: Chroma Vector 간의 변동성을 나타내어 음성 신호의 음악적 특성을 파악합니다.

- 음성 신호는 아래와 같이 시간, 진폭, 주파수 세가지 domain으로 이루어져 있습니다.
  - **Time Domain**: 음성 신호는 연속적인 샘플 값으로 표현됩니다.
  - **Amplitude Domain**: 음성 신호의 강도를 나타냅니다.
  - **Frequency Domain**: 음성 신호 주파수의 크기와 분포를 나타냅니다.

![fig1](/assets/img/for_post/20231031-0.jpg)

## **Filter Bank**
### **1. 음성 데이터 불러오기**
```python
sample_rate, signal = scipy.io.wavfile.read('test.wav') 
signal = signal[0:int(5 * sample_rate)]
```
![fig2](/assets/img/for_post/20231031-1.png)

### **2. Pre-emphasis Filter**
- $y(t) = x(t) - \alpha x(t-1)$ where $\alpha = 0.95, 0.97$
- 신호에 Pre-emphasis filter를 적용하여 고주파 대역의 진폭을 증가시키고 낮은 대역의 진폭은 감소시킵니다.
- 제한된 컴퓨팅 자원으로 개발해야했던 과거에 보편적으로 사용되었습니다. 현대에서는 잘 사용하지 않으며 mean normalization 등으로 대체합니다.
- 하지만 이 필터는 다음과 같은 장점을 가집니다.
  - 보통 고주파는 저주파에 비해 진폭이 작기 때문에 주파수 스펙트럼을 균형있게 유지합니다.
  - 푸리에 변환 중 수치적인 문제들을 피하게 해줍니다.
  - SNR (Signal-to-Noise Ratio) 를 개선할 수 있습니다. 
  

```python
pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
```
![fig3](/assets/img/for_post/20231031-2.png)

### **3. 신호를 프레임으로 분할**
- 신호는 short-time 프레임으로 분할합니다.
- 신호 내의 주파수는 시간에 따라 변하기 때문에, 신호 전체에 푸리에 변환을 사용하게 되면 시간에 따른 주파수 형태를 잃게 됩니다.
  - 따라서, 신호 내 주파수가 short-time 동안 안정적이라고 가정합니다.
  - short-time 프레임에 대해 푸리에 변환을 수행하여 인접한 프레임을 연결합니다.
  - 이를 통해 전체 신호의 주파수 형태를 근사할 수 있습니다.
- 음성 처리에서의 전형적인 프레임 크기는 20ms에서 40ms 입니다. 

```python
frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z)

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

print("Shape of Frames:", np.shape(frames))
```
```shell
Shape of Frames: (328, 1200)
```

### **4. Window 함수 적용**
- Hamming Window 함수 적용

$w[x] = 0.54 - 0.46 \cos (\frac{w\pi n}{N-1})$ where $0<=n<=N-1,$
$\text{N is the window length.}$

```python
frames *= np.hamming(frame_length)
```
- 이를 통해 FFT가 데이터가 무한하다고 가정하는 것을 보정하고 주파수 영역에서의 정보를 누출 (Leakage)을 줄입니다.

### **5. 각 프레임에 대해 Short-time Fourier Transform (STFT)를 수행하여 Power Spectrum을 계산합니다.**
- 고속 푸리에 변환 (FFT, Fast Fourier Transform)을 통해 음성 신호를 주파수 도메인으로 변환하여 스펙트럼을 얻을 수 있습니다. 
- 각 프레임에 N-point FFT(Fast Fourier Transform) (=STFT) 를 수행하여 주파수 스펙트럼, 즉 Short-Time Fourier Transform (STFT)을 계산합니다.
- N은 일반적으로 256, 512 값을 세팅합니다.
- Power Spectrum (Periodogram)은  다음과 같이 계산됩니다.
  - $P = \frac{\| \text{FFT} (x_i) \| ^2}{N}$

```python
NFFT = 256
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
```

### **6. Filter Bank를 계산합니다.**
- Mel Spectrum은 사람의 청각 시스템이 높은 주파수보다 낮은 주파수 대역에서 민감하다는 특성을 반영하여 설계된 스펙트럼 표현 방법입니다.
  - 이때 사용되는 Mel Scale 값은 물리적인 주파수와 사람이 인식하는 주파수 간의 관계를 나타내는 척도로 사용됩니다.
- Filter Bank를 추출하기 위해 주로 Mel Scale에서 파생된 40개의 삼각형 필터를 Power Spectrum에 적용합니다.
- 헤르츠(Hertz, f)와 멜(Mel, m) 사이의 변환은 다음과 같은 방정식을 사용하여 수행할 수 있습니다:
  - $m = 2595\log_{10}(a + \frac{f}{700})$ 
  - $f = 700(10^{m/2595}-1)$

- Filter Bank 내 각 필터는 삼각형 모형이며, 중심 주파수에서의 response가 1이고 주변 두 필터의 중심 주파수에 도달할 때 까지 선형적으로 감소하여 response가 0이 됩니다.

![fig4](/assets/img/for_post/20231031-3.jpg)

```python
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB
```

- Power Spectrum을 Mel Scale에서의 Filter Bank로 변환하여 주파수 대역을 더 구체적으로 분석 가능 합니다.

![fig5](/assets/img/for_post/20231031-4.png)


### **7. Mean Normalization**
- Mel-scaled Filter Bank 특성을 원한다면 해당 과정은 스킵하셔도 됩니다.
- 스펙트럼을 균형있게 해주며 SNR을 향상시키기 위해 각 계수의 평균을 모든 프레임에서 빼는 계산을 수행합니다.

```python
filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
```

## **MFCC (Mel-Frequency Cepstral Coefficients)**
### **1. 이전 단계에서 계산한 Filter Bank에 Discrete Cosine Transform (DCT)를 적용한 후 결과로 나온 계수들 중 일부만 보존합니다.**
  - Filter Bank는 높은 상관관계를 가질 수 있는데 이는 머신러닝/딥러닝 알고리즘에 치명적입니다.
  - 따라서, Filter Bank 계수의 상관관계를 제거하고 Filter Bank의 압축된 표현을 DCT를 통해 얻습니다.
  - 일반적으로 자동 음성 인식 (ASR)에서 결과 Cepstral 계수 중 2에서 13까지의 계수를 보존하고 나머지는 버립니다.
  - 이를 통해 미세한 세부사항이 아닌 주요 정보만을 보존하여 모델 성능을 향상시킬 수 있습니다.

```python
num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]  # Keep 2-13
```

### **2. MFCC내 큰 계수들을 강조하거나 무시하는 데  Sinusoidal liftering을 적용할 수 있습니다.**
- 노이즈가 있는 환경에서 더 정확한 음성 인식이 가능합니다.

```python
cep_lifter = 22
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift  #*

print("Shape of MFCCs:", np.shape(mfcc))
```

```shell
Shape of MFCCs: (328, 12)
```

![fig6](/assets/img/for_post/20231031-5.png)

### **3. Mean Normalization**

```python
mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
```



## **음성 데이터로 간단하게 데이터 전처리 하기**
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
- 이러한 기법들을 비교하면서 모델에 맞는 특성을 선택해서 사용하면 예측 성능을 향상시킬 수 있습니다.


#### **References**
```shell
[1]
@misc{fayek2016,
  title   = "Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between",
  author  = "Haytham M. Fayek",
  year    = "2016",
  url     = "https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html"
}

[2] Speech Emotion Recognition, https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
```
