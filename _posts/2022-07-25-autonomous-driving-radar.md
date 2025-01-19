---
title: "자율주행과 레이더센서 | Autonomous Driving"
date: 2022-07-25 17:00:00 +/-TTTT
categories: [Domain Knowledge]
tags: [lg-aimers, radar, autonomous-driving]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: 자율주행, 레이더센서, 자율주행 딥러닝, 자율주행 AI, 레이더센서 딥러닝, 레이더센서 AI, Autonomous Driving, Radar
---



------------------

> 자율주행과 레이더에 대한 기술과 최근 동향을 소개합니다.
{: .prompt-info }

미래 모빌리티는 자율주행, 연결성, 전기화 기술을 중심으로 발전하고 있으며, 자율주행 시장은 센서 기술의 성장과 함께 미래 전망이 밝아지고 있습니다. 

레이더 기술은 거리 및 속도 측정을 통해 차량 인식과 안전성을 향상시키는 중요한 기술입니다.

&nbsp;
&nbsp;
&nbsp;

## **자율주행 시장동향**

### **미래 모빌리티 메가 트렌드**

- Autonomous Driving ➔ 운전자의 개입없이 스스로 안전하게 주행이 가능한 자율주행 고도화
- Connectivity ➔ 고도화된 연결형 자율주행을 통한 탑승자의 안전 및 교통관리 효과성 극대화
- Electrification ➔ 높은 에너지 효율성 기반 1회 충전으로 최대 주행거리 확보



### **자율주행 단계 고도화**

- **자율주행 단계**
  - 수동운전 ➔ 주행보조 ➔ 부분적 자율주행 ➔ 조건적 자율주행 ➔ 고도 자율주행 ➔ 완전 자율주행



### **자율주행 자동자 시장 동향 예측**

- 2025년 시장 점유율
  - 부분 자율주행 : 12.4% / 완전 자율주행 : 0.5%
- 2035년 시장 점유율
  - 부분 자율주행 : 15% / 완전 자율주행 : 9.8%



### **자율 주행 센서**

- **Camera**
  - 장거리 및 인식률 개선을 위한 고화소화, 픽셀 사이즈 소형화, 저조도 개선
  - 고온 동작의 품질 확보를 위한 Lens/Housing 구조 최적화
  - 생산에서 Active alignment와 calibration 공정 기술 차별화
- **Radar**
  - 고해상도 4D Imaging radar 구현을 위한 안테나 및 신호처리 S/W 기술 발전
  - Perception SW 고도화로 사물의 형상구분 및 상황예측까지 성능 발전
  - 생산에서는 평탄도 관리 및 Calibration 및 EOL 공정 기술 고도화
- **LiDAR**
  - ADAS용 LiDAR는 차량신뢰성, 디자인, Cost 우선 순위로 진화
  - Lv4/5를 위해 Redundancy를 고려한 Sensor fusion 핵심 부품으로 성장



### **자율주행 SoC 동향**

- **Tesla**
  - 카메라 2D 이미지만으로 실시간 3D 이미지 합성하는 기술
  - Edge case 중심의 서버를 통한 딥러닝과 시뮬레이션으로 정확도 향상
- **엔비디아**
  - ADAS 시스템에서 자율주행용 Hyperion 시스템 발전
  - 2D 카메라 중심에서 초음파, LiDAR, Radar 병행하는 3D 방식으로 전환
- **모빌아이**
  - 자율주행 EyeQ 시리즈 + 인포테인먼트 인텔 Atom C3000 솔루션
  - SD맵과 HD맵의 하이브리드 방식인 Autonomous vehicle 방식



#### Remark

- 미래 모빌리티 메가 트렌드는 A.C.E
- 자율주행 단계는 현재 Lv3, 2025년 Lv4, 2030년 이후 Lv5 완전 자율 주행화 예상
- 자율주행 완성차는 2035년까지 CAGR 3%, 자율주행 센서는 CAGR 7% 성장 예상
- 자율주행 센서는 카메라, 레이다, 라이다, 5G C-V2X 통신, 오디오 등이 필요
- 기존 개별 센서의 역량의 한계를 극복하기 위해 센서 Pod 기술로 발전
- 자율주행 솔루션 업체별 Lv4/Lv5의 상용차 중심의 자율주행을 개발중

&nbsp;
&nbsp;
&nbsp;

## **Radar**

- Radio detection and ranging
- Radio wave를 이용한 사물 감지 기술
- 차량 radar는 차량/보행자/도로 인프라를 인식하여 차량과의 거리, 상대속도, 각도, 높이 등의 정보를 수집



### **거리 측정**

- Measure the time of flight (ToF) in order to calculate the distance : $$d = \frac{c_0t}{2}$$
- With c<sub>0</sub> being the speed of light and t the ToF



### **속도 측정**

- Pulsed radar ㅣ Two succesive measurements
- FMCW radar ㅣ Exploit the Doppler shift



### **Radar 필요 기술**

- **Antenna**
  - High gain, 광각, 고해상도, peak gain, 방사패턴 최적화, array 안테나 설계
- **mmWave** 회로
  - 저손실, EMC 최소화 설계, Main IC 기반 플랫폼 설계, Transition 최소화 및 RF 매칭
- **SW**
  - System SW, radar 신호처리, perception 알고리즘
- **기구**
  - Radome 전파 투과율 최적화, 고신뢰성 및 방수, 방진, 방열 설계, simulation



### **Radar 기술 동향**

- 2D ADAS Basic (X, Y, Doppler) ➔ 2D ADAS improved (X, Y, Doppler) ➔ 3D (X, Y, Z, Doppler) ➔ 4D HR (High resolution; X, Y, Z, Doppler, depth)
 - 4D UHR (Ultra high resolution; X, Y, Z, Doppler, depth) ➔ Imaging (X, Y, Z, Doppler, depth, AI/Deeplearning)



### **Radar 시장 동향**

- 차량 제어를 위해 AEB 기능 채용 확대
- Front radar의 고해상도로 채용률 성장
- Corner radar의 low cost ➔ 차량당 4개 이상 적용되어 360도 서라운드 센싱
- 안전과 편의 기능으로 강화를 위한 In-cabin용 신규 application 개화



#### Remark

- Radar 필요기술은 안테나, mmWave 회로, SW, 기구, PCB, 공정설계
- Radar 종류는 SRR, MRR, LRR이고 향후 4D Imaging radar로 고도화
- 차량용 radar는 Infineon, TI, NXP가 주로 사용

&nbsp;
&nbsp;
&nbsp;


----------
## Reference
> 본 포스팅은 LG Aimers 프로그램에서 학습한 내용을 기반으로 작성되었습니다. (전체 내용 X)
{: .prompt-warning }

1. LG Aimers AI Essential Course Module 6. 자율주행과 레이더센서의 이해, LG이노텍 김경석 연구위원


