---
title: "구글 쿠버네티스 엔진이란? | Google Kubernetes Engine (GKE)"
date: 2022-08-28 17:00:00 +/-TTTT
categories: [Cloud, Google Cloud Platform (GCP)]
tags: [gcp, kubernetes, gke, k8s]
math: true
author: seoyoung
img_path: /assets/img/for_post/
image:
  path: 20220828-t.png
  alt: ""
description: 구글 쿠버네티스, 쿠버네티스, 쿠버네티스 컨테이너, k8s, 쿠버네티스 시작하기, 쿠버네티스 도커, Google Kubernetes, Cloud Function, Cloud Run
---



--------------------------

> 구글 쿠버네티스(Google Kubernetes)와 Cloud Function 및 Cloud Run을 설명합니다.
{: .prompt-info }

쿠버네티스와 구글 쿠버네티스 엔진(GKE)의 기능을 비교하면서, 
앱 배포 시 리소스를 효율적으로 활용하고 이동성이 우수한 독립적이고 경량화된 패키지 형태로 컨테이너를 선택하는 방법을 소개합니다.

&nbsp;
&nbsp;
&nbsp;

## **Kubernetes**
- 컨테이너 인프라를 더 효과적으로 관리하기 위해 사용합니다.
- 컨테이너 인프라를 온프레미스(on-premise) 또는 클라우드에서 조정하고 관리할 수 있습니다.
- 컨테이너 중심의 관리 환경을 제공합니다.
  - **자동화(Automation)** ㅣ 컨테이너화된 애플리케이션의 배포, 확장, 부하 분산, 로깅, 모니터링, 기타 관리를 자동화하여 플랫폼 서비스(Platform as a service)로 제공합니다.
  - **인프라 서비스(Infrastructure as a Service)** ㅣ 다양한 사용자 환경 설정과 구성의 유연성을 지원합니다.
  - **선언적 구성(Declarative Configuration)** ㅣ 인프라를 선언적으로 관리하며, 원하는 시스템 상태를 항상 문서화합니다.
    - Kubernetes를 사용할 때 사용자가 원하는 상태를 설명하면, Kubernetes는 배포된 시스템을 해당 상태로 유지하며 장애가 발생하더라도 지속적으로 해당 상태를 유지합니다.
  - **명령형 구성(Imperative Configuration)** ㅣ 명령어를 실행하여 시스템 상태를 변경하는 방식을 제공합니다.

### **Kubernetes Features**
- 상태 저장(Stateful) 애플리케이션과 상태 비저장(Stateless) 애플리케이션을 모두 지원합니다. 
  - **상태 저장 애플리케이션**은 데이터를 지속적으로 저장해야 하는 애플리케이션을 의미합니다.
- **자동 확장(Autoscaling)** ㅣ 리소스 사용률에 따라 컨테이너화된 애플리케이션을 자동으로 수평 확장하거나 축소할 수 있습니다.
- **리소스 제한(Resource Limits)** ㅣ 리소스를 제어하여 클러스터 내 전반적인 워크로드 성능을 개선합니다.
- **확장성(Extensibility)** ㅣ 플러그인과 부가 기능을 통해 기능을 확장할 수 있습니다.
- **이식성(Portability)** ㅣ 온프레미스(On-premise), GCP, 다양한 클라우드 서비스 제공업체 간 워크로드 이동성을 제공합니다. 
  - 이를 통해 공급업체에 종속되지 않습니다.

&nbsp;
&nbsp;
&nbsp;

## **Google Kubernetes Engine <sup>GKE</sup>**
- Kubernetes는 강력하지만, 인프라를 관리하는 데 많은 시간이 소요됩니다.    
  ➔ 구글 클라우드 플랫폼(GCP)에서 이를 보완하기 위한 기능을 제공하는 것이 GKE입니다.
- GKE를 사용하면 GCP에서 컨테이너화된 애플리케이션을 구축하기 위해 Kubernetes 환경을 손쉽게 배포, 관리, 확장할 수 있습니다.
- GKE는 워크로드를 쉽게 배포할 수 있도록 지원합니다.


### **GKE Features**
- **완전 관리형(Fully Managed)** ㅣ 기본 리소스를 프로비저닝(Provisioning)할 필요가 없습니다.
- **컨테이너 최적화 OS(Container-optimized OS)** ㅣ Google이 관리하는 운영체제로, 빠른 확장에 최적화되어 있습니다.
- **자동 업그레이드(Auto upgrade)** ㅣ 최신 버전의 Kubernetes를 유지합니다.
- **클러스터(Cluster)** ㅣ 인스턴스화된 Kubernetes 시스템을 의미합니다.
- **자동 복구(Auto Repair)** ㅣ 서비스가 비정상적인 노드를 자동으로 복구합니다.
- **노드(Node)** ㅣ GKE 클러스터 내에서 컨테이너를 호스팅하는 가상 머신을 의미합니다.
- **클러스터 확장(Cluster Scaling)** ㅣ 클러스터를 자동으로 확장하거나 축소할 수 있습니다.
- **원활한 통합(Seamless integration)** ㅣ GKE는 Google Cloud의 Cloud Build 및 Container Registry와 원활하게 통합됩니다.
- **ID 및 액세스 관리(Identity and Access Management)** ㅣ 사용자와 리소스에 대한 권한을 관리할 수 있습니다.
- **통합 로깅 및 모니터링(Integrated Logging and Monitoring)** ㅣ Stackdriver를 통해 Google Cloud 시스템 서비스, 컨테이너, 애플리케이션, 인프라를 모니터링하고 관리할 수 있습니다.    
  ➔ GKE는 Stackdriver Monitoring과 통합되어 애플리케이션 성능을 효과적으로 파악할 수 있습니다.
- **통합 네트워킹(Integrated Networking)** ㅣ GKE는 Virtual Private Cloud(VPC)와 통합되며, GCP의 고급 네트워킹 기능을 활용합니다.
- **클라우드 콘솔(Cloud Console)** ㅣ GKE 클러스터 및 리소스에 대한 정보를 제공하며, 클러스터의 리소스를 확인, 검사, 삭제할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;

## **Computing Options**
- 완전히 사용자 지정이 가능한 가상 머신을 제공합니다.
- 영구 디스크와 선택적으로 사용할 수 있는 로컬 SSD를 지원합니다.
- 글로벌 로드 밸런싱과 자동 확장을 지원합니다.
- 초 단위 과금(Per-second Billing)을 제공합니다.


### **Compute Engine Use Cases**
- 운영 체제와 가상 하드웨어에 대한 완전한 제어 권한을 제공합니다.
- 기존 워크로드를 클라우드로 이전하는 리프트 앤드 시프트(Lift-and-shift) 마이그레이션에 적합합니다.
- 가장 유연한 컴퓨팅 솔루션으로, 관리형 솔루션이 제약이 많을 때 주로 사용됩니다.



### **App Engine**
- 완전 관리형의 코드 우선(Code-first) 플랫폼을 제공합니다.
- 애플리케이션 배포와 확장성을 간소화합니다.
- 인기 있는 프로그래밍 언어와 애플리케이션 런타임을 지원합니다.
- 통합된 모니터링, 로깅, 진단 기능을 제공합니다.
- 버전 관리, 카나리아 테스트(Canary Testing), 롤백을 간단하게 처리할 수 있습니다.
- ex. 웹 사이트, 모바일 앱 및 게임 백엔드, RESTful API 개발에 적합합니다.


### **Google Kubernetes Engine**
- 완전 관리형 Kubernetes 플랫폼을 제공합니다.
- 클러스터 확장, 영구 디스크, 자동 업그레이드, 자동 노드 복구를 지원합니다.
- Google Cloud 서비스와 내장 통합을 제공합니다.
- 여러 환경 간 이식성을 지원합니다.
- 하이브리드 컴퓨팅 및 멀티 클라우드 컴퓨팅을 지원합니다.
- ex. 컨테이너화된 애플리케이션, 클라우드 네이티브 분산 시스템, 하이브리드 애플리케이션


### **Cloud Run**
- 웹 요청 또는 Cloud Pub/Sub 이벤트를 통해 상태 비저장(Stateless) 컨테이너를 실행할 수 있는 관리형 컴퓨팅 플랫폼입니다.
- **서버리스(Serverless)** ㅣ 모든 인프라 관리를 추상화하여 애플리케이션 개발에만 집중할 수 있습니다.
  - 인프라 관리를 추상화하여 요청 또는 이벤트 기반의 상태 비저장 워크로드를 서버 걱정 없이 실행할 수 있습니다.
- 자동으로 확장 및 축소를 수행합니다.
- 오픈 API와 런타임 환경을 지원하며, 일관된 개발자 환경이 포함된 상태 비저장 컨테이너를 완전 관리형 환경 또는 자체 GKE 클러스터에 배포할 수 있도록 선택할 수 있습니다.
- ex. HTTP 요청이나 이벤트를 수신 대기하는 상태 비저장 컨테이너를 배포할 수 있습니다.
  - 모든 언어와 프레임워크, 도구를 사용해 애플리케이션을 빌드할 수 있습니다.


### **Cloud Functions**
- 이벤트 기반(Event-driven) 서버리스 컴퓨팅 서비스로, JavaScript, Python, 또는 Go로 작성된 코드를 업로드하기만 하면 GCP가 적합한 컴퓨팅 용량에 자동으로 배포합니다.
- 자동 확장과 고가용성, 내결함성 디자인을 지원합니다.
- 코드를 실행할 때만 과금이 발생합니다.
- Google Cloud 서비스, HTTP 엔드포인트, Firebase의 이벤트를 기반으로 실행됩니다.
- ex. **마이크로서비스 아키텍처 지원**
  - 서버리스 애플리케이션 백엔드 ㅣ 모바일 및 IoT 백엔드, 서드파티 서비스 및 API와 통합.
  - 지능형 애플리케이션 ㅣ 가상 비서와 챗봇, 동영상 및 이미지 분석.


&nbsp;
&nbsp;
&nbsp;


---------------------------------
## Reference

1. [Getting Started with Google Kubernetes Engine, Coursera](https://www.coursera.org/learn/google-kubernetes-engines)
