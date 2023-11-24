---
title: "구글 쿠버네티스 엔진이란? | Google Kubernetes Engine(GKE)"
date: 2022-08-28 17:00:00 +/-TTTT
categories: [Cloud, Google Cloud Platform (GCP)]
tags: [gcp, kubernetes, gke]
math: true
author: seoyoung
img_path: /assets/img/for_post/
image:
  path: 20220828-t.png
  alt: ""
description: Google Kubernetes, Cloud Function, Cloud Run
---



--------------------------

> Google Kubernetes와 Cloud Function 및 Cloud Run을 설명합니다.
{: .prompt-info }

Cloud Build를 활용하여 컨테이너를 생성하고, 해당 컨테이너를 컨테이너 레지스트리에 저장합니다. 

쿠버네티스와 구글 쿠버네티스 엔진의 기능을 비교하여, 앱을 배포할 때 리소스를 효율적으로 활용하고 이동성이 뛰어난 독립적인 경량 패키지로 선택하기 위해 컨테이너를 고려합니다. 

&nbsp;
&nbsp;
&nbsp;

## **Kubernetes**
- Container 인프라를 더 효과적으로 관리하기 위함
- **Kubernetes** : container 인프라를 on-premise 또는 클라우드에서 조정, 관리 가능
  - Container 중심의 관리 환경
  - Automation : Container화된 app의 배포, 확장, 부하 분산, 로깅, 모니터링, 기타 관리 등을 자동화   ➔ platform as a service
  - Infrastructure as a service : 다양한 사용자 환경설정과 구성 유연성을 지원
  - Declarative configuration : 인프라를 선언적으로 관리 ➔ 원하는 시스템 상태가 항상 문서화됨   
    ➔ Kubernetes를 사용할 때 원하는 상태를 설명하면 Kubernetes는 배포된 시스템을 원하는 상태에 맞게 만들고 장애가 발생해도 상태를 유지
  - Imperative configuration : 명령형 구성으로 명령어를 실행하여 시스템 상태를 변경



### Kubernetes features
- Supports both stateful and stateless applications : 데이터를 지속적으로 저장해야하는 app
- Autoscaling : resource 사용률에 따라 container화된 app을 자동으로 수평확장/축소 가능
- Resource limits : resource를 제어하여 클러스터 내의 전반적인 워크로드 성능 개선
- Extensibility : 플러그인, 부가기능 확장
- Portability : on-premise 또는 GCP, 여러 클라우드 서비스 제공업체 간 워크로드 이동성 (공급업체 제약X)


&nbsp;
&nbsp;
&nbsp;

## **Google Kubernetes engine (GKE)**
- Kubernetes is powerful, but managing the infrastructure is a full-time job   
➔ Google cloud 내에서 유용한 기능 : GKE
- GKE를 사용하면 GCP에서 container화된 app을 위해 kubernetes 환경을 배포, 관리, 확장 가능
- GKE lets you deploy workloads easily



### GKE features
- Fully managed : 기본 resource를 provisioning 할 필요X
- Container-optimized OS : Google이 관리하는 이 운영체제는 빠른 확장에 최적화
- **Auto upgrade**
  - 자동 업그레이드로 최신버전의 kubernetes 유지
  - Cluster : 인스턴스화 한 kubernetes 시스템
- **Auto repair**
  - 서비스가 비정상 노드를 자동으로 복구
  - Node : GKE 클러스터 내에서 container를 호스팅 하는 가상머신
- **Cluster scaling**
- **Seamless integration**
  - GKE는 google의 cloud build/container registry와 원활하게 통합
- **Identity and access management**
- **Integrated logging and monitoring**
  - Stackdriver : Google cloud 시스템 서비스, 컨테이너, app, 인프라를 모니터링하고 관리   
➔ GKE는 Stackdriver monitoring과 통합되어 app 성능을 파악
- **Integrated networking**
  - GKE는 virtual private cloud (VPC) 와 통합되며 GCP의 네트워킹 기능을 사용
- **Cloud console**
  - GKE 클러스터와 resource에 대한 정보를 제공하며 클러스터의 resource를 확인, 검사, 삭제 가능


&nbsp;
&nbsp;
&nbsp;

## **Computing options**
- Fully customizable virtual machines
- Persistent disks and optional local SSDs
- Global load balancing and autoscaling
- Per-second billing



### Compute engine use cases
- Complete control over the OS and virtual hardware
- Well suited for lift-and-shift migrations to the cloud
- Most flexible compute solution, often used when a managed solution is too restrictive



### App engine
- Provides a fully managed, code-first platform.
- Streamlines application deployment and scalability.
- Provides support for popular programming languages and application runtimes.
- Supports integrated monitoring, logging, and diagnostics.
- Simplifies version control, canary testing, and rollbacks.
- Use cases : websites, mobile app and gaming backends, RESTful APIs



### Google kubernetes engine
- Fully managed kubernetes platform
- Supports cluster scaling, persistent disks, automated upgrades, and auto node repairs
- Built-in integration with Google cloud services
- Portability across multiple environments
  - Hybrid computing, multi-cloud computing
- Use cases : Containerized applications, cloud-native distributed systems, hybrid applications



### Cloud run
- 웹 요청 또는 cloud put/sub 이벤트를 통해 stateless container를 실행할 수 있는 관리형 컴퓨팅 플랫폼
- Serverless : 모든 인프라 관리를 추상화 ➔ app 개발에만 집중가능
- Abstract away infrastructure management ➔ 서버 걱정없이 요청 또는 이벤트 기반 stateless 워크로드를 실행가능
- Automatically scales up and down
- Open API and runtime environment ➔ 일관된 개발자 환경이 포함된 stateless container를 완전 관리형 환경 또는 자체 GKE cluster에 배포하도록 선택가능
- Use cases : 
  - HTTP 요청을 통해 전달되는 요청이나 이벤트를 수신대기하는 stateless container를 배포가능
  - Build applications in any language using any frameworks and tools.



### Cloud functions
- Event-driven, serverless compute service ➔ 자바스크립트, python 또는 go로 작성한 코드를 업로드하기만 하면 GCP가 코드를 실행하는데 적합한 컴퓨팅 용량을 자동으로 배포
- Automatic scaling with highly available and fault-tolerant design
- Charges apply only when your code runs
- Triggered based on events in google cloud services, HTTP endpoints, and Firebase
- Use cases : 
  - Supporting microservice architecture
  - Serverless application backends : mobile and IoT backends, integrate with third-party services and APIs
  - Intelligent applications : virtual assistant and chat bots, video and image analysis


&nbsp;
&nbsp;
&nbsp;

## Reference

1. [Getting Started with Google Kubernetes Engine, Coursera](https://www.coursera.org/learn/google-kubernetes-engines)
