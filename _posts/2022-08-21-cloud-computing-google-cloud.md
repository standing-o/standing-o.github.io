---
title: "클라우드 컴퓨팅과 구글 클라우드 | Cloud Computing and Google Cloud"
date: 2022-08-21 17:00:00 +/-TTTT
categories: [Cloud, Google Cloud Platform (GCP)]
tags: [gcp, kubernetes, gke, k8s]
math: true
author: seoyoung
img_path: /assets/img/for_post/
image:
  path: 20220821-t.png
  alt: ""
description: 구글 클라우드, 구글 클라우드 플랫폼, 구글 클라우드 API, Google Cloud, Compute Engine
---

--------------------------

> Compute Engine, Resource, 그리고 GCP Billing에 대한 내용을 포함합니다.
{: .prompt-info }

클라우드 컴퓨팅은 필요할 때 요청하여 사용하며, 사용한 만큼 비용을 지불하는 서비스를 말합니다. 

구글 클라우드(Google Cloud)는 4가지의 컴퓨팅 서비스를 제공하며, 지역과 존으로 구성되어 있습니다. 


&nbsp;
&nbsp;
&nbsp;


## **Cloud Computing and Google Cloud**
- 컴퓨팅 리소스가 온디맨드 셀프 서비스(On-demand Self-service)로 제공됩니다.    
  ➔ 사람의 개입 없이 필요한 처리 능력, 스토리지, 네트워크를 확보할 수 있습니다.
- 광범위한 네트워크 접근(Broad Network Access)을 제공합니다.
- 리소스 풀링(Resource Pooling)을 지원하며, 고객은 리소스의 물리적 위치를 파악할 필요가 없습니다.
- 빠른 확장성(Rapid Elasticity)을 제공합니다.
- 사용량 기반 과금(Measured Service)을 지원하며, 말 그대로 사용한 만큼만 비용을 지불합니다.



### **Google Cloud가 제공하는 서비스**
- **Compute Engine** ㅣ Google Cloud의 인프라 서비스(Infrastructure-as-a-service) 솔루션으로, 클라우드에서 주문형 가상 머신을 실행할 수 있게 해줍니다.
- **Google Kubernetes Engine (GKE)** ㅣ Google이 관리하는 클라우드 환경에서 컨테이너화된 애플리케이션을 실행할 수 있으며, 사용자에게 관리 권한을 제공합니다.    
  ➔ **컨테이너화** ㅣ 코드를 패키지화하여 이동성을 극대화하고 리소스를 효율적으로 사용할 수 있도록 하는 방식입니다.
- **App Engine** ㅣ GCP의 관리형 플랫폼 서비스(Platform-as-a-Service) 프레임워크로, 인프라를 걱정하지 않고 클라우드에서 코드를 실행할 수 있습니다.
- **Cloud Functions** ㅣ 서비스로서의 기능(Function-as-a-Service)으로, 이벤트 발생 빈도에 관계없이 이벤트에 응답하여 코드를 실행할 수 있습니다.



### **데이터베이스 솔루션 구축 방법**
- **직접 구축** ㅣ Compute Engine, Google Kubernetes Engine (GKE)을 사용하여 구성.
- **관리형 서비스 사용** ㅣ 스토리지(Storage), Cloud Bigtable, Cloud Storage, Cloud SQL, Cloud Spanner, Datastore



### **Google Cloud가 제공하는 추가 서비스**
- **빅데이터(Big Data)** ㅣ BigQuery, Pub/Sub, Dataflow, Dataproc, AI Platform Notebooks.
- **머신러닝(Machine Learning)** ㅣ Vision API, AI Platform, Speech-to-Text API, Cloud Translation API, Cloud Natural Language API

&nbsp;
&nbsp;
&nbsp;

## **Resource Management**
- Google Cloud는 멀티 리전(Multi-region), 리전(Region), 존(Zone)을 통해 리소스를 제공합니다.
  - **멀티 리전(Multi-region)** ㅣ America, Europe, Asia-Pacific으로 나뉘며, 각 리전으로 세분화됩니다.
  - **리전(Region)** ㅣ 같은 대륙 내에서 독립된 지리적 위치로 구성되며, 리전 내 네트워크 연결 속도가 빠릅니다.    
    ➔ 리전은 여러 개의 존으로 나뉩니다.
  - **존(Zone)** ㅣ 특정 지리적 위치 내에서 GCP 리소스가 배포되는 위치를 의미합니다.
- 인터넷 사용자가 Google 리소스에 트래픽을 전송하면, Google은 지연 시간이 가장 낮은 엣지 네트워크(Edge Network) 위치에서 응답합니다.



### **Zonal Resources**
- Zonal 리소스는 단일 존에서만 작동합니다.
- GCP 서비스와 리소스를 사용할 때 리소스의 지리적 위치를 지정할 수 있습니다.
- Zonal 리소스는 단일 존 내에서 실행되며, 해당 존이 사용할 수 없게 되면 리소스도 사용할 수 없게 됩니다.
  - ex. Compute Engine 가상 머신 인스턴스(VM Instance), 영구 디스크(Persistent Disk), GKE의 노드(Node)



### **Regional Resources**
- Regional 리소스는 하나의 리전 내에서 여러 존에 걸쳐 실행됩니다.    
➔ 중복 배포가 가능하여 가용성을 높일 수 있습니다.



### **Global Resources**
- 글로벌 리소스는 멀티 리전을 통해 관리됩니다.
  - ex. HTTPS 로드 밸런서(Load Balancers), VPC(Virtual Private Cloud) 네트워크



### **Resources and Projects**
- 사용자가 사용하는 GCP 리소스는 위치와 상관없이 반드시 프로젝트에 속해야 합니다.
- **프로젝트(Project)** ㅣ 항목을 논리적으로 구성하며, 고유 ID와 번호로 식별됩니다.
  - 폴더로 그룹화할 수 있습니다.
- GCP 리소스 계층 구조는 조직 내 여러 부서와 팀의 리소스를 효과적으로 관리할 수 있도록 설계되었습니다.
- **Cloud IAM (Cloud Identity and Access Management)** ㅣ 사용자가 사용하는 모든 GCP 리소스에 대해 세부적인 액세스 제어를 제공합니다.
- 상위 수준에서 적용한 정책은 하위 수준으로 상속됩니다.
  - ex. 조직 ➔ 폴더 ➔ 프로젝트 ➔ 리소스
- 결제는 프로젝트 수준에서 누적됩니다.


&nbsp;
&nbsp;
&nbsp;


## **Billing**
- GCP 결제는 GCP 프로젝트 수준에서 설정됩니다.
- 결제 계정은 하나 이상의 프로젝트에서 연결할 수 있습니다.
- 결제 계정은 매월 또는 기준액 도달 시 자동으로 청구되고 인보이스가 발행되도록 설정할 수 있습니다.
- GCP 서비스를 재판매하는 GCP 고객들은 하위 계정을 사용할 수도 있습니다.

### **How to Keep Your Billing Under Control**
- **Budgets and Alerts, Billing Export, Reports**
  - **Budgets and Alerts** ㅣ 결제 알림을 기반으로 자동으로 제어하여 예산을 관리합니다.
  - **Billing Export** ㅣ 결제 데이터를 BigQuery 데이터셋으로 전송하여 결제 세부 정보를 저장할 수 있습니다.
  - **Reports** ㅣ 지출을 모니터링하는 시각적 도구입니다.
  - **Quotas** ㅣ 리소스를 과소비하는 것을 방지하기 위해 설계된 오류나 악성 공격에 대응하는 제한입니다.
- **Quotas**
  - **비율 할당량** ㅣ 특정 시점을 기준으로 재설정되며, 예를 들어 GKE 클러스터 자체에서 받는 호출을 제한하는 방식입니다. 
    - 이 할당량은 정기적으로 재설정됩니다.
  - **배정 할당량** ㅣ 프로젝트 내에서 보유할 수 있는 리소스의 수를 제어하며, 시간이 지나도 재설정되지 않습니다.
- **측정형 서비스**
  - 사용한 리소스에 대해서만 비용을 지불하는 방식입니다.


&nbsp;
&nbsp;
&nbsp;


## **Interacting with Google Cloud**
- **Google 도구 또는 인터페이스** ㅣ GCP 리소스를 관리하고 구성하는 데 사용됩니다.



### **Ways to Interact with Google Cloud**
- **Google Cloud Console 웹 사용자 인터페이스** ㅣ GCP 리소스를 관리하는 웹 기반 GUI입니다.
  - 모든 Google Cloud 리소스를 관리할 수 있는 웹 기반 GUI를 제공합니다.
  - 간단한 마우스 클릭으로 일반적인 작업을 실행할 수 있습니다.
  - Google Cloud 프로젝트와 리소스에 대한 가시성을 제공합니다.
  - **Cloud Console 사용 방법** ㅣ console.cloud.google.com에 접속하여 GCP 콘솔에 로그인합니다.
- **Cloud SDK 및 Cloud Shell 명령줄 인터페이스**
  - **Cloud SDK** ㅣ gcloud, kubectl, gsutil, bq 등으로 주기적인 업데이트를 위한 자동화 스크립트를 작성할 수 있습니다.
  - **Cloud Shell** ㅣ 직접 접근 가능하며, gcloud의 지속적인 사용과 임시 컴퓨팅 엔진 가상 머신 인스턴스를 제공합니다.
- **Cloud Console 모바일 앱 (iOS 및 Android)**
  - Compute Engine 인스턴스에 SSH로 연결할 수 있으며, 최신 결제 정보와 알림, 맞춤형 그래프를 제공합니다.
- **REST 기반 API** ㅣ 사용자 정의 애플리케이션을 위한 REST API를 사용할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


---------------------
## Reference

1. [Getting Started with Google Kubernetes Engine, Coursera](https://www.coursera.org/learn/google-kubernetes-engine)

