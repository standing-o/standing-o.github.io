---
title: "클라우드 컴퓨팅과 구글 클라우드 | Cloud Computing and Google Cloud"
date: 2022-08-21 17:00:00 +/-TTTT
categories: [Cloud, Google Cloud Platform (GCP)]
tags: [gcp, kubernetes, gke]
math: true
author: seoyoung
img_path: /assets/img/for_post/
image:
  path: 20220821-t.png
  alt: ""
description: Google cloud, compute engine, resource, billing
---

--------------------------

> Compute engine, Resource, 그리고 GCP billing에 대한 내용을 포함합니다.
{: .prompt-info }

클라우드 컴퓨팅은 필요할 때 요청하여 사용하며, 사용한 만큼 비용을 지불하는 서비스를 말합니다. 

구글 클라우드는 4가지의 컴퓨팅 서비스를 제공하며, 지역과 존으로 구성되어 있습니다. 

&nbsp;
&nbsp;
&nbsp;

## **Cloud Computing and Google Cloud**
- Computing resources가 On-demand self-service로 제공됨   
➔ 사람 개입없이 필요한 처리 능력, 스토리지 네트워크 확보 가능
- Broad network access
- Resource pooling
➔ 고객은 resource의 물리적 위치를 파악할 필요 없음
- Rapid elasticity
- Measured service
➔ 사용한 만큼만 지불



### Google cloud offers a range of services:
- **Compute engine** : 클라우드에서 주문형 가상머신을 실행하게 해 주는 Google cloud infrastructure-as-a-service 솔루션
- **Google Kubernetes engine (GKE)** : google이 관리하는 클라우드 환경에서 컨테이너화된 어플을 실행하며 사용자에게 관리 권한을 부여   
➔ 컨테이너화 : 이동성을 극대화 하고 resource를 효율적으로 사용할 수 있도록 코드를 패키지화 하는 방식
- **App engine** : GCP의 관리형 Platform-as-a-service 프레임워크, 인프라 걱정없이 클라우드에서 코드를 실행
- **Cloud functions** : 서비스로서의 기능, 이벤트 발생 빈도에 상관없이 이벤트에 응답하여 코드를 실행



### Build your own database solution
- Compute engine, Google Kubernetes engine (GKE)



### Use a managed service
- Storage : Cloud bigtable, cloud storage, cloud SQL, cloud spanner, datastore



### Google cloud offers a range of services
- Big data : Bigquery, pub/sub, dataflow, dataproc, AI platform notebooks
- Machine learning : Vision API, AI platform, speech-to-text API, cloud translation API, cloud natural language API

&nbsp;
&nbsp;
&nbsp;

## **Resource management**
- Google cloud는 multi-region, region, zone을 통해 resource를 제공
  - Multi-region : America, Europe, Asia-Pacific ➔ divided into regions
  - Region : 같은 대륙 내에서 독립된 지리적 위치, region 내에서 네트워크 연결이 빠름   
  ➔ divided into zones
  - Zones : 집중된 지리적 위치 내에 있는 GCP resource의 배포 위치
- 인터넷 사용자가 구글 resource로 트래픽을 전송하면, google은 지연시간이 가장 낮은 edge network 위치에서 응답함



### Zonal resources operate exclusively in a single zone
- GCP service와 resource를 이용하면 resource의 지리적 위치를 지정 가능
- Zonal resource는 단일 영역 내에서 실행됨 ➔ 해당 영역이 사용 불가능해지면 resource 역시 사용 불가능
  - Compute Engine Virtual Machine instance, 영구디스크, GKE의 노드



### Regional resources span multiple zones
- 하나의 region 내에 여러 영역에 걸쳐 실행됨 ➔ 중복배포가능



### Global resources
- Multi-region을 통해 관리 가능
  - HTTPS load balancers, VPC, Virtual Private Cloud networks



### Resources sit in projects, Resources have hierarchy
- 사용자가 사용하는 GCP resource는 위치와 상관없이 프로젝트에 속해야함
  - 프로젝트 : 항목을 논리적으로 구성, 고유 ID와 번호로 식별, 폴더로 그룹핑 가능
- GCP resource 계층 구조는 조직 내 여러 부서와 팀의 resource를 관리할 수 있도록 도와줌
- Cloud IAM (Cloud identity and access management) : 사용자가 사용하는 모든 GCP resource에 세부적인 엑세스 제어를 가능케함
- 사용자가 선택한 수준에서 적용한 정책은 낮은 수준으로 상속됨 (조직 ➔ 폴더 ➔ 프로젝트 ➔ 리소스).
- 결제는 프로젝트 수준에서 누적됨


&nbsp;
&nbsp;
&nbsp;

## **Billing**
- GCP 결제는 GCP 프로젝트 수준에서 설정함
- 결제 계정은 하나 이상의 프로젝트에서 연결 가능
- 결제 계정은 매월 또는 기준액 도달 시 자동으로 청구 및 invoice 되도록 설정가능
- GCP service를 재판매하는 GCP 고객들은 하위 계정을 사용하기도함
- How to keep your billing under control :
  - Budgets and alerts, billing export, reports   
  ➔ Budgets and alerts keep your billing under control; 결제 알림 기반으로 자동화 제어   
  ➔ Billing export allows you to send your billing data to a BigQuery dataset; 결제 세부정보 저장   
  ➔ Reports is a visual tool to monitor expenditure; 지출 모니터링   
  ➔ Quotas are helpful limits; 할당량은 오류나 악성 공격으로 인해 resource를 과소비하는 것을 방지하기 위해 설계됨   
  - **할당량** <sup>Quotas</sup>
    - 비율 할당량 : 특정 시점을 기준으로 재설정, GKE 클러스터 자체에서 받는 호출을 제한하는 방식, 정기적으로 재설정
    - 배정 할당량 : 프로젝트 내에서 보유할 수 있는 resource 수를 제어, 시간이 지나도 재설정X
- 측정형 서비스 : 사용한 resource에 대해서만 비용 지불


&nbsp;
&nbsp;
&nbsp;


## **Interacting with Google Cloud**
- Google tools or interfaces : GCP resource를 관리하고 구성



### Ways to interact with Google cloud
- Google cloud console web user interface : GCP resource를 관리
  - Web-based GUI to manage all google cloud resources
  - Executes common tasks using simple mouse clicks
  - Provides visibility into google cloud projects and resources
  - Interacting with the Cloud console : `console.cloud.google.com` ➔ GCP console log-in 
- Cloud SDK and cloud shell command-line interface
  - Cloud SDK : gcloud, kubectl, gsutil, bq ➔ 주기적 업데이트를 위한 자동화 스크립트 작성
  - Cloud shell : access directly, constant availability of gcloud, ephemeral compute engine virtual machine instance
- Cloud console mobile app for iOS and android
  - SSH to conneect to compute engine instances, up-to-date billing info and alerts, customizable graphs
- REST-based API for custom applications



### Google Cloud's choices for organizing compute workloads
- Service name : description
- Kubernetes engine : a managed environment for deploying containerized applications   
➔ 컨테이너에 대한 기본적인 지원과 더불어 관리형 컴퓨팅 플랫폼을 제공
- Compute engine : a managed environment for deploying virtual machines
- App engine : a managed serverless platform for deploying applications
- Cloud functions : a managed serverless platform for deploying event-driven functions


&nbsp;
&nbsp;
&nbsp;

## Reference

1. [Getting Started with Google Kubernetes Engine, Coursera](https://www.coursera.org/learn/google-kubernetes-engine)

