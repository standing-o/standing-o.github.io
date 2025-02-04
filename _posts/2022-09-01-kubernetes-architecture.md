---
title: "구글 쿠버네티스의 구조에 대하여 | Google Kubernetes Architecture"
date: 2022-09-01 17:00:00 +/-TTTT
categories: [클라우드 | Cloud, 구글 클라우드 플랫폼 | GCP]
tags: [gcp, kubernetes, gke, k8s]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: 🚂 구글 쿠버네티스 엔진(GKE)의 구조와 Object Management를 알아봅시다.
---

-------------------------

> **<u>KEYWORDS</u>**     
> Object Management, Kubernetes, GKE, Object Management, Kubernetes Controller, Kubernetes Cluster
{: .prompt-info }

-------------------------

&nbsp;
&nbsp;
&nbsp;

## **쿠버네티스 구조**
- **Kubernetes Objects** ㅣ 클러스터 상태를 나타내는 지속적인 엔티티
  - **Object Spec** ㅣ 만들려는 각 객체에 대해 객체 사양을 Kubernetes에 제공
  - **Object Status** ㅣ Kubernetes가 설명하는 현재 상태
- **Containers in a Pod Share Resources**
  - **Pod** ㅣ Kubernetes에서 실행 중인 모든 컨테이너를 포함하는 기본 구성 요소입니다.
    ➔ 여러 컨테이너를 수용할 수 있는 환경을 제공합니다.
    ➔ 각 Pod는 고유한 IP 주소를 할당받으며, Pod 내의 모든 컨테이너는 동일한 네트워크 네임스페이스를 공유합니다.
- **Desired State Compared to Current State**
  - Kubernetes는 원하는 상태를 정의한 후 해당 상태를 나타내는 객체를 만들고 유지하는 방식으로 작업을 실행합니다.
    ➔ Kubernetes는 이 원하는 상태를 현재 상태와 비교하며, 클러스터의 상태를 지속적으로 모니터링하고 필요한 경우 상태를 수정합니다.



### **Cooperating Processes Make a Kubernetes Cluster Work**
- **제어영역** ㅣ Kubernetes는 제어영역을 통해 전체 클러스터를 조정합니다. 제어영역은 하나의 컴퓨터에서 실행됩니다.
- **노드** ㅣ 노드는 다른 컴퓨터로, 실제로 Pod를 실행하는 데 사용됩니다.
- **kube-APIserver** ㅣ Kubernetes에서 사용자가 직접 상호작용하는 단일 구성 요소로, 클러스터 상태를 보고 변경하는 명령을 수락합니다.
  - `kubectl` ㅣ kube-APIserver에 연결하여 통신합니다.
  - `etcd` ㅣ Kubernetes 클러스터의 데이터베이스로, 클러스터 상태를 안정적으로 저장합니다.
  - `kube-scheduler` ㅣ 이 구성 요소는 Pod를 특정 노드에 예약하는 역할을 합니다.
  - `kube-controller-manager` ㅣ 이 구성 요소는 kube-APIserver를 통해 클러스터 상태를 지속적으로 모니터링하여 클러스터 상태를 유지합니다.
  - `kube-cloud-manager` ㅣ 클라우드 제공업체와 상호작용하는 컨트롤러 관리 기능을 담당합니다.
- 각 노드는 제어영역의 일부 구성 요소를 실행합니다.


### **Google Kubernetes Engine (GKE)**
- **GKE는 모든 제어 영역 구성 요소를 관리**
  - GKE는 사용자를 대신하여 모든 제어 영역 구성 요소를 관리합니다. 
  - Kubernetes 환경에서 노드는 Kubernetes 자체가 아닌, 클러스터 관리자가 외부에서 프로비저닝합니다.
- **Node pools를 사용해 다양한 노드를 관리**
  - GKE에서는 Node pool을 사용하여 여러 종류의 노드를 관리할 수 있습니다. 
  - Node pool 수준에서 자동 노드 생성, 자동 노드 복구 및 클러스터 자동 확장을 설정할 수 있습니다.
- **Zonal 클러스터와 Regional 클러스터**
  - 앱의 여러 복제본을 배포하여 가용성을 향상시키고, 더 많은 노드를 추가할 수 있습니다.
  - GKE에서 Regional Cluster를 사용하면 클러스터가 실행 중인 앱이 여러 영역에서 가용성을 유지하도록 할 수 있습니다. 
  - 전체 컴퓨팅 영역이 다운되면, GKE는 regional cluster를 통해 앱의 가용성을 보장합니다.
- **GKE 클러스터는 비공개 클러스터로 설정할 수 있음**
  - Regional 또는 Zonal GKE 클러스터는 비공개 클러스터로 설정할 수 있으며, 이 경우 Google Cloud 제품은 클러스터의 제어 영역에 접근할 수 있습니다. 
  - 승인된 네트워크는 제어 영역에 접근할 수 있는 IP 주소 범위를 기본적으로 제공하며, 노드는 제한된 아웃바운드 접근 권한을 갖습니다. 
  - 이를 통해 다른 Google Cloud 서비스와 통신할 수 있습니다.
- **GKE 클러스터와 GKE**
  - GKE 클러스터에서는 Compute Engine 가상 머신을 사용해 노드가 프로비저닝됩니다. 
  - GKE에서는 마스터가 Google Cloud 고객에게 노출되지 않는 GKE 서비스의 추상화된 부분으로 프로비저닝됩니다.

  
### **Object Management**
- 모든 Kubernetes 객체는 고유한 이름과 고유 식별자로 구분됩니다.
- **Objects are defined in a YAML file**
  - Kubernetes에서 만들고 유지할 객체는 **manifest 파일**을 사용하여 정의합니다.
  - `apiVersion` ㅣ 객체를 만드는데 사용되는 Kubernetes API 버전을 나타냅니다.
  - `kind` ㅣ 원하는 객체의 종류를 정의합니다 (ex. Pod).
  - `metadata` ㅣ 객체를 식별하기 위한 이름, 고유 ID, namespace 등을 포함합니다.
  - `spec` ㅣ Pod의 manifest 파일에서 Pod의 container image를 정의하는 필드입니다.
  - YAML 파일은 **버전 관리 저장소**에 저장됩니다.    
  ➔ GCP 고객은 Cloud Source Repositories를 사용하여 저장할 수 있으며, 다른 GCP 리소스와 동일한 방식으로 해당 파일의 권한을 제어할 수 있습니다.
- **Object Names**
  - Kubernetes 객체를 만들 때 이름을 고유한 문자열로 지정합니다.
  - 클러스터의 수명 주기 중에 생성된 모든 객체는 Kubernetes에서 생성된 고유 ID(UID)를 할당받습니다.
  - Label은 key-value 쌍으로 객체에 태그를 달 수 있으며, 객체 생성 중이나 생성 후에 이 라벨을 사용할 수 있습니다.
- **`nginx` 웹 서버 3개를 만드는 한 가지 방법**
  - Pod 객체 3개를 선언하여 각각에 고유한 YAML 섹션을 할당합니다.
  - Kubernetes의 기본 예약 알고리즘은 사용 가능한 노드에 워크로드를 균등하게 분배하는 것을 선호합니다.
- **Allocating Resource Quotas**
  - 여러 프로젝트가 하나의 클러스터에서 실행 중일 때, **리소스 할당량을 어떻게 분배할 수 있을까요?**
  - Kubernetes는 단일 물리적 클러스터를 namespace라는 가상 클러스터로 추상화할 수 있습니다.
  - Namespace를 사용하면 클러스터 전체에 리소스 할당량을 정의하고 적용할 수 있습니다.
- **배포 객체** ㅣ 지정된 시간에 정의된 포드 집합이 실행되도록 보장합니다.
- **Namespaces**
  - **기본 namespace** ㅣ 다른 namespace가 정의되지 않은 객체가 포함됩니다.
  - **Kube-system namespace** ㅣ Kubernetes 시스템 자체에서 만든 객체가 포함됩니다.
  - **Kube-public namespace** ㅣ 모든 사용자가 읽을 수 있는 공개 객체가 포함됩니다.    
    ➔ Namespace를 만들 때, 해당 namespace에 리소스를 적용하려면 명령줄에서 namespace 플래그를 사용하거나 리소스 YAML 파일에 namespace를 지정할 수 있습니다.    
    ➔ Namespace를 사용하면 클러스터 전체에 리소스 할당량을 구현하고, 동일한 이름의 객체를 다른 namespace 내에서 사용할 수 있습니다.
- **서비스의 용도**
  - Pod에 부하 분산 네트워크 엔드포인트를 제공하며, Pod를 외부에 노출하는 방법을 선택합니다.



### **Kubernetes Architecture**
- **App 설계**
  - 앱을 설계할 때, 지연 시간을 최소화하기 위해 컨테이너가 가능한 한 서로 가까이 위치하도록 배치해야 합니다.
  - 이를 해결하려면 동일한 Pod에 컨테이너를 배치하는 것이 이상적입니다.
- **ex. Kubernetes Engine Regional Cluster 배포**
  - 첫 번째 영역의 기본 풀에 4개의 머신이 있는 새 Kubernetes Engine Regional Cluster를 배포하고 영역 수를 기본값으로 둡니다.     
    ➔ 배포된 Compute Engine 머신의 개수는 12개입니다.
- **Production 앱의 리소스 우선순위 지정**
  - Kubernetes 클러스터에서 실행 중인 프로덕션 앱이 테스트 및 스테이징 배포의 영향을 받지 않도록 하려면, namespace를 구분하여 test, staging, production을 위한 리소스 우선순위를 지정합니다.
- **Stateful 앱을 위한 스토리지 구성**
  - Stateful 앱이 pod 실패나 삭제 후에도 데이터 손실 없이 계속 실행되려면, 네트워크 기반 스토리지를 사용해 내구성 높은 볼륨을 Pod에 제공해야 합니다.
- **DaemonSet**
  - DaemonSet은 클러스터 내의 모든 노드에 배포해야 하는 새로운 로깅 및 감사 유틸리티 등을 처리하기 위해 사용됩니다.
- **App의 여러 복사본 배포 및 부하 분산**
  - 앱의 여러 복사본을 배포하여 트래픽 부하를 분산하려면, 해당 앱의 Pod를 클러스터의 production namespace에 배포하고 실행할 복제본 수를 지정하는 배포 manifest를 작성합니다.


&nbsp;
&nbsp;
&nbsp;


-------------------------
## Reference

1. [Getting Started with Google Kubernetes Engine, Coursera](https://www.coursera.org/learn/google-kubernetes-engine)


