---
title: "구글 쿠버네티스 워크로드 이해하기 | Google Kubernetes Workload"
date: 2022-12-06 17:00:00 +/-TTTT
categories: [클라우드 | Cloud, 구글 클라우드 플랫폼 | GCP]
tags: [gcp, kubernetes, gke, k8s]
math: true
author: seoyoung
img_path: /assets/img/for_post/
description: 🕸️ 구글 쿠버네티스(Kubernetes)의 배포와 포드 네트워킹, 볼륨에 대한 내용을 공부합니다.
---

------------------------

> **<u>KEYWORDS</u>**     
> Kubernetes 배포, 쿠버네티스 배포, 쿠버네티스 포드 네트워킹, Kubernetes, GKE
{: .prompt-info }

------------------------

&nbsp;
&nbsp;
&nbsp;

## **kubectl Command**
- `Kubectl` ㅣ 관리자가 Kubernetes 클러스터를 제어하는 데 사용하는 유틸리티입니다.
- `Kubectl` **transforms your command-line entires into API calls**
  - `kubectl`은 명령줄 입력 내용을 API 호출로 변환하여 선택한 Kubernetes 클러스터 내의 kubeAPI 서버로 전송합니다.
- **Use `kubectl` to see a list of Pods in a cluster**
  - `kubectl get pods` ㅣ `kubectl`은 이 명령어를 API 호출로 변환하고 클러스터 제어 영역 서버에서 HTTPS를 통해 kubeAPI 서버로 전송합니다.
  - `kubeAPI` 서버는 `etcd`를 통해 요청을 처리하고, HTTP를 통해 `kubectl`에 결과를 반환합니다. 
    - `kubectl`은 API 응답을 해석하여 명령 프롬프트에서 관리자에게 결과를 표시합니다.
- **`kubectl` must be configured first**
  - `kubectl`은 설정 파일을 필요로 합니다 (`$HOME/.kube/config`).
  - 설정 파일은 클러스터의 이름과 자격 증명 정보를 포함합니다.
  - 현재 설정을 확인하려면 `kubectl config view` 명령어를 사용합니다.
- **Connect to a Google Kubernetes Engine Cluster**
  - `kubectl config view` ㅣ `kubectl` 명령어 자체의 구성 상태를 알려줍니다.
  - `gcloud` 명령줄 도구와 `kubectl`을 설치한 환경에서 `gcloud get credentials` 명령어를 사용하여 사용자 인증 정보를 가져올 수 있습니다.
- **The `kubectl` command syntax has several parts**
  - **Type** ㅣ 명령어가 적용되는 Kubernetes 객체를 정의합니다. `kubectl`에 어떤 작업을 어떤 객체에 수행할 것인지 알립니다.
  - **Name** ㅣ Type에 정의된 객체를 지정합니다.


&nbsp;
&nbsp;
&nbsp;

## **Deployments**
- **Deployments declare th state of Pods**
  - Pod 사양을 업데이트할 때마다, 변경된 Deployment 버전과 일치하는 새 ReplicaSet이 생성됩니다.    
    ➔ 배포는 제어된 방식으로 업데이트된 Pod를 롤아웃하는 방법을 제공합니다. 
    - 기존 Pod는 이전 ReplicaSet에서 제거되고 새 ReplicaSet의 새로운 Pod로 대체됩니다.
- Deployment는 stateless 앱용으로 설계됩니다.    
  ➔ **Stateless 앱** ㅣ 데이터나 앱 상태가 클러스터나 영구 스토리지에 저장되지 않습니다.
- **Deployment is a two-part process**
  - 원하는 상태는 Pod의 특성이 포함된 Deployment YAML 파일에 설명되며, Pod를 운영 가능하게 실행하고 수명 주기 이벤트를 처리하는 방법이 함께 제공됩니다.    
    ➔ 이 파일을 Kubernetes 제어 영역에 제출하면, 배포 컨트롤러가 생성되며, 컨트롤러는 원하는 상태를 실현하고 유지하는 역할을 합니다.
  - **배포** ㅣ 상태를 선언하는 Pod의 상위 수준 컨트롤러입니다.
- **Deployment has three different lifecycle states**
  - Progressing state, Complete state, Failed state


&nbsp;
&nbsp;
&nbsp;


## **Pod Networking**
- **Pod** ㅣ 공유된 스토리지와 네트워킹을 갖춘 컨테이너 그룹입니다.
  - Kubernetes의 'Pod별 IP' 모델을 기반으로 하며, 각 Pod에 단일 IP 주소가 할당되고, Pod 내의 컨테이너들은 해당 IP 주소를 포함하여 동일한 네트워크 네임스페이스를 공유합니다.
- **Your workload doesn't run in a single pod**
  - 워크로드는 서로 통신해야 하는 다양한 앱으로 구성됩니다.
  - 각 Pod에는 고유한 IP 주소가 있으며, 노드에서 Pod는 노드의 루트 네트워크 네임스페이스를 통해 서로 연결됩니다.     
    ➔ 해당 VM에서 Pod가 서로를 찾고 연결합니다. 
  - 루트 네트워크 네임스페이스는 노드의 기본 NIC에 연결되며, 노드의 VM NIC를 사용하여 트래픽을 전달합니다.     
    ➔ Pod의 IP 주소를 노드가 연결된 네트워크에서 라우팅할 수 있어야 합니다.
- **Node get pod IP addresses from address ranges assigned to your Virtual Private Cloud**
  - GKE에서 노드는 **Virtual Private Cloud (VPC)**에 할당된 주소 범위에서 Pod의 IP 주소를 가져옵니다.
  - VPC는 GCP 내에서 배포된 리소스에 대한 연결을 제공하는 논리적으로 격리된 네트워크입니다.     
    ➔ 이 리소스에는 Kubernetes 클러스터 Compute Engine 인스턴스, App Engine 가변형 인스턴스 등이 포함됩니다.
- **Addressing the Pods**
  - **GKE Cluster Node** ㅣ GKE가 맞춤 설정하고 관리하는 컴퓨팅 인스턴스입니다. 해당 머신이 있는 VPC 서브넷의 IP 주소가 할당됩니다.
  - VPC 기반 GKE 클러스터는 Pod에 대해 별도의 별칭 IP 범위를 생성할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;

## **Volumes**
### **Kubernetes Offers Storage Abstraction Options**
- **Volumes** ㅣ Pod에 스토리지를 연결하는 방법입니다.
  - 일부 Volumes는 일시적입니다.
  - 일부 Volumes는 지속적입니다.
- **Persistent Storage Options**
  - Block storage 또는 networked file systems로 이루어진 지속적인 스토리지 옵션을 제공합니다.
  - Pod 외부에서 내구성 있는 스토리지를 제공합니다.
  - Pod의 생애 주기와 독립적으로 존재할 수 있습니다.
  - Pod 생성 전에 존재할 수 있으며, Pod가 이를 요청하여 사용할 수 있습니다.




&nbsp;
&nbsp;
&nbsp;


----------------
## Reference

1. [Getting Started with Google Kubernetes Engine, Coursera](https://www.coursera.org/learn/google-kubernetes-engine)
