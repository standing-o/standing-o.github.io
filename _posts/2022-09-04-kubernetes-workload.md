---
title: "Google Kubernetes 워크로드"
date: 2022-12-06 17:00:00 +/-TTTT
categories: [Google Cloud, Kubernetes]
tags: [gcp, kubernetes, gke, kubectl]
use_math: true
---



------------------------

- 본 포스팅은 Kubernetes의 배포와 포드 네트워킹, 볼륨에 대한 내용을 포함하고 있습니다.
- Keyword : Kubernetes, GKE, deployment, pod networking, volume



# **Kubernetes Workload**
## **`kubectl` command**
- `Kubectl` : 관리자가 kubernetes cluster를 제어하는 데 사용하는 유틸리티
- `Kubectl` transforms your command-line entires into API calls
  - `kubectl`은 명령줄 입력 내용은 API 호출로 전환한 후 선택한 kubernetes cluster 내 kubeAPI 서버로 전송
- Use `kubectl` to see a list of Pods in a cluster
  - `kubectl get pods` : `kubectl`은 이 명령어를 API 호출로 전환하고 cluster 제어 영역 서버에서 HTTPS를 통해 `kubeAPI` 서버로 보냄
  - `kubeAPI` 서버는 `etcd` 쿼리를 통해 요청을 처리 ➔ `kubeAPI` 서버는 HTTP를 통해 `kubectl`에 결과를 반환 ➔ `kubectl`은 API 응답을 해석하여 명령 프롬프트에서 관리자에게 결과를 표시
- `kubectl` must be configured first
  - Relies on a config file : `$HOME/.kube/config`
  - Config file contains:
    - Target cluster name, credentials for the cluster
  - Current config: `kubectl` config view
- Connect to a google kubernetes engine cluster
  - `kubectl config view` : `kubectl` 명령어 자체의 구성에 대해 알려줌
  - gcloud 명령줄 도구와 `kubectl`을 설치한 다른환경에서 `get credentials gcloud` 명령어를 사용 : 사용자 인증 정보 가져오기
- The `kubectl` command syntax has several parts
  - Type : 명령어가 적용되는 kubernetes 객체를 정의, command와 함께 사용되어 어떤 작업을 어떤 type의 객체에 수행하길 원하는지 `kubectl`에 알림
  - Name : type에 정의된 객체를 지정


-----------------
## **Deployments**
- Deployments declare th state of Pods
  - Pod 사양을 업데이트 할 때마다 변경된 deployment 버전과 일치하는 새 ReplicaSet이 생성   
  ➔ 배포가 제어된 방식으로 업데이트된 pod를 롤아웃 하는 방법 ➔ 기존 pod는 이전 ReplicaSet에서 제거되고 새 ReplicaSet의 새로운 pod로 대체
  - 배포는 stateless app용으로 설계됨    
  ➔ stateless app : 데이터 또는 app 상태의 cluster나 영구 스토리지에 저장X
- Deployment is a two-part process
  - 원하는 상태는 pod의 특성이 포함된 배포 YAML 파일에 설명되어 있으며 pod를 운영 가능하게 실행하고 수명 주기 이벤트를 처리하는 방법이 함께 제공   
  ➔ 이 파일을 Kubernetes 제어 영역에 제출하면 배포 컨트롤러가 생성되며  이 컨트롤러는 원하는 상태를 실현하고 원하는 상태를 유지하는 역할을 함
  - 배포 : 상태를 선언하는 pod의 상위 수준 컨트롤러
- Deployment has three different lifecycle states
  - Progressing state, complete state, failed state


--------------
## **Pod networking**
- Pod : a group of containers with shared storage and networking
  - Kubernetes의 'pod별 IP' 모델을 기반     
  ➔ 각 pod에 단일 IP 주소가 할당되고 pod 내의 container는 해당 IP 주소를 포함하여 동일한 네트워크 namespace를 공유
- Your workload doesn't run in an single pod
  - Workload : 서로 통신해야 하는 다양한 app으로 구성됨
  - 각 pod에는 고유한 IP 주소가 있고, 노드에서 pod는 노드의 루트 네트워크 namespace를 통해 서로 연결    ➔ 해당 VM에서 pod가 서로를 찾고 연결   
  - 루트 네트워크 namespace는 노드의 기본 NIC에 연결 ➔ 노드의 VM NIC를 사용하여 루트 네트워크 namespace는 해당 노드에서 트래픽을 전달   
    ➔ Pod의 IP 주소를 노드가 연결된 네트워크에서 라우팅할 수 있어야 한다는 뜻
- Node get pod IP addresses from address ranges assigned to your virtual private cloud
  - GKE에서 노드는 Virtual Private Cloud 즉, VPC에 할당된 주소 범위에서 pod IP 주소를 가져옴
  - VPC : 는 GCP 내에서 배포하는 resource에 대한 연결을 제공하는 논리적으로 격리된 네트워크   
    ➔ 이러한 resource에는 Kubernetes cluster Compute Engine 인스턴스 App Engine 가변형 인스턴스가 있음

- Addressing the pods
  - GKE cluster node : GKE가 맞춤설정하고 관리하는 컴퓨팅 인스턴스 ➔ 해당 머신이 있는 VPC 서브넷의 IP 주소가 할당
  - VPC 기반 GKE 클러스터는 pod에 대해 별도의 별칭 IP 범위도 생성


------------------
## **Volumes**
### **Kubernetes offers storage abstraction options**
- Volumes
  - Volumes are the method by which you attach storage to a pod
  - Some volumes are ephemeral
  - Some volumes are persistent
- Persistent storage options
  - Are block storage, or networked file systems
  - Provide durable storage outside a pod
  - Are independent of the pod's lifecycle
  - May exist before pod creation and be claimed




----

#### **References**
```
[1] Getting Started with Google Kubernetes Engine, Coursera
```