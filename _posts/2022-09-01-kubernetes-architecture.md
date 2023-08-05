---
title: "Google Kubernetes의 구조"
date: 2022-09-01 17:00:00 +/-TTTT
categories: [Google Cloud, Kubernetes]
tags: [gcp, kubernetes, gke]
use_math: true
---



-------------------------

- 본 포스팅은 GKE의 구조 및 object management에 대한 내용을 포함하고 있습니다.
- Keyword : Kubernetes, GKE, object management



# **Kubernetes architecture**
- Kubernetes objects : persistent entities representing the state of the cluster
  - Object spec : 만들려는 각 객체에 대해 객체 사양을 kubernetes에 제공
  - Object status : current state described by kubernetes
- Containers in a Pod share resources
  - Pod : 표준 kubernetes 모듈의 기본 구성요소, kubernetes 시스템에서 실행 중인 모든 컨테이너   
  ➔ Container가 위치한 환경을 구현하며 해당 환경은 하나 이상의 container 수용가능   
  ➔ Kubernetes는 각 pod에 고유한 IP 주소를 할당; pod 내의 모든 container는 네트워크 namespaces를 공유   
- Desired state compared to current state
  - Kubernetes가 원하는 상태를 지정했다고 가정했을 때, 해당 상태를 나타내는 하나 이상의 객체를 만들고 유지하도록 kubernetes에 지시하여 작업을 실행   
    ➔ kubernetes는 원하는 상태를 현재 상태와 비교 ➔ kubernetes의 제어영역이 cluster 상태를 지속적으로 모니터링하여 상태를 수정



### **Cooperating processes make a kubernetes cluster work**
- 제어영역 : 한 컴퓨터 ➔ 전체 클러스터를 조정 
- 노드 : 다른 컴퓨터 ➔ pod를 실행
- kube-APIserver : 사용자가 직접 상호작용하는 단일 구성요소 ➔ cluster 상태를 보거나 변경하는 명령어를 수락하는 것   
➔ `kubectl` : kube-APIserver에 연결, 통신   
➔ `etcd` : cluster의 데이터베이스 ➔ cluster 상태를 안정적으로 저장   
➔ `kube-scheduler` : pod를 노드에 예약   
➔ `kube-controller-manager` : kube-APIserver를 통해 cluster 상태를 지속적으로 모니터링   
➔ `kube-cloud-manager` : 기본 cloud 제공업체와 상호작용하는 컨트롤러 관리   
- 각 노드는 제어 영역 구성요소의 작은 그룹도 실행



### **Google kubernetes engine**
- GKE manages all the control plane components
  - GKE는 사용자를 대신하여 모든 제어 영역 구성요소를 관리
  - 모든 kubernetes 환경에서 노드는 kubernetes 자체가 아닌 cluster 관리자가 외부에서 만듬
- Use node pools to manage different kinds of nodes
  - Node pool은 GKE 기능    
  ➔ node pool 수준에서 자동 노드 생성, 자동 노드 복구 cluster 자동 확장을 사용 설정
- Zonal versus regional clusters
  - 더 많은 노드를 추가하고 app의 여러 복제본을 배포하면 app의 가용성이 일정 수준까지 향상
  - 전체 컴퓨팅 영역이 다운된다면?   
  ➔ GKE regional cluster 사용 ➔ app의 가용성이 단일 region 내의 여러 영역에서 유지되도록함
  - GKE에서 regional cluster을 구성하려는 목적 : cluster에서 실행 중인 app이 영역 손실을 견뎌낼 수 있도록 하기위함
- A regional or zonal GKE cluster can also be set up as a private cluster
  - Google cloud 제품이 cluster 제어 영역에 access 가능
  - 승인된 네트워크는 기본적으로 제어 영역에 access 하도록 신뢰를 주는 IP 주소 범위
  - 노드는 제한된 outbound access 권한을 비공개 google access를 통해 보유 가능 ➔ 다른 google cloud 서비스와 통신가능

- GKE cluster vs GKE
  - GKE cluster에서는 compute engine 가상머신으로 노드가 프로비저닝
  - GKE에서는 마스터가 google cloud 고객에게 노출되지 않는, GKE 서비스의 추상화 부분으로 프로비저닝



### **Object management**
- 모든 kubernetes 객체는 고유한 이름과 고유 식별자로 구분됨
- Objects are defined in a YAML file
  - Kubernetes가 만들고 유지할 객체를 manifest 파일을 사용하여 정의
  - `apiVersion` : 객체를 만드는데 사용되는 kubernetes API 버전을 나타냄
  - `kind` : 원하는 객체를 정의 (pod)
  - `metadata` : 객체를 식별가능하도록 이름, 고유 ID, namespace를 사용
  - `spec` : pod의 manifest 파일에서 pod의 container image를 정의하는 필드
  - YAML 파일은 버전 관리 저장소에 저장   
  ➔ GCP 고객은 cloud source repositories를 사용 ➔ 다른 GCP resource와 동일한 방식으로 해당 파일의 권한을 제어가능
- Object names
  - Kubernetes 객체를 만들 때 이름을 고유한 문자열로 지정
  - Cluster의 수명 주기 중에 만든 모든 객체에는 kubernetes에서 생성된 고유 ID(UID)가 할당
  - Label은 key-value 쌍이며, 이를 사용하여 생성 중이나 생성 후에 객체를 태그
- `nginx` 웹 서버 3개를 만드는 한 가지 방법
  - Pod 객체 3개를 선언 ➔ 각 pod에 고유한 YAML 섹션이 있음
  - kubernetes의 기본 예약 알고리즘은 사용 가능한 노드에 워크로드를 균등하게 분산하는 것을 선호
- Allocating resource quotas
  - Multiple projects run on a single cluster ➔ How can I allocate resources quotas?
  - Kubernetes를 사용하면 단일 물리적 cluster를 namespace라고 하는 여러 가상 cluster로 추상화 가능
  - Namespace를 사용하면 cluster 전체에 resource 할당량을 적용 가능 ➔ resource 사용량 한도를 정의
- 배포 객체 : 지정된 시간에 정의된 포드 집합이 실행되도록함
- Namespaces
  - 기본 namespace : 다른 namespace가 정의되지 않은 객체를 포함
  - Kube-system namespace : kubernetes 시스템 자체에서 만든 객체를 포함.
  - Kube-public namespace : 모든 사용자가 읽을 수 있도록 공개된 객체를 포함   
➔ Namespace를 만들 때 namespace에 resource를 적용하려면 명령줄 namespace 플래그를 사용하거나 resource에 대한 YAML파일에 namespace를 지정가능   
➔ Namespace를 사용하면 cluster 전체에 resource 할당량을 구현가능, 서로 중복되는 객체 이름 사용가능
- 서비스의 용도
  - Pod의 부하 분산 네트워크 엔드포인트 제공, 포드 노출 방법 선택



### **Kubernetes architecture**
- App을 설계하고 있으며 지연 시간을 최소화하기 위해 container가 가능한 한 서로 가까이 있기를 원함 ➔ 동일 pod에 container 배치
- (EX) 첫 번째 영역의 기본 풀에 4개의 머신이 있는 새 kubernetes engine regional cluster를 배포하고 영역 수를 기본값으로 둠.    
➔ 계정에 대해 배포되는 compute engine machine 개수 : 12
- Kubernetes cluster에서 실행 중인 프로덕션 app이 test 및 staging 배포의 영향을 받지 않는지 확인해야함 ➔ production app을 위한 resource의 우선순위를 지정하려면? : test, staging, production을 위한 namespace를 구성하고, test 및 staging namespace에 특정된 kubernetes resource 할당량을 구성
- Stateful app을 위한 스토리지를 구성할 때 pod가 실패하거나 삭제되더라도 app의 데이터가 삭제되지 않도록 container 내부에 파일 시스템 스토리지를 제공하려면? : 네트워크 기반 스토리지를 사용해 볼륨을 생성하여 pod에 원격으로 내구성 높은 스토리지를 제공하고 이를 pod에 지정
- DaemonSet : cluster 내의 모든 nod에 배포해야하는 새로운 로깅 및 감사 유틸리티를 처리하기 위함
- App의 여러 복사본을 배포하여 이 복사본 전체에 트래픽 부하를 분산하려고 합니다. 이 app의 pod를 cluster의 production namespace에 배포하는 방법: 실행할 복제본 수를 지정하는 배포 manifest 제작


-------------------
## **Summary**
- Kubernetes controllers keep the cluster state matching the desired state.
- Kubernetes consists of al family of control plane components, running on the control plane and the nodes.
- GKE abstracts away the control plane.
- Declare the state you want using manifest files.




----

#### **References**
```
[1] Getting Started with Google Kubernetes Engine, Coursera
```

