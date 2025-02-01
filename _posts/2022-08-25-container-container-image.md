---
title: "컨테이너와 이미지 | Container and Container Image"
date: 2022-08-25 17:00:00 +/-TTTT
categories: [Cloud, Google Cloud Platform (GCP)]
tags: [gcp, kubernetes, k8s]
math: true
author: seoyoung
img_path: /assets/img/for_post/
image:
  path: 20220825-t.png
  alt: ""
description: 컨테이너, 컨테이너 이미지, 구글 클라우드 컨테이너, 도커 컨테이너, 구글 클라우드 도커, Container, Container Image
---

---------------------------

> 구글 클라우드 플랫폼에서의 Container와 Container Image에 대한 개념을 정리합니다.
{: .prompt-info }

컨테이너는 가상 머신과 달리 사용자 공간만 가상화하여 가볍고, 빠르게 생성 및 실행할 수 있으며, 독립된 환경에서 앱을 실행할 수 있는 기술입니다.

Docker와 같은 소프트웨어를 사용하여 컨테이너 이미지를 빌드하고, 컨테이너 레지스트리에서 이미지를 다운로드하여 사용할 수 있습니다.

&nbsp;
&nbsp;
&nbsp;

## **Containers**
- Dedicated Server (애플리케이션 코드, 종속 항목, 커널, 하드웨어)    
➔ Virtual Machine (애플리케이션 코드, 종속 항목, 커널, 하드웨어 + 하이퍼바이저)
  - 애플리케이션을 실제 컴퓨터에 배포합니다.    
    ➔ 리소스 낭비가 크고 대규모 배포와 유지보수에 많은 시간이 소요되어 가상화가 필요합니다.
  - 하이퍼바이저는 가상 머신을 생성하고 관리합니다.
- Running Multiple Apps on a Single VM    
➔ 종속 항목을 공유하는 애플리케이션들이 서로 격리되지 않는 문제 발생합니다.



### **The VM-Centric Way to Solve This Problem**
- 애플리케이션마다 전용 가상 머신을 실행합니다.
  - 각 애플리케이션에서 고유한 종속 항목을 유지 관리합니다.     
    ➔ 커널이 격리되어 있어 애플리케이션들 간에 성능에 영향을 주지 않습니다.
  - 하지만 대규모 시스템의 경우 전용 VM은 중복적이며 낭비가 발생하고, VM 시작 속도가 느립니다.



### **User Space Abstraction and Containers**
- **종속 항목 문제 해결** ㅣ 애플리케이션과 종속 항목 수준에서 추상화를 구현합니다.
  - 전체 머신이 아니라 사용자 공간만 가상화합니다.
  - **사용자 공간** ㅣ 커널 위에 있는 모든 코드로, 애플리케이션과 종속 항목을 포함합니다.
  - 운영체제 전체를 실행하지 않아 가벼우며, 빠르게 만들고 종료할 수 있습니다.
  - 기본 시스템 위에서 예약하고 패키징하기 때문에 효율적입니다.     
    ➔ 컨테이너를 만듭니다.
- **Container (애플리케이션 코드, 종속 항목)** ㅣ 단일 애플리케이션 코드를 생성하는 격리된 사용자 공간입니다.



### **Why Developers Like Containers**
- 애플리케이션 중심으로 확장성이 높고 고성능 애플리케이션을 제공하며, 기본 하드웨어와 소프트웨어를 전제로 작업할 수 있습니다.
- 애플리케이션을 쉽게 빌드할 수 있습니다.     
  ➔ 느슨하게 결합되고 세분화된 구성 요소를 사용한 모듈식 설계입니다.
- 애플리케이션의 종속 항목을 서로 격리할 방법이 필요합니다. 
  - 가상 머신에서 애플리케이션을 패키징하는 것은 낭비입니다.
- 개발자의 노트북에서는 잘 작동하지만 프로덕션 환경에서는 실패하는 애플리케이션의 문제를 해결할 수 있습니다.


&nbsp;
&nbsp;
&nbsp;


## **Container Image**
- **Image** ㅣ 애플리케이션과 종속 항목    
  ➔ **Container** ㅣ 실행 중인 이미지 인스턴스
  ➔ 소프트웨어를 컨테이너 이미지로 빌드하면 개발자는 애플리케이션을 손쉽게 패키징하고 제공할 수 있습니다.
  ➔ 소프트웨어가 필요합니다 (Docker).



### **Containers Use a Varied Set of Linux Technologies**
- **Processes** ㅣ 리눅스 프로세스마다 서로 분리된 고유 가상 메모리 주소 공간이 존재하며, 빠르게 생성하고 삭제할 수 있습니다.
- **Linux Namespaces** ㅣ 프로세스 ID 번호, 디렉토리 트리, IP 주소를 제어합니다. (≠ Kubernetes namespaces)
- **cgroups** ㅣ 애플리케이션이 사용 가능한 CPU 시간, 메모리, I/O 대역폭, 기타 리소스의 최대 사용량을 제어합니다.
- **Union File Systems** ㅣ 애플리케이션과 종속 항목을 간결한 최소 레이어 모음으로 캡슐화합니다.



### **Containers Are Structured in Layers**
- 컨테이너 매니페스트 파일로 이미지를 빌드합니다.
- Docker 형식의 컨테이너 이미지 ➔ **Dockerfile** (컨테이너 이미지 내부 레이어가 지정됨)
  - `FROM ubuntu:18.04` ㅣ FROM 문으로 기본 레이어를 공개 저장소에서 가져와 생성합니다.
  - `COPY ./app` ㅣ COPY 명령어로 빌드 도구의 현재 디렉토리에서 복사된 파일이 포함된 레이어를 추가합니다.
  - `RUN make /app` ㅣ RUN 명령어는 make 명령어를 사용하여 애플리케이션을 빌드하고, 빌드 결과를 세 번째 레이어에 배치합니다.
  - `CMD python /app/app.py` ㅣ 마지막 레이어는 실행 시 컨테이너 내에 실행할 명령어를 지정합니다.
- 요즘은 배포 및 실행하는 컨테이너에 애플리케이션을 빌드하는 것을 권장하지 않습니다.    
  ➔ 애플리케이션 패키징에 다단계 빌드 프로세스를 사용합니다.
- 이미지에서 새 컨테이너를 만들면, 컨테이너 런타임에서는 쓰기 가능한 레이어를 기본 레이어 위에 추가합니다 (컨테이너 레이어).



### Containers Promote Smaller Shared Images
- 여러 컨테이너가 동일한 기본 이미지에 접근 권한을 공유하면서 자체 데이터 상태를 보유합니다.
- 컨테이너를 실행하면 컨테이너 런타임에서 필요한 레이어를 가져옵니다.     
  ➔ 업데이트 시 차이 나는 항목만 복사합니다.



### How Can You Get Containers?
- 컨테이너화된 소프트웨어를 컨테이너 레지스트리에서 다운로드합니다. 
  - ex. gcr.io
  - 공개 오픈소스 이미지가 다수 포함되어 있으며, 구글 클라우드 고객은 이를 사용하여 비공개 이미지를 클라우드 IAM과 잘 통합되는 방식으로 저장할 수 있습니다.
- **Docker** ㅣ 오픈소스 Docker 명령어를 사용하여 자신만의 컨테이너를 빌드합니다.
  - 반드시 신뢰할 수 있는 컴퓨터에서 빌드해야 합니다.
- **Build Your Own Container Using Cloud Build**
  - 빌드에 필요한 소스 코드를 다양한 스토리지 위치에서 검색할 수 있습니다 (Cloud Storage, Git repo, Cloud Source Repo).
  - 빌드 단계를 구성하여 종속 항목을 가져오고 ➔ 소스 코드를 컴파일하고 ➔ 통합 테스트 (Docker 컨테이너에서 실행).
  - 빌드한 이미지를 다양한 실행 환경에 제공할 수 있습니다 (GKE, App Engine, Cloud Functions).


&nbsp;
&nbsp;
&nbsp;


---------------------
## Reference

1. [Getting Started with Google Kubernetes Engine, Coursera](https://www.coursera.org/learn/google-kubernetes-engine)

