---
title: "Container와 Container image"
date: 2022-08-25 17:00:00 +/-TTTT
categories: [Google Cloud, Kubernetes]
tags: [gcp, kubernetes, container, container-image]
math: true
---



---------------------------

- 본 포스팅은 container와 container image에 대한 내용들을 소개하고 있습니다.
- Keyword : Container, container image



# **Container and Container image**
## **Containers**
- Dedicated server (Application code, dependencies, kernel, hardware) ➔  Virtual machine (Application code, dependencies, kernel, hardware + hypervisor)
  - 어플리케이션을 실제 컴퓨터에 배포   
  ➔ resource 낭비가 크고 대규모 배포와 유지보수에 많은 시간이 소요 ➔ 가상화 필요
  - Hypervisors create and manage virtual machines
- Running multiple apps on an single VM   
➔ 종속 항목을 공유하는 app이 서로 격리되지 않는 문제 발생



### **The VM-centric way to solve this problem**
- App마다 전용 가상 머신을 실행
  - 각 app에서 고유 종속 항목을 유지 관리 ➔ 커널이 격리되어 있으므로 app끼리 성능에 영향X
  -  But 대규모 시스템의 경우 전용 VM은 중복적이며 낭비, VM 시작속도가 느림



### **User space abstraction and containers**
- 종속 항목 문제 해결 : app과 종속 항목 수준에서 추상화를 구현
  - 전체 머신이 아니라 사용자 공간만 가상화
  - 사용자 공간 : 커널 위에 있는 모든 코드; app과 종속항목 포함
  - 운영체제 전체를 실행하지 않아 가벼움 ➔ 빠르게 만들고 종료 가능
  - 기본 시스템 위에서 예약/패키징 하므로 효율적   
➔ container를 만든다
- Container (Application code, dependencies) : 단일 app 코드를 생성하는 격리된 사용자 공간



### **Why developers like containers**
- App 중심으로 확장성 높은 고성능 app을 제공, 기본 하드웨어와 소프트웨어를 전제로 작업가능
- App을 쉽게 빌드 가능 ➔ 느슨하게 결합되고 세분화된 구성요소 사용 (모듈식 설계)
- App의 종속 항목을 서로에게서 격리할 방법이 필요, 가상머신에서 app을 패키징하는것은 낭비
- 개발자의 노트북에서는 작동하지만 production에서는 실패하는 app의 문제를 해결하기가 어려움


---------------
## **Container image**
- Image : app과 종속 항목   
➔ Container : 실행중인 image instance    
➔ 소프트웨어를 container image로 빌드하면 개발자는 손쉽게 app을 패키징/제공 가능    
➔ 소프트 웨어가 필요 (Docker)    



### **Containers use a varied set of Linux technologies**
- Processes : Linux process마다 서로 분리된 고유 가상 메모리 주소 공간이 존재, 빠르게 생성/삭제 가능
- Linux namespaces : process ID 번호, directory tree, IP 주소를 제어 (**≠** kubernetes namespaces)
- cgroups : app이 사용가능한 CPU시간, 메모리, I/O 대역폭, 기타 resource의 최대 사용량을 제어
- Union file systems : app과 종속항목을 간결한 최소 레이어 모음으로 캡슐화



### **Containers are structured in layers**
- Container manifest 파일로 image build
- Docker 형식의 container image ➔ Dockerfile (Container image 내부 레이어가 지정됨)
  - `FROM ubuntu:18.04` : FROM 문으로 기본 레이어를 공개 저장소에서 가져와 생성
  - `COPY ./app` : COPY 명령어로 빌드 도구의 현재 디렉토리에서 복사된 파일이 포함된 레이어를 추가
  - `RUN make /app` : RUN 명령어는 make 명령어를 사용하여 app을 빌드하고, 빌드 결과를 세번째 레이어에 배치
  - `CMD python /app/app.py` : 마지막 레이어는 실행 시 container 내에 실행할 명령어를 지정
- 요즘은 배포 및 실행하는 container에 app을 빌드 권장 X ➔ app 패키징에 다단계 빌드 process를 이용
- Image에서 새 container를 만들면, container 런타임에서는 쓰기 가능한 레이어를 기본 레이어 위에 추가 (container layer)



### **Containers promote smaller shared images**
- 여러 container가 동일한 기본 image에 access권한을 공유하면서 자체 데이터 상태를 보유
- Container를 실행하면 container 런타임에서 필요한 레이어를 가져옴 ➔ 업뎃시, 차이 나는 항목만 복사



### **How can you get containers?**
- Download containerized software from a container registry such as `gcr.io.`
  - 공개 오픈소스 image가 다수 포함, google cloud 고객도 이를 사용하여 비공개 image를 cloud IAM과 잘 통합되는 방식으로 저장
- Docker : bulid your own container using the open-source docker command
  - 반드시 신뢰가능한 컴퓨터에 빌드해야함
- Build your own container using Cloud bulid
  - 빌드에 필요한 소스코드를 다양한 스토리지 위치에서 검색가능 (Cloud storage, git repo, cloud source repo)
  - 빌드 단계를 구성하여 종속 항목 가져오기 ➔ 소스코드 컴파일 ➔ 통합 테스트 (Docker container에서 실행)
  - 빌드한 image를 다양한 실행환경에 제공 (GKE, app engine, cloud functions)




----
#### **References**
```
[1] Getting Started with Google Kubernetes Engine, Coursera
```
