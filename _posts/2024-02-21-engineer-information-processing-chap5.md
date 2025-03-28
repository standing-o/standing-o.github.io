---
title: "정보처리기사 필기 | 5. 정보시스템 구축 관리"
date: 2024-02-21 00:00:00 +/-TTTT
categories: [리뷰, 자격증]
tags: [정보처리기사]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
pin: false
description: 💯 자격증 정보처리기사 필기의 "Chapter 5. 정보시스템 구축 관리" 내용을 요약합니다.
---

--------------------

> **<u>KEYWORDS</u>**         
> 정보처리기사, 정보시스템 구축 관리, 소프트웨어 개발 방법론, DDoS
{: .prompt-info }

---------------------

&nbsp;
&nbsp;
&nbsp;

## **Introduction**
- `1) 소프트웨어 개발 방법론 활용`에서는 소프트웨어를 만드는 방법을 정의하고, 비용 산정 기법 및 소프트웨어 프레임워크를 소개합니다.
- `2) IT 프로젝트 정보 시스템 구축 관리`는 IT 프로젝트 정보 시스템 구축 및 네트워크 관련 신기술에, 클라우드 컴퓨팅, 소프트웨어 정의 네트워크 등에 대한 기본 개념과 특징을 다루고 있습니다.
- `3) 소프트웨어 개발 보안 구축`에서는 시스템 개발 수명 주기의 구성 요소와 주요 모형을 요약하며, 취약점 대비 방법 및 보안에 대한 관리 방식들을 다룹니다.
- `4) 시스템 보안 구축`는 정보 시스템 보안의 필요성, 방화벽의 종류와 기능, DoS 및 DDoS 공격, 보안 아키텍처, 보안 프레임워크 등 다양한 보안 주제를 포함하고 있습니다.

&nbsp;
&nbsp;
&nbsp;



## **1) 소프트웨어 개발 방법론 활용**

### **<u>소프트웨어 개발 방법론</u>**

#### 소프트웨어 개발 방법론의 개요

- 소프트웨어 생명주기 관리 모델에서 프로젝트가 어떠한 방식으로 진행될지, 어떤 산출물을 점검할지에 대해 주로 관심을 가졌다면, 소프트웨어 개발 방법론에서는 소프트웨어를 만드는 방법에 관심을 지님
- 단계별 산출물 뿐만 아니라 산출물 제작 순서와 만드는 방식, 활용 도구 등을 구체적으로 정의
- 개발 과정들을 정리하고 표준화하여 일관성을 유지하고 개발자들 간의 효과적인 협업이 이루어지도록 도움


&nbsp;
&nbsp;
&nbsp;



#### 소프트웨어 개발 방법론의 종류

- **<u>구조적 방법론</u>**
  - **흐름**
    - 요구사항 분석 🠊 구조적 분석 🠊 구조적 설계 🠊 구조적 프로그래밍
  - **특징**
    - 구조와 흐름 등이 간결, 고객이 원하는 요구사항을 이끌어내며 이를 명세화
    - 구조적 분석은 고객들이 원하는 기능 및 시스템 환경, 데이터 등을 종합하여 하나의 데이터 흐름도를 작성
    - 구조적 설계는 모듈 중심으로 설계를 하는 단계를 의미하는 것, 재활용/결합도 등을 감소시켜 독립성을 높임
    - 구조적 프로그래밍은 프로그램의 복잡성을 최소화하기 위해 순차, 조건, 반복이라는 3개의 논리 구조로 구성
- **<u>정보 공학적 방법론</u>**
  - SD (데이터 시스템 구현, 코드 생성) 🠊 BSD (시스템 관점의 데이터 설계, 프로시저 설계) 🠊 BAA (업무영역별 데이터 모델링, 데이터 연관 분석) 🠊 ISP (정보 전략 수집, 전략적 비전, 업무 영역 설정)
  - **특징**
    - 기업 조직의 정보 시스템을 구축하기 위해 계획, 분석, 설계 등 전 과정을 정형화시킨 절차 및 방법론
    - 단순 소프트웨어 개발이 아닌 기업 조직의 경영 전략에 초점을 맞춤
    - 일관성 있고 통일된 정보시스템 구축이 가능, 정보 공학의 효과를 위해 장기간 필요하며 특정한 사업 영역으로부터 독립된 시스템 개발에는 부적합
- **<u>객체 지향 방법론</u>**
  - **흐름**
    - 요건 정의 🠊 객체 지향 분석 (객체 모델링, 동적 모델링, 기능 모델링) 🠊 객체지향 설계/구현 (구현, 객체 설계, 시스템 설계) 🠊 테스트/배포 (테스트, 패키지, 프로젝트 평가)
  - **특징**
    - 프로그램을 객체와 객체 간 인터페이스 형태로 구성하기 위해 문제 영역에서 객체, 클래스 간의 관계를 식별하여 설계 모델로 변환
    - 추상화, 캡슐화, 정보 은폐, 상속, 다형성의 특징
    - 상속에 따른 재사용성이 높고 유지보수가 용이
    - 현실 세계의 반영을 통한 분석 설계의 갭을 최소화


&nbsp;
&nbsp;
&nbsp;



- **<u>CBD 분석 방법론</u>**
  - **흐름**
    - 비즈니스 요구사항 🠊 아키텍쳐 이해 🠊 도메인 분석, 설계 🠊 컴포넌트 설계 🠊 컴포넌트 개발 🠊 인증 🠊 Repository
    - Application 요구 사항 🠊 CBD 설계 🠊 컴포넌트 특화 🠊 컴포넌트 조립 🠊 신규 Application
  - **특징**
    - 재사용이 가능한 컴포넌트 개발이나 상용 컴포넌트 등을 조합하여 애플리케이션 개발 생산성 및 품질을 높임
    - 시스템 유지보수 비용 최소화, 생산성/품질/비용/위험개선이 좋음, 인증 환경 미흡
- **<u>애자일 방법론</u>**
  - 애자일은 스프린트 단계로 이루어짐
  - **특징**
    - 작은 요소들을 출시 할 때 빠르게 만들 수 있음, 시작할 때 프로젝트를 정확히 규정하지 않아도 됨
    - 점진적으로 테스트 되므로 초기에 버그를 발견 가능, 진행 중간에 필요 요소를 수정 가능
- **제품 계열 방법론**
  - 공통의 유사한 기능을 지닌 소프트웨어 제품 또는 소프트웨어 시스템의 집합
  - 사전에 구축된 소프트웨어 아키텍처 등의 소프트웨어 핵심 자산을 재사용
  - 비즈니스 도메인의 어플리케이션들의 공통된 특성을 Core 계층으로 분리, 재사용
  - **목적** ㅣ 다품종화, 생산성, 재사용성
  - **구성** ㅣ Core Asset, Management, Product Develop



&nbsp;
&nbsp;
&nbsp;




### **<u>비용 산정 비법</u>**

#### 하향식 비용 산정 기법

- 과거 유사한 경험을 기반으로 회의를 통해 산정하는 비과학적인 기법
- **분류**
  - **전문가 감정 기법** ㅣ 조직 내 경험이 있는 2명 이상의 전문가에게 비용 산정을 의뢰, 신속하다는 이점이 있으나 편견이 존재
  - **델파이 기법** ㅣ 한 명의 조정자 (중재자)와 여러 전문가의 의견을 종합해 비용을 산정, 전문가 감정 기법의 단점을 보완



&nbsp;
&nbsp;
&nbsp;




#### 상향식 비용 산정 기법

- 프로젝트의 세부적인 작업 단위 별로 비용을 산정 후 전체 비용을 산정
- **<u>분류</u>**
  - **LOC (원시 코드 라인 수) 기법**
    - 각 기능의 원시 코드의 라인 수의 비관치 (가장 많은 라인 수), 낙관치 (가장 적은 라인 수), 기대치 (평균 라인 수)를 측정하여 예측치를 구해 비용을 산정
    - **예측치** = (낙관치 + 4*기대치 + 비관치)/6
    - **노력** = 개발기간 * 투입인원 = LOC/1인 당 월 평균 생성 코드 라인 수
    - **개발 비용** = 노력 * 단위 비용 (1인 당 월 평균 인건비)
    - **생산성** = LOC/노력
  - **개발 단계별 인원 수 (Effort per Task) 기법**
    - 생명 주기의 각 단계별로 노력을 산정, LOC 보다는 더 정확한 기법
    - LOC는 라인 수만 있으며 EPT는 가중치 (일의 어려움) 까지 측정




&nbsp;
&nbsp;
&nbsp;



#### 수학적 산정 기법

- **COCOMO(Constructive Cost Model)**
  - 개발할 소프트웨어의 규모 (LOC)를 예측한 후 소프트웨어의 종류에 따라 각 비용 산정 공식에 대입하여 비용을 선정하는 방식
  - **<u>소프트웨어 개발 분류</u>**
    - **조직형(Organic Mode)** ㅣ 중/소규모의 5만 라인 이하의 소프트웨어
      - 사무처리용, 업무용, 과학용 응용 소프트웨어 개발에 적합
    - **반 분리형(Semi-Detached Mode)** ㅣ 조직형과 ㅐ장형의 중간형, 30만 라인 이하의 소프트웨어
      - 컴파일러, 인터프리터 등의 유틸리티 개발에 적합
    - **내장형(Embedded Mode)** ㅣ 최대 규모의 30만 라인 이상의 소프트웨어
      - 신호기제어 시스템, 미사일 유도 시스템, 실시간 처리 시스템 등의 시스템 프로그램 개발에 적합
- **Putman 모형**
  - 대형 프로젝트의 노력 분포 산정에 이용
  - 시간에 따른 함수로 표현되는 Rayleigh-Norden 곡선의 노력 분포도를 기초로 함
  - 소프트웨어 생명 주기의 전 과정 동안에 사용될 노력의 분포를 가정 해주는 모형
- **기능 점수(FP, Function Point) 모형**
  - 기능 점수를 구한 후 이를 이용해서 비용을 산정하는 기법
  - 소프트웨어 개발에 사용되는 언어와 무관, 소프트웨어 개발 주기 전체 단계에서 사용 가능
  - 라인 수와 무관하게 기능이 많으면 규모도 크고 복잡도도 높다고 판단
  - **기능 점수** = 총 기능 점수 * (0.65 + (0.1 * 총 영향도))
  - 입출력, DB 테이블, 인터페이스, 조회 등의 수를 판단의 근거로 삼음
  - 경험을 바탕으로 단순, 보통, 복잡 정도에 따라 가중치를 부여
  - 측정의 일관성을 유지하기 위해 개발 기술, 개발 방법, 품질 수준 등은 고려하지 않음
  - 구현 관점 (물리적 파일, 화면, 프로그램 수) 가 아닌 사용자 관점의 요구 기능을 정량적으로 산정



&nbsp;
&nbsp;
&nbsp;




### **<u>소프트웨어 개발 방법론 결정</u>**

#### 소프트웨어 개발 방법론과 프로젝트 관리의 비교

- **소프트웨어 개발 방법론**
  - 비즈니스 이슈를 해결하기 위한 근간을 정의, 요구되는 최종 결과물 결과 속성을 규정, 주요 결과물의 서술 및 구성
  - 기술적 역할 및 책임 등을 규정, 비즈니스 조건에 관한 진도를 파악
- **프로젝트 관리**
  - 일의 계획 및 관리 등을 위한 근간을 정의, 최종 결과물과 결과를 일정/예산 목표 등에 맞추기 위한 방법을 규정
  - 결과물을 만들어내기 위한 일을 서술 및 구성, 관리적 역할 및 책임을 규정, 프로젝트 계획에 대한 진도 파악



&nbsp;
&nbsp;
&nbsp;




### **<u>소프트웨어 개발 표준</u>**

#### CMMI <sup>Capability Maturity Model Integration</sup>

- 소프트웨어 개발 및 전산장비 운영 업체들의 업무 능력 및 조직의 성숙도를 평가 하기 위한 모델
- 기존 능력 성숙도 모델 (CMM) 을 발전시킨 것으로서, 기존 소프트웨어 품질 보증 기준으로 사용되며 SW-CMM과 시스템 엔지니어링 분야의 품질 보증 기준으로 사용되던 SE-CMM을 통합하여 개발
- CMMI는 1~5 단계까지 있으며, 5 단계가 가장 높은 수준
- **<u>표현 방법</u>**
  - **단계별 표현 방법** ㅣ 이전에 정의 된 프로세스 집합을 평가하여 이를 통해 조직의 전체적인 프로세스 능력을 알아보는 것
  - **연속적 표현 방법** ㅣ 개별적 프로세스의 영역별 평가를 통해 능력을 알아보는 것




&nbsp;
&nbsp;
&nbsp;



#### ISO/IEC 12207

- **기본 생명주기 프로세스** ㅣ 공급, 획득, 운영, 개발, 유지보수 프로세스
- **지원 생명주기 프로세스** ㅣ 검증, 품질 보증, 확인, 감사, 활동 검토, 형상관리, 문서화, 문제 해결 프로세스
- **조직 생명주기 프로세스** ㅣ 기반 구조, 관리, 훈련, 개선 프로세스


&nbsp;
&nbsp;
&nbsp;





#### SPICE

- 소프트웨어 품질 및 생산성 향상을 위해 소프트웨어 프로세스를 평가 및 개선하는 국제 표준
- **목적**
  - 계약 체결을 위한 수탁 기관의 프로세스를 평가
  - 프로세스 개선을 위한 개발 기관이 스스로 평가
  - 기관에서 지정 및 요구한 조건 등의 만족 여부를 개발 조직이 스스로 평가
- **SPICE의 5개 범주**
  - 고객-공급자, 공학, 지원, 관리, 조직 프로세스
- **SPICE의 6단계**
  - 불완전 🠊 수행 🠊 관리 🠊 확립 🠊 예측 🠊 최적화




&nbsp;
&nbsp;
&nbsp;





### **<u>소프트웨어 개발 방법론 테일러링</u>**

#### 소프트웨어 개발 방법론 테일러링

- 프로젝트의 상황 특성 및 상황 등에 적용하기 위해 기존 개발 방법론의 절차, 기법, 결과물 등을 수정하여 적용
- **<u>필요성</u>**
  - **내부적 조건**
    - **목표 환경** ㅣ 시스템 개발 유형 및 환경 등이 상이
    - **요구 사항** ㅣ 프로젝트 생명 주기 활동 측면에서 개발, 운영, 유지 보수 등 우선 사항이 서로 상이 하기 때문
    - **프로젝트 규모** ㅣ 사업비, 참여 인력, 개발 기간 등 각 규모별 적용되어질 프로젝트 규모가 서로 상이
    - **보유 기술** ㅣ 프로세스, 방법론, 결과물, 인력의 숙련도 등 상이
  - **외부적 조건**
    - **법적 제약 사항** ㅣ 프로젝트 별로 적용되어질 IT 컴플라이언스가 서로 상이
    - **표준 품질 기준** ㅣ 금융, 제조, 의료 업종 별 표준 품질 기준이 서로 상이



&nbsp;
&nbsp;
&nbsp;






### **<u>소프트웨어 개발 프레임 워크</u>**

#### 소프트웨어 개발 프레임워크

- **스프링 프레임 워크**
  - 자바 플랫폼을 위한 오픈소스 애플리케이션 프레임웤, 웹 사이트 개발을 위함
  - **<u>특징</u>**
    - 경량 컨테이너로서 자바 객체를 직접 관리, 스프링은 Plan Old Java Object 방식의 프레임 워크
    - 제어 반전(IoC, Inversion of Control) 지원, 의존성 주입(DI, Dependency Injection) 지원
    - 관점 지향 프로그래밍(AOP, Aspect-Oriented Programming) 지원
    - 영속성과 관련된 다양한 서비스 지원, 확장성이 높음
- **전자 정부 프레임 워크**
  - 대한민국의 공공부문 정보화 사업 시 플랫폼 별 표준화된 개발 프레임워크
  - **<u>특징</u>**
    - **개방형 표준 준수** ㅣ 오픈소스 기반의 범용화되고 공개된 기술의 활용, 특정 사업자에 대한 종속성 배제
    - **상용 솔루션 연계** ㅣ 상용 솔루션과 연계가 가능한 표준을 제시하여 상호 운용성 보장
    - **표준화 지향** ㅣ 민/관/학계로 구성된 자문협의회를 통해 표준화 수행
    - **변화 유연성** ㅣ 각 서비스를 모듈화로 교체가 용이하며 인터페이스 기반 연동으로 모듈 간 변경 영향 최소화
    - **편리하고 다양한 환경 제공** ㅣ 이클립스 기반의 모델링 (UML, ERD). 에디팅, 컴파일링, 디버깅 환경 제공
- **닷넷 프레임 워크**
  - 마이크로 소프트사에서 개발한 윈도우 프로그램 개발 및 실행 환경을 말하는 것
  - 네트워크 작업, 인터페이스 등의 많은 작업을 캡슐화하였으며, 공통 언어 런타임(CLR, Common Language, Runtime) 이라는 이름의 가상 머신 위에서 작동
  - **제공 서비스**
    - 네트워크 작업, 메모리 관리, 유형 및 메모리의 안전성, 보안



&nbsp;
&nbsp;
&nbsp;






## **2) IT 프로젝트 정보 시스템 구축 관리**

### **<u>네트워크 관련 신기술 및 구축 관리</u>**

#### 네트워크 관련 신기술

- **모바일 컴퓨팅(Mobile Computing)**
  - 통상적으로 모바일 컴퓨팅을 이동 환경에서 컴퓨터 또는 컴퓨터와 유사한 기기를 사용하는 것
  - 모바일 컴퓨팅의 이동이라는 의미는 단순히 움직이면서 작업할 수 있다는 포터블의 의미와, 어디서나 자신이 원하는 정보와 연결될 수 있다는 이동형 네트워크의 의미
  - 모바일 컴퓨팅 단말은 인터넷과 무선 데이터 통신 기술이 모바일 컴퓨팅과 결합되는 추세에 따라, 이동형 네트워킹 장비에 대한 개인의 관심이 점차 증대
  - 모바일 컴퓨팅 단말은 노트북, PDA, 핸드 헬드 PC, 팜사이즈 PC, 컴패니언 노트북, 펜 컴퓨팅, 일렉트로닉 오거나이저 등을 포함
- **사물 인터넷(IoT)**
  - 정보통신 기술을 기반으로 모든 사물을 연결하여 사람과 사물간의 소통을 가능케하는 지능형 인프라 및 서비스 기술
  - 제품들 서로 간 또는 인터넷과 직접적/간접적으로 통신이 가능한 임베디드 기술을 활용
  - **<u>특징</u>**
    - **이종성** ㅣ 다양한 종류의 디바이스 (센서, 액추에이터) 들이 서로 상이한 플랫폼에서 동작하면서 디바이스 간의 상이한 프로토콜을 이용한 통신이 가능
    - **정보 보안** ㅣ IoT 기술의 발달로 CCTV 영상, 사용자 건강 정보 등 다양한 영역에서 민감 정보가 생성되고 있으며 동시에 그 데이터 양도 증가하는 추세
    - **자원 제약성** ㅣ CPU, 배터리, 메모리 등 자원 제약성을 가진 IoT 디바이스들은 최소 자원으로 필요성을 만족해야함
    - **이동성** ㅣ IoT는 높은 이동성으로 네트워크 토폴리지가 동적인 것이 특징이나, 낮은 성능 및 대역폭으로 연결성은 좋지 않음



&nbsp;
&nbsp;
&nbsp;


- **클라우드 컴퓨팅(Cloud Computing)**
  - 클라우드를 통해 가상화된 컴퓨터의 시스템 리소스를 요구하는 즉시 제공하는 것
  - 정보를 자신의 컴퓨터가 아닌 클라우드에 연결된 다른 컴퓨터로 처리하는 기술
  - 클라우드 컴퓨팅과 스토리지 솔루션들은 사용자와 기업들에게 개인 소유나 타사 데이터 센터의 데이터를 저장 및 가공하는 다양한 기능을 제공
  - 전기 망을 통한 전력망과 비슷한 일관성 및 규모의 경제를 달성하기 위해 자원의 공유에 의존하게 됨
  - **<u>특징</u>**
    - 기술 인프라 스트럭쳐 자원들의 재 보충, 추가, 확장에 대한 사용자의 유연성을 제고함
    - 각 사용자의 컴퓨터에 애플리케이션을 설치할 필요가 없고 다른 위치에서 접근이 가능
    - 여러 개의 과다한 사이트들을 이용할 때 신뢰성이 제고되며, 잘 설계된 클라우드 컴퓨팅을 사업 연속성, 재해 복구에 적합하도록 만들어줌
    - 사용률의 증가가 필요할 때 확장하고, 자원이 사용되지 않을 때 축소되는 기능
- **인터 클라우드 컴퓨팅(Inter-Cloud Computing)**
  - 2가지 이상의 클라우드 서비스 제공자 간의 상호 연동을 가능케 하는 기술
  - **<u>제공 형태</u>**
    - **인터 클라우드 대등 접속(Peering)** ㅣ 두 클라우드 서비스 제공자 간 직접 연계
    - **인터 클라우드 연합(Federation)** ㅣ 클라우드 서비스 제공자 간의 자원 공유를 기본으로 사용자의 클라우드 사용 요구량에 따라 동적 자원 할당 지원, 논리적인 하나의 서비스 제공
    - **인터 클라우드 중계(Intermediary)** ㅣ 복수의 클라우드 서비스 제공자 간의 직간접적 자원 연계 및 단일 서비스 제공자를 통한 중개 서비스 제공

&nbsp;
&nbsp;
&nbsp;


- **소프트웨어 정의 네트워크(SDN)**
  - 네트워크 장비의 제어 기능을 데이터 전달 기능과 분리하여 소프트웨어적으로 네트워크를 제어하고 관리
  - 트래픽 경로를 지정하는 컨트롤 플레인과 트래픽 전송을 수행하는 데이터 플레인이 분리되어 있으므로 네트워크의 세부적인 구성 정보에 얽매이지 않고 요구 사항에 따라 네트워크 관리가 가능
  - **소프트웨어 정의 네트워크를 활용한 조직을 차별화할 수 있는 구조**
    - **네트워크 프로그래밍 기능** ㅣ 물리적 연결을 제공하는 네트워킹 장치 외부의 소프트웨어를 사용해 네트워크 동작을 제어 가능
    - **인텔리전스 및 제어 기능의 논리적 중앙화** ㅣ 논리적으로 중앙화 된 네트워크 토폴로지를 기반으로 네트워크 리소스의 지능형 제어 및 관리를 지원
    - **네트워크 추상화** ㅣ SDN 기술에서 실행되는 서비스 및 애플리케이션은 네트워크 제어에서 물리적인 연결을 제공하는 기본적인 기술
    - **개방성** ㅣ SDN 아키텍쳐는 여러 공급업체 간의 상호 운용성을 지원할 뿐만 아니라 공급 업체의 중립적인 생태계를 조성
- **차세대 통신망(NGN, Next Generation Networking)**
  - 망 구축비용, 운용비용 절감 및 유연한 네트워크 솔루션을 제공
  - 차세대 통신망은 멀티미디어 통신 서비스를 제공하기 위해 멀티미디어 정보 표현 및 전달에 유리한 패킷 망으로 구성, IP망으로 구축
  - 유무선 망이 발전하여 통합된 형태의 IP망으로 수렴한 망이며 타 망과 각종 게이트웨이를 통해 연동
- **NDN(Named Data Networking)**
  - 데이터 중심 네트워킹 또는 정보 중심 네트워킹과 동일한 개념
  - 인터넷 콘텐츠 자체 정보와 라우터의 기능만을 활용해 목적지로 데이터를 전송하는 기술
  - 사용자들의 요청에 의해 빠른 정보의 전달이 가능한 네트워크 또는 콘텐츠 종류 등에 따라 식별자 체계를 계층적으로 만들어 정보를 식별

&nbsp;
&nbsp;
&nbsp;


- **초광대역(UWB, Ultra-wideband)**
  - 이전의 스펙트럼에 비해 상당히 넓은 대역에 걸쳐 저전력으로 대용량의 정보를 전송하는 무선통신 기술
  - 초광대역은 PC의 대용량 데이터를 프린터에 고속 전송/인쇄, HDTV 동영상을 PC에 전송/저장, 10km ~ 1km 전송 거리
  - 짧은 펄스 폭을 이용한 단거리 고속 무선 통신 기술, 작은 회로와 적은 전력 소모
  - **장점**
    - 저전력 및 속도 면에서 구현 유리, 광대역/저전력에 의한 낮은 간섭
    - 다중 경로 페이딩에 대한 강인성, 벽과 같은 장애물에 대한 투과율이 좋음
  - **단점**
    - 전송거리가 제한, 타 기기에 간섭이 유발되어 전력 제한 등의 저 전력화가 필수

&nbsp;
&nbsp;
&nbsp;


- **근거리 무선 통신(NFC, Near Field Communication)**
  - 13.56MHz 대역을 가지며 NFC 기능을 탑재한 전자기기들이 근거리 무선 통신을 할 수 있게 하는 것, 가까운 거리의 무선 통신
  - 압호화 기술이 적용되어 무선 통신 중에도 정보가 외부로 유출되지 않음
  - **<u>활용 분야</u>**
    - **역/공항** ㅣ 게이트 패스, 스마트 포스터, 인포메이션 키오스크, 버스와 택시
    - **차량** ㅣ 좌석 위치 조정, 차량 문 오픈, 주차 비용 지불
    - **사무실** ㅣ 사무실 출입 명함, PC 로그인, 복사기 인쇄
    - **가게** ㅣ 신용카드 결제, 포인트 적립, 쿠폰 적립, 정보 공유 및 고객 쿠폰 전달
    - **극장/경기장** ㅣ 입장, 이벤트 정보 획득
    - **모든 장소** ㅣ 다운로드 및 애플리케이션 개인화, 사용 기록 확인, 티켓 다운로드, 원격 전화 잠금

- **지리 정보 시스템(GIS, Geographic Information System)**
  - 인간 생활에 필요한 지리 정보를 컴퓨터 데이터로 변환하여 효율적으로 활용
  - **주요 기능**
    - 모든 정보를 수치의 형태로 표현, 3차원 이상의 동적인 지리 정보 제공 가능
    - 다량의 자료를 컴퓨터 기반으로 구축하여 빠르게 검색 가능, 도형 자료와 속성 자료를 쉽게 결합/분석


&nbsp;
&nbsp;
&nbsp;



- **유비쿼터스 센서 네트워크(USN, Ubiquitous Sensor Network)**
  - 어느 곳에나 부착된 센서로부터 사물 및 환경정보를 감지, 저장, 가공, 전달
  - 저전력, 저속도, 근거리 통신
  - **<u>특징</u>**
    - 저가이면서 공격 받기 쉬운 많은 센서 노드로 구성, 간접 저항하는 노드를 구성하기 어려움
    - 저전력 소비 특성 ㅣ 노드의 구조/특성, 토폴로지 및 동작 특성은 소비 전력에 크게 의존
    - 자가 설정이 가능한 네트워크 ㅣ 사전 구성이 어려움, 센서 노드의 실패 가능성, Ad-noc 네트워크 특성
- **자동 구성 네트워크(SON, Self-Organizing Network)**
  - 주변의 상황에 자동으로 적응하여 스스로 망을 구성
  - 통신망 커버리지 및 전송 용량 확장의 경제성 문제를 해결하고, 망 운용과 망 관리의 경제적 효율성을 높임
  - 물리적인 설치 이후 전원만 공급하면 자동적으로 주변 환경에 적응하여 핵심망에 접속
  - **<u>주요 특성</u>**
    - **자율 구성(Self-Configuration)** ㅣ 전원 공급망으로 스스로 IP 주소 등 기타 물리적 파라미터들을 설정하여 핵심망과 연결됨
    - **자율 최적화(Self-Optimization)** ㅣ 주변 환경의 변화에 따라 각종 파라미터들을 최적화하여 망의 전송 성능을 높임, 효율적인 관리
    - **자율 치유(Self-Healing)** ㅣ 망 내의 전송 실패 및 전송 지연 등의 장애 요소를 스스로 발견하고 치유
- **Ad-hoc 네트워크**
  - 서로 독립된 단말끼리 외부의 도움 없이 자신들만으로 자율적인 임시 망을 구성
  - **<u>특징</u>**
    - **동적 토폴로지** ㅣ 이동이 자유롭기 때문에 네트워크 토폴로지가 동적으로 변함
    - **유연한 망 구성** ㅣ 임시 망의 구성은 각 이동 단말이 서로 가까이 있을때만 통신망을 구성
    - **중앙 제어 없음** ㅣ 어떠한 중앙 제어나 표준 지원 서비스의 도움 없이 임시로 망을 구성
    - **정보 전달 방식** ㅣ 일대일 다중 홉 라우팅 방식으로 전달
    - **각 노드 역할 다양성** ㅣ Mesh Network에서는 각 노드가 메세지를 보내거나 받을 수 있으며 라우터 역할도 가능
    - **Self-Healing** ㅣ 하나의 연결이 끊어져도 네트워크는 자동적으로 다른 연결로 메세지를 전송 가능
    - **신호 강도** ㅣ 각 노드가 가까워질수록 신호가 매우 좋아짐
    - **노드 추가 및 탈퇴 유연성** ㅣ 간단하게 노드들이 추가/탈퇴 됨



&nbsp;
&nbsp;
&nbsp;




#### 네트워크 구축 관리

- **네트워크 (통신망)**
  - 컴퓨터에 의한 데이터 전송 기술 및 정보처리 기술이 통합된 형태
  - 원격지의 컴퓨터 상호 간 전기 통신 매체를 통해 데이터를 송수신함
  - PC통신 서비스 회사가 통신망을 설치하여 가입한 사람들에게 여러 정보서비스를 제공하는 형태
  - **<u>종류</u>**
    - **버스 형(Bus)**
      - 하나의 통신회선에 여러 단말기를 접속, 각 컴퓨터는 동등하며 단방향 통신이 가능
      - 타 노드에 영향을 주지 않으므로 단말기의 증설 및 삭제가 용이
      - 회선의 끝에는 종단 장치가 필요, 설치 비용 최소
      - 각 노드의 고장이 타 부분에 전혀 영향을 미치지 않으나 기저 대역 전송 방식을 쓰는 경우에는 거리에 민감 (중계기 필요)
    - **트리 형(Tree)**
      - 중앙에 있는 컴퓨터에 여러 대의 단말기 연결되고 각각의 단말기들은 일정 지역에 설치된 단말기와 다시 접속
      - 분산 처리 시스템에 활용, 단말기들을 가까운 지역별로 하나의 통신 회선에 연결하기 위해 단말기 제어기에 연결
      - 중앙의 컴퓨터와 일정 지역의 단말기까지는 하나의 회선으로 연결되어 있고, 이웃하는 단말기는 이 단말기로부터 근처의 다른 단말기로 회선이 연장된 형태
    - **링 형(Ring)**
      - 컴퓨터들이 이웃한 것들끼리만 연결된 형태로 원모양을 형성, LAN에 사용, 양방향 통신 가능
      - 장치가 단순, 분산제어 및 검사/회복 가능, 동일 링에 오류가 생기면 전체 네트워크 통신이 불가능하여 이중화 대책 필요
      - 단말기 추가 시 회선을 절단해야 하며 기밀 보안이 어려움, 전체 통신량이 증가
    - **성 형(Star)**
      - 중앙에 컴퓨터를 위치시키고 그 주위에 단말기들이 분산되어 중앙 컴퓨터와 1:1로 연결된 중앙 집중식
      - 장애 발생 시 장애 발생 지점을 발견하기 쉬워 보수/관리가 용이
      - 하나의 단말기가 고장나더라도 타 단말기에 영향을 주지 않음, 문제 발생 시 컴퓨터 추가/제거가 쉬워야 하는 경우 구성
      - 많은 회선이 필요, 복잡, 중앙 컴퓨터 고장 시 전체에 문제
    - **망 형(Mesh)**
      - 스타 형과 링 형이 결합된 형태, 모든 단말기들이 각각 연결, 그물 구조
      - 신뢰성이 있고 집중/분산 제어 가능
      - 하나의 컴퓨터가 고장 나더라도 타 시스템에 적은 영향
      - 시스템 구축에 많은 비용 발생, 백본망 구성에 활용, WAN, PSTN, PSDN
    - **혼합 형**
      - Star-bus, Star-ring과 같이 두 개 이상의 토폴로지가 혼합됨


&nbsp;
&nbsp;
&nbsp;





### **<u>DB 구축</u>**

#### 빅데이터

- 기존의 데이터 아키텍쳐로 처리하기 어려운 데이터를 지칭, 대용량 데이터, 데이터 유형의 다양성, 생성 시간 가속화, 빠른 변화 특성
- **<u>특징</u>**
  - **정확성(Veracity)** ㅣ 수집 데이터 정확성에 대한 필요성 대두
  - **가변성(Variability)** ㅣ 데이터가 맥락에 따라 의미가 달라짐
  - **시각화(Visualization)**



&nbsp;
&nbsp;
&nbsp;




#### 데이터 베이스

- **<u>구성 요소</u>**
  - **관리자(DBA, Data Base Administrator)** ㅣ DB의 설계 정의, 효과적인 관리 운영, 총괄 관리 및 제어
  - **응용 프로그래머(Application Programmar)** ㅣ DB 관리자가 정리한 자료들을 토대로 최종 사용자들의 요구에 맞는 인터페이스 및 응용프로그램 개발
  - **최종 사용자(End User)** ㅣ 관리자 및 프로그래머가 만들어준 것을 기반으로 사용하는 사람
- **<u>설계</u>**

1. **요구 조건 분석**
   - 사용자가 원하는 DB의 용도를 파악, 요구사항 명세화
2. **개념적 설계**
   - 사용자들의 요구사항을 이해하기 쉬운 형태로 기술, 개체 관계 모델 사용
   - 트랜잭션 모델링 (처리 중심 설계) 및 개념 스키마 모델링 (데이터 중심 설계) 병행
3. **논리적 설계**
   - DB 관리를 위해 선택한 DBMS의 데이터 모델을 사용하여 논리적 스키마로 변환, E-R Diagram을 특정 DBMS 구조로 변환
4. **물리적 설계**
   - 논리적 DB 구조를 내부 저장 장치 및 접근 경로 설계, 튜닝 및 인덱스 구축
   - **성능 기준** ㅣ 트랜잭션 처리율, 응답시간, 전체 DB에 대한 보고서 생성 시간
   - 단계 논리적 설계로 인해 만들어진 논리 DB 구조로부터 물리적 DB 구조를 설계

&nbsp;
&nbsp;
&nbsp;


- **관계형 DB**
  - 정규화를 통한 합리적인 테이블 모델링으로 이상 현상을 제거하고 데이터 중복을 피함, 동시성 관리, 병행 제어, 체계화, 표준화, SQL
  - 논리적으로 연결된 2차원 관계의 분석 형태
  - 각 테이블은 고유 이름을 가짐, 각 행은 일련의 값들 사이의 관계
  - **<u>특징</u>**
    - **SQL** ㅣ 데이터 조작 언어의 발달
    - **정규형** ㅣ 중복 문제에 대한 해결 기법의 제공
    - **개념화 기법의 발달** ㅣ 데이터의 독립성 보장




&nbsp;
&nbsp;
&nbsp;





## **3) 소프트웨어 개발 보안 구축**

### **<u>시스템 개발 수명 주기</u> <sup>SLDC, System Development Life Cycle</sup>**

#### 시스템 개발 수명 주기

- 대부분의 시스템 작업을 위한 기초 및 구조를 형성하며, 시스템 개발 수명 주기의 각 단계에서 한 개 이상의 전달 가능한 요소를 산출
- **<u>특징</u>**
  - 각 단계별로 수행해야 하는 활동들이 존재
  - 각 단계별로 필요로 하는 결과물들이 있음
  - 각 단계별 활동 정의는 모든 조직에서 동일
- **단계**

1. **시스템 조사** ㅣ 실현 가능성 조사
2. **시스템 분석** ㅣ 기능 요구 사항
3. **시스템 설계** ㅣ 시스템 명세서
4. **시스템 구현** ㅣ 작동하는 시스템
5. **시스템 유지 보수** ㅣ 개선된 시스템

&nbsp;
&nbsp;
&nbsp;

- **장점**
  - 소프트웨어 개발 과정에 관한 명확한 통제 가능, 문서화 용이
  - 대규모 개발에 적합
- **단점**
  - SDLC 후반에 사용자의 요구사항을 반영하기 어려움
  - 사용자의 요구사항의 변경에 대해 적절히 대응하기 어려움

- **<u>대표 모형</u>**
  - **폭포수 모형** ㅣ 검토 및 승인 등을 거쳐 순차적, 하향식으로 개발이 진행되는 생명 주기 모델
    - 이해가 용이, 다음 단계 진행 전 결과 검증, 관리 용이, 요구 도출이 어려움, 설계와 코딩/테스트 지연, 문제 발견의 지연
  - **프로토타입 모형** ㅣ 핵심 기능을 우선 만들어 평가한 후 구현하게 되는 점진적 개발법
    - 요구사항 도출이 용이, 시스템 이해 용이, 의사소통의 향상, 폐기되는 프로토타입의 존재
  - **나선형 모형** ㅣ 폭포수 및 프로토타입 모델 장점에 위험 분석을 추가
    - 테스트 용이, 복잡한 관리
  - **반복 점증적 모형** ㅣ 시스템을 여러 번 나누어 릴리즈
    - 위험 조기 발견 및 최소화 전략 구현이 가능, 변경 관리 용이, 관리 어려움, 경험 부족



&nbsp;
&nbsp;
&nbsp;




#### 입력 데이터 검증 및 표현

- **입력 데이터 검증 및 표현**
  - 프로그램 입력 값에 대한 검증 누락 또는 부적절한 검증, 데이터의 잘못 된 형식 지정으로 인해 발생할 수 있는 보안 위협
  - **취약점 명**
    - SQL 삽입, 경로 조작 및 자원 삽입, 크로스사이트 스크립트, 운영체제 명령어 삽입, 위험한 형식 파일 업로드, 신뢰되지 않는 URL 주소로 자동 접속 연결, XQuery 삽입, XPath 삽입, LDAP 삽입, 크로스사이트 요청 위조, HTTP 응답 분할, 정수 오버플로우, 보안 기능 결정에 사용되는 부적절한 입력값, 메모리 버퍼 오버플로우, 포맷 스트링 삽입

- **보안 기능**
  - 보안 기능 (인증, 접근 제어, 기밀성, 암호화, 권한 관리)를 부적절하게 구현 시 발생 가능한 보안 약점으로 적절한 인증 없는 중요 기능 허용, 부적절 인가 포함
  - **취약점 명**
    - 적절한 인증 없는 중요 기능 허용, 부적절한 인가, 중요한 자원에 대한 잘못된 권한 설정, 취약한 암호화 알고리즘 사용, 중요 정보 평무넞장, 중요 정보 평문 전송, 하드코드된 비밀번호, 충분하지 않은 키 길이 사용, 적절하지 않은 난수 값 사용, 하드코드된 암호화 키, 취약한 비밀번호 허용, 사용자 하드디스크 저장되는 쿠키를 통한 정보 노출, 주석문 안에 포함된 시스템 주요 정보, 솔트 없이 일방향 해쉬 함수 사용, 무결성 검사 없는 코드 다운로드, 반복된 인증 시도 제한 기능 부재

&nbsp;
&nbsp;
&nbsp;

- **시간 및 상태**
  - 동시 수행을 지원하는 병렬 시스템이나 하나 이상의 프로세스가 동작되는 환경에서 시간 및 상태를 적절하게 관리하지 못해 발생할 수 있는 보안 약점
  - **취약점 명**
    - 검사 시점과 사용 시점 (TOCTOU), 종료되지 않는 반복문 또는 재귀 함수
- **에러 처리**
  - 에러를 처리하지 않거나 불충분하게 처리하여 시스템의 중요 정보가 노출
  - 취약점 명
    - 오류 메세지를 통한 정보 노출, 오류 상황 대응 부재, 부적절한 예외 처리
- **코드 오류**
  - 타입 변환의 오류, 자원 등의 부적절한 반환 등과 같이 개발자가 범할 수 있는 코딩 오류로 인해 유발되는 보안 약점
  - **취약점 명**
    - NULL 포인터 역 참조, 부적절한 자원 해제, 해제된 자원 사용, 초기화되지 않는 변수 사용
- **캡슐화**
  - 중요 데이터 또는 기능성 등을 불충분하게 캡슐화 하여 정보 노출, 권한 문제 발생
  - **취약점 명**
    - 잘못된 세션에 의한 데이터 정보 노출, 제거되지 않고 남은 디버그 코드, 시스템 데이터 정보 노출, public 메소드로부터 반환된 private 배열, private 배열에 public 데이터 할당


&nbsp;
&nbsp;
&nbsp;





### **<u>소프트웨어 개발 및 보안 구현</u>**

#### 암호 알고리즘 <sup>Cryptographic Algorithm</sup>

- 암호 알고리즘은 협의의 의미로는 평문을 암호문으로 변환하고 암호문을 다시 평문으로 변환 시 사용되는 알고리즘, 광의의 의미에서는 암호 기술에서 사용되는 모든 알고리즘
- 평문을 암호문으로 바꾸는 과정을 암호화라고 하며 반대의 과정을 복호화라고 함
- 현대 암호 알고리즘에서는 암호화와 복호화 하는 과정에 암호 키가 필요하며, 이 키가 없으면 암호문을 다시 평문으로 변환할 수 없다
- **<u>암호 알고리즘 보안 서비스</u>**
  - **비밀성(Confidentiality, Secrecy, Privacy)** ㅣ 인가된 사용자들만 데이터의 내용을 열람 가능
  - **무결성(Integrity)** ㅣ 비인가된 데이터의 변경을 발견할 수 있도록 해주는 서비스
  - **인증(Authentication)** ㅣ 식별과 검증을 합한 말, 주장된 것을 검증하는 것을 의미
  - **부인 방지(Non-Repudiation)** ㅣ 개체가 지난 행위나 약속을 부인하지 못하도록 함



&nbsp;
&nbsp;
&nbsp;




#### 보안 운영체제 <sup>Secure OS</sup>

- 보안 운영체제는 운영체제에 내제된 결함으로 인해 발생 가능한 각종 해킹으로부터 보호하기 위해 보안 기능이 통합된 보안 커널을 추가 이식한 운영체제
- 기본으로 열린 취약 서비스를 모두 차단해 계정 관리 및 서비스 관리에 있어서 좀 더 나은 보안 체계를 가지고 운영
- 시스템에서 일어나는 프로세스의 활동이 보안 정책에 위반되지 않는지 검사, 일정 수준 CPU 점유, 성능이 중요한 경우 운영체제 도입에 신중을 기해야 함
- **<u>목적</u>**
  - **안정성** ㅣ 중단 없는 안정적인 서비스를 지원
  - **신뢰성** ㅣ 중요 정보의 안전한 보호를 통한 신뢰성 확보
  - **보안성** ㅣ 주요 핵심 서버에 대한 침입차단 및 통합 보안 관리, 안전한 운영체제 기반 서버 보안 보호 대책 마련
    - 버퍼오버 플로우, 인터넷 웜 등의 여러 다양해지는 해킹 공격을 효과적으로 방어할 수 있는 서버 운영 환경 구축

&nbsp;
&nbsp;
&nbsp;


- **<u>기능</u>**
  - **식별 및 인증/계정 관리**
    - 고유 사용자 신분에 대한 인증 및 검증, 시스템 사용 인식/인증, Root 기능 제한 및 보안 관리자 권한 분리
    - 계정의 패스워드 관리 및 유효기간 관리, 사용자별 SU 권한 제어
    - 로그인 시에 사용자 권한 부여 및 해킹에 의한 권한 변경 금지, PAM/LAM 인증 지원, 서비스 별 보안 인증 제어 기능
  - **강제적 접근 통제**
    - 보안 관리자 또는 운영체제에 의해 정해진 규칙에 따라 자동적/강제적 사용자 접근 통제
    - Role Based Access Control, 주체와 객체에 대한 보안 그룹화 및 Role 명시, 정책 검색 기능
    - 모든 접근 제어 설정에 대해 개별적/전체적으로 사전 탐지 기능
  - **임의적 접근 통제**
    - 사전에 보안 정책이나 보안 관리자가 이해 개별 사용자에게 합법적으로 부여한 한도 내 재량권에 따라 사용자가 해당 재량권을 적용하여 접근 통제
  - **객체 재사용 방지**
    - 메모리에 이전 사용자 정보가 남아있지 않도록 기억 장치 공간 정리
  - **완전한 중재 및 조정**
    - 모든 접근 경로에 대한 완전한 통제
  - **감사 및 감사 기록 축소**
    - 보안 관련 사건 기록의 유지 및 감사 기록 보호, 막대한 양의 감사 기록에 대한 분석 및 축소
  - **안전한 경로**
    - 패스워드 설정 및 접근 허용의 변경 등과 같은 보안 관련 작업
  - **보안 커널 변경 방지**
    - 보안 커널의 관리 기능과 시스템의 분리, 시스템 루트 권한으로 접근하더라도 보안 커널 변경 금지
  - **해킹 방지**
    - 커널의 참조 모니터를 통해 알고리즘에 의한 해킹을 근원적으로 탐지하고 막음
    - BOF, Format String, Race Condition, Process Trace, Root Shell 등의 해킹 기법에 대한 직접적 대응
    - Remote/Local Attack 대응, 알려지지 않은 Worm 대응
    - 해킹의 즉각적 탐지, 실시간 차단, 실시간 정보
  - **통합 관리**
    - 다수의 서버 보안 관리를 하나의 관리자 스테이션에서 종합으로 관리 가능




&nbsp;
&nbsp;
&nbsp;





## **4) 시스템 보안 구축**

### **<u>정보 시스템 보안의 필요성</u>**

#### 정보 시스템 보안의 필요성

- **방화벽(Firewall)**
  - 기업이나 조직의 모든 정보가 컴퓨터에 저장되면서, 컴퓨터의 정보 보안을 위해 외부에서 내부, 내부에서 외부의 정보 통신망에 불법으로 접근하는 것을 차단
  - 외부의 인터넷과 조직 내부의 전용 통신망 경계에 건물의 방화벽과 같은 기능을 가진 시스템, 라우터 및 응용 게이트웨이 등을 설치하여 모든 정보의 흐름이 이들을 통해서만 이루어짐
  - **<u>종류</u>**
    - **서킷 게이트웨이(Circuit Gateway)**
      - OSI 7계층 구조에서 세션 계층에서 어플리케이션 계층 사이 접근 제어를 실시
      - 어플리케이션 게이트웨이와는 달리 각 서비스별로 프락시가 존재하는 것이 아니라 일반적인 대표 프락시를 활용
      - 서킷 게이트웨이 방화벽을 통해 내부 시스템으로 접속하기 위해서는 사용자측 PC에 방화벽에 위치한 대표 프락시와 통신하기 위한 수정된 클라이언트 프로그램 필요
      - 클라이언트 프로그램은 모든 통신에 앞서 방화벽에 있는 프락시와 연결을 맺고 안전한 통신 채널인 서킷을 구성하여 내부 시스템과 통신함
      - **<u>특징</u>**
        - 내부의 IP 주소를 숨길 수 있음
        - 수정된 클라이언트 프로그램이 설치된 사용자에게 별도의 인증 절차 없이 투명한 서비스 제공 가능
        - 방화벽 접속을 위한 서킷 게이트웨이를 인식 가능한 클라이언트를 수정해야하는 번거로움
    - **어플리케이션 게이트웨이(Application Gateway)**
      - OSI 7계층 모델 중 어플리케이션 계층까지 동작하며 지나가는 패킷의 헤더 안의 데이터 영역까지 체크
      - 해당 서비스별로 프락시라는 통신 중계용 데몬이 구동되어 각 서비스 요청에 대해 방화벽이 접근 규칙을 적용하고 연결을 대신하는 역할을 수행
      - 외부 시스템 및 내부 시스템은 방화벽의 프락시를 통해서만 연결이 허용, 직접 연결은 비허용, 외부에 대한 내부망의 완벽한 경계선 방어 및 내부의 IP 주소를 숨길 수 있음
      - **<u>특징</u>**
        - 패킷 필터링 기능의 방화벽보다 보안성이 뛰어남, 강력한 로깅 및 감사 기능
        - 프락시의 특성인 프로토콜 및 데이터 전달 기능을 사용하여 사용자 인증 및 바이러스 검색 지원
        - 트래픽이 OSI 7계층에서 처리되는 관계로 타 방식과 비교하여 성능이 떨어짐, 투명한 서비스 제공이 어려움
        - 방화벽에서 새 서비스 제공을 위해서는 새 프락시 데몬이 필요하므로 유연성이 떨어짐
    - **패킷 필터링(Packet Filtering)**
      - OSI 7계층 구조에서 전송 계층과 네트워크 계층에서 동작하며, 지나가는 패킷의 헤더 안의 IP 주소 및 포트 주소만을 단순 검색하여 통제
      - **<u>특징</u>**
        - 세션 관리 및 어플리케이션의 내용을 참조하지 않는 관계로 부가 기능의 지원 및 보안성 등이 떨어짐
        - 사용자 인터페이스 및 로깅 기능이 취약하여 관리가 불편
        - TCP Header의 데이터 영역을 보지 않기에 바이러스에 감염된 메일 및 파일 등을 전송할 경우 차단 불가능
    - **<u>하이브리드 방식(Hybrid)</u>** ㅣ Packet Filtering + Application Gateway
      - 대부분의 방화벽이 채택하는 방식
      - 편의성/유연성 but 관리 복잡


&nbsp;
&nbsp;
&nbsp;





### **<u>시스템 보안</u>**

#### DoS <sup>Denial of Service</sup>

- 공격 대상이 수용 가능한 능력의 정보나 사용자/네트워크의 용량을 초과시켜 정상 작동을 못하게 하는 공격
- 특정 서버에 대한 수많은 접속 시도로 타 이용자가 정상적으로 서비스 이용을 하지 못하게 하거나 서버의 TCP 연결을 바닥나게함
- 통상적으로 DoS는 유명 사이트, 은행, 신용카드 지불 게이트웨이, 루트 네임 서버를 상대로 이루어짐
- **<u>분류</u>**
  - **파괴 공격** ㅣ 디스크나 데이터, 시스템 파괴
  - **시스템 자원의 고갈** ㅣ CPU, 메모리, 디스크의 사용에 과다 부하 가증
  - **네트워크 자원 고갈** ㅣ 쓰레기 데이터로 네트워크 대역폭을 고갈 시킴
- **기본 공격 유형**
  - 전산 자원 소진, 구성 정보 교란, 상태 정보 교란, 물리적 전산망 요소 교란, 원래 사용자와 희생물 사이의 통신 매체 차단


&nbsp;
&nbsp;
&nbsp;



#### DDOS 공격

- DoS 공격이 짧은 시간에 여러 곳에서 일어나게 하는 공격, 심각한 피해, 대책 없음
- 공격자의 위치와 구체적인 발원지 파악 불가능, 자동화 툴 이용, 공격 범위 방대
- 최종 공격 대상 이외에도 공격 증폭을 위한 중간자가 필요
- **<u>구성 요소</u>**
  - **공격자** ㅣ 공격을 주도하는 해커의 컴퓨터
  - **마스터** ㅣ 공격자에게서 직접 명령을 받는 시스템, 여러 에이전트를 관리
  - **에이전트** ㅣ 공격 대상에 직접적인 공격을 가하는 시스템

- **대응책**
  - 방화벽 설치 및 운영, IDS 설치 및 운영
  - 안정적 네트워크 설계, 시스템 패치, 모니터링, 서비스별 대역폭의 제한


&nbsp;
&nbsp;
&nbsp;





#### 자원 고갈 공격

- TCP를 이용하여 PPS(Packet Per Second)를 증가시킴으로써 네트워크 장비 또는 서버 장비의 CPU 부하를 유발시키는 PPS 소비 유형
- bps는 많지 않으나 pps의 급증가로 장비의 다운을 유발
- SYN Flood는 TCP Connection 연결을 사용하는 모든 TCP 통신은 TCP Three-way Handshake 완료 후 데이터를 전송할 수 있는 원리를 이용
- 서버는 SYN ACK 패킷을 보내고 대기 상태에 놓여있게 됨
- 공격자는 ACK를 발송하지 않고 계속 새로운 연결 요청을 하게 되어 서버는 자원 할당을 해지하지 않고 자원만 소비




&nbsp;
&nbsp;
&nbsp;



#### 애플리케이션 공격

- 과다한 웹 접속을 통해 웹 서비스를 교란
- 최근 많이 이루어지는 Slowloris DDoS 공격은 3가지 유형 가운데 웹서비스 지연에 해당됨, 적은 양의 대역폭과 트래픽을 사용
- **<u>유형</u>**
  - HTTP GET Flooding은 과도한 Get 메세지를 이용하여 웹서버의 과부하 유발
  - HTTP CC Attack은 Cache를 사용하지 않도록 하여 Get 메세지를 요청, 적은 트래픽을 통해 웹서버에 과부하 유발



&nbsp;
&nbsp;
&nbsp;






### **<u>보안 아키텍쳐</u>**

#### 보안 아키텍처

- 특정 시나리오 또는 환경과 관련된 필수적 잠재 위험을 다루는 통합 보안 설계
- 언제 어디에 보안 통제들이 적용되는지 명시, 4가지 요인인 위험관리, 벤치마킹, 경제성, 법/규제 등에 따라 결정
- 내부 IT 아키텍처의 구성요소들과 연관성을 나타냄, 구성요소들 간의 의존성을 보여줌
- 세부 보안 통제 명세서는 일반적으로 별도 문서들로 작성
- **<u>장점</u>**
  - **표준화의 이점**
    - 아키텍처에 명시된 통제들의 재사용으로 얻는 비용 효과
  - **원활한 의사소통의 수단**
    - 전문성이 낮은 관련자들 또는 다국적 조직에 대한 의사소통을 향상 시킴


&nbsp;
&nbsp;
&nbsp;





#### 현 시대 보안문제 및 취약점

- 웹 기반 응용시스템의 구조는 보통 웹 클라이언트, 서버, 데이터베이스에 연결한 기업 정보시스템 등을 포함
- 이들 각각의 구성요소는 보안 문제와 취약점을 가짐
- 홍수, 화재, 전력 고장 및 기타 전기적 문제들 역시 네트워크의 어느 지점에서든지 장애를 일으킴



&nbsp;
&nbsp;
&nbsp;




### **<u>기타 보안</u>**

#### Wi-Fi 보안 문제

- 많은 와이파이 네트워크는 인가 없이 네트워크 자원에 접근하기 위한 스니퍼 프로그램을 사용하여 침입자가 쉽게 침투 가능


&nbsp;
&nbsp;
&nbsp;



#### 멀웨어 <sup>Malware</sup>

- **바이러스** ㅣ 사용자의 인지나 허락 없이 실행되도록 하기 위해 자신을 다른 소프트웨어 프로그램이나 파일에 첨부시키는 악성 소프트웨어
- **웜** ㅣ 네트워크를 통해 한 컴퓨터에서 다른 컴퓨터로 자기 자신을 복사할 수 있는 독립적 컴퓨터 프로그램
- **트로이 목마** ㅣ 처음과 다르게 어느 순간 기대와 다른 것을 수행하는 소프트웨어 프로그램



&nbsp;
&nbsp;
&nbsp;




#### 해커와 컴퓨터 범죄

- **스푸핑(Spoofing)**
  - 가짜 이메일 주소를 사용하거나 다른 누군가로 위장함
  - 원래 사용자가 방문하고자 하는 곳 처럼 위장한 사이트를 이용하여 원래 주소와 다른 주소로 웹 연결을 재설정하는 행위
  - 인터넷이나 로컬에 존재하는 모든 연결에 스푸핑이 가능하며, 정보를 얻어내는 것 포함 시스템 마비도 가능
- **스니핑(Sniffing)**
  - 정보를 데이터 속에서 찾는 것, 대비 및 탐지가 용이하지 않음, 수동적 공격
  - 랜에서의 스니핑은 프러미스큐어스(Promiscuous) 모드에서 작동
  - **<u>공격 도구</u>**
    - **TCP Dump** ㅣ 일반적, 네트워크 관리 툴, Snort라는 IDS 기반 프로그램, 법적 효력 있음
    - **DSniff** ㅣ 스니핑 자동화 도구, 패키지 툴 제공, 암호화 계정 및 패스워드를 읽을 수 있음
- **Sniffer Pro (Window)**
  - 윈도우에서는 프러미스큐어스 모드 비지원, WinPCAP와 같은 라이브러리 이용, 뛰어난 GUI로 네트워크 상태 점검 및 패킷 통계를 내기 위한 목적으로 많이 사용



&nbsp;
&nbsp;
&nbsp;




#### 컴퓨터 범죄

- 범법 행위, 수사 및 기소를 위한 컴퓨터 기술 지식을 포함한 형사법에 대한 모든 위반 행위
- **<u>분류</u>**
  - **범죄의 표적으로서의 컴퓨터** ㅣ 보안 대상인 전산 데이터의 기밀성 침해, 허가 없는 컴퓨터 시스템 접속
  - **범죄 수단으로서의 컴퓨터** ㅣ 기업 비밀의 절도, 위협 또는 희롱을 위한 전자 메일의 사용
  - **신원도용(Identity Theft)** ㅣ 다른 누군가로 위장하기 위하여 개인정보를 훔치는 행위
  - **피싱(Phishing)** ㅣ 사용자에게 개인 기밀 데이터를 요구하는 합법적 기업 사이트로 보이는 웹 사이트를 구축하거나 이메일 메세지를 보냄
  - **이블 트윈스(Evil Twins)** ㅣ 인터넷용 와이파이 접속을 제공하는 척 위장하는 무선 네트워크
  - **파밍(Pharming)** ㅣ 사용자가 자신의 브라우저에 정확한 웹페이지 주소를 입력하더라도 가짜 웹페이지로 접속되게 함
  - **클릭 사기(Click Fraud)** ㅣ 온라인 광고에 대한 사기 클릭
  - **국제적 위협** ㅣ 사이버 테러리즘, 사이버 전쟁



&nbsp;
&nbsp;
&nbsp;




### **<u>보안 프레임 워크</u>**

#### 보안 프레임 워크

- 조직 구성원 모두 전문가가 아니더라도 조직의 보안 수준을 유지/향상하기 위한 체계

- **<u>모델</u>**

  - **ISMS(Information Security Management System)** ㅣ 정보 보호 관리 체계

    - 기업이 민감한 정보를 안전하게 보존하도록 관리할 수 있는 체계적인 경영 시스템

  - **PDCA(Plan, Do, Check, Act)** ㅣ 계획, 수행, 점검, 조치를 반복적으로 순환하여 수행하는 모델

    - PDCA를 통해 ISMS를 발전시킴
    - **<u>계획</u>** ㅣ ISMS 수립 단계로 조직이 가진 위험 관리, 정보 보안 목적 달성을 위해 전반적인 정책을 수립

      1. 프로세스를 위한 입출력 규정
      2. 프로세스 별 범위 정의, 고객 요구사항 규정
      3. 프로세스 책임자 규정
      4. 프로세스 네트워크의 전반적인 흐름과 구성도 전개
      5. 프로세스 간의 상호작용 규정
      6. 의도한 결과나 그렇지 않은 결과의 특성 지정
      7. 기준 측정
      8. 모니터링 분석을 위한 방법 지정
      9. 비용, 시간, 손실 등의 경제적 문제 고려
      10. 자료 수집을 위한 방법 규정

    - **<u>수행</u>** ㅣ ISMS 구현과 운영 단계로 수립된 정책을 현재 업무에 적용

      1. 각 프로세스를 위한 자원 분배
      2. 의사소통 경로 수집
      3. 대내외에 정보 제공
      4. 피드백 수용
      5. 자료 수집
      6. 기록 유지

    - **<u>점검</u>** ㅣ ISMS 모니터링, 검토 단계로 실제 정책이 얼마나 잘 적용 및 운영되는지 확인

      1. 프로세스의 측정과 이행이 정확한지 모니터링
      2. 수집된 정량적/정성적 정보 분석
      3. 분석 결과 평가

    - **<u>조치</u>** ㅣ ISMS 관리와 개선 단계로 제대로 운영되지 않는 경우 원인을 분석하고 개선

      1. 시정 및 예방 조치 실행
      2. 시정 및 예방 조치의 유효성과 이행 여부 검증
