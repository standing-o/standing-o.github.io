---
title: "깃허브 블로그에서 구글 검색 엔진 최적화 하기 | Jekyll Google SEO"
date: 2023-12-13 00:00:00 +/-TTTT
categories: [Programming, Blogging]
tags: [jekyll, google-seo, seo]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
pin: false
description: Jekyll에서 구글 검색 엔진 최적화 하기 | Jekyll Google SEO, Search Engine Optimization, Github Blog
---

> 구글 검색 엔진 최적화 (Google SEO) 의 개념과 활용법을 소개합니다.
{: .prompt-tip }

개인 깃허브 블로그를 운영한다면 내 게시글이 구글 검색에 효율적으로 노출되도록 하여 더 많은 사람들이 블로그에 접속하도록 하는 것이 중요합니다.

검색창에서 특정 키워드를 검색했을 때 내 게시글이 고정적으로 상위에 노출된다면, 안정적인 조회수를 확보할 수 있습니다.

Search Engine Optimization (SEO)는 검색 엔진에서 게시글이 상위에 노출될 수 있도록 사이트를 최적화 하는 방법입니다.

&nbsp;
&nbsp;
&nbsp;

## 구글 검색 엔진 최적화란? <sup>Google Search Engine Optimization</sup>
- **Google 검색**은 **크롤링**, **색인 생성**, **검색결과 게재**의 세 단계로 작동합니다. 
  - 크롤러는 웹을 탐색하여 페이지의 텍스트, 이미지, 동영상을 다운로드하고, Google은 페이지의 정보를 분석하여 대규모 데이터베이스에 저장합니다. 
  - 사이트 소유자가 직접 제출하지 않아도 크롤러가 발견한 페이지들이 자동으로 Google 색인에 추가됩니다.

### 크롤링 <sup>Crawling</sup>
- 구글의 **웹 크롤러**는 새로운 페이지를 찾기 위해 웹을 탐색하는데, 중앙 레지스트리가 없어서 지속적으로 새로운 페이지를 찾아내야 합니다. 
  - 이를 위해 URL을 검색하고, 이미 방문한 페이지는 파악된 목록에 추가되며, 새로운 페이지는 이전 페이지에서의 링크를 따라 찾아갑니다. 
  - 사이트 소유자가 제출한 **사이트맵**을 통해서도 페이지를 찾을 수 있습니다. 
- 구글은 페이지를 렌더링하여 최신 버전의 Chrome을 사용하여 자바스크립트를 실행하며, 크롤링 **Googlebot**의 액세스 여부는 사이트의 설정과 상태에 따라 달라지는데, 이를 위해 Search Console의 설정이나 사이트의 응답을 고려합니다.

### 색인 생성 <sup>Indexing</sup>
- 구글은 페이지를 크롤링한 후에 해당 페이지의 내용을 이해하기 위해 **색인**을 생성합니다. 
  - 이때 페이지의 주요 콘텐츠 요소와 속성들을 처리하고 분석하는데, 예로 `<title>` 요소, Alt 속성, 이미지, 동영상 등의 텍스트 콘텐츠가 있습니다.
- 색인 생성 중에는 페이지가 다른 인터넷 페이지와 중복되는지, 혹은 표준적인 페이지인지를 판단하며 표준 페이지는 검색 결과에 나타날 수 있는 페이지를 의미합니다.
  - 구글은 표준 페이지와 콘텐츠에 관한 정보를 수집하며, 이 정보는 검색 결과에 페이지를 나타내는 데 사용됩니다.
- 이렇게 수집된 정보들은 대규모 데이터베이스인 Google 색인에 저장되지만, 모든 페이지의 색인 생성이 보장되는 것은 아니며 색인 생성은 페이지의 콘텐츠와 메타데이터에 따라 다릅니다.

### 검색 결과 게재
- 구글은 사용자가 검색한 내용과 관련성이 높은 페이지를 색인에서 찾아 반환하는데, 이때 사용자의 위치, 언어, 기기 등 다양한 정보를 고려해 결과를 결정합니다. 
- 페이지가 색인에 있더라도 검색 결과에 표시되지 않을 수 있는데, 이는 여러 가지 원인으로 인해 발생합니다.

&nbsp;
&nbsp;
&nbsp;

## 주요 SEO 방법
### 사이트맵 <sup>Sitemap</sup>
- **Sitemap**은 웹 사이트의 페이지 구조를 검색 엔진에게 알려주는 XML 형식의 파일이나 페이지의 집합이며, 웹 사이트의 모든 중요한 페이지 및 그 구조에 대한 정보들을 포함하고 있습니다.
- 검색 엔진 크롤러에게 어떤 페이지를 색인하고 어떤 페이지를 무시해야 하는지에 대한 지침을 제공하여 SEO에 도움을 줍니다.
- 웹 사이트의 구조를 설명하고 오류를 식별하여 검색 엔진이 웹 사이트를 더 효과적으로 이해하고 사용자에게 더 나은 검색 결과를 제공할 수 있도록 합니다.

![fig1](20231213-1.png){: width="700"}
_https://standing-o.github.io/sitemap.xml_

### RSS <sup>Really Simple Syndication</sup>
- RSS는 웹 사이트의 컨텐츠를 쉽게 공유하고 구독할 수 있도록 하는 표준 웹 피드 형식입니다.
- 콘텐츠 자동 공유로 백링크를 증가시켜 검색 엔진에서 웹 사이트의 신뢰성을 높이는 데 기여하며, 신속한 검색 엔진 색인화를 통해 업데이트된 콘텐츠를 빠르게 검색 결과에 반영할 수 있습니다. 
  - 다른 웹사이트에서 내 웹사이트로 자연스럽게 링크가 걸리게 하는 것을 백링크라고 합니다.
  - 검색 엔진에서는 이를 긍정적으로 평가하여 웹사이트의 검색 노출 순위를 상승시키고 트래픽을 늘릴 수 있습니다.

### robots.txt
- `robots.txt` 파일은 웹 사이트에서 검색 엔진 크롤러에게 허용되는 URL을 지정하여 크롤링 트래픽을 관리하는 역할을 합니다. 
- 이 파일은 주로 사이트 오버로드를 방지하기 위해 사용되며, Google에 표시되지 않도록 하는 메커니즘은 아닙니다. 
  - 크롤링 트래픽을 관리하거나 특정 페이지를 차단할 수 있습니다.


> 위와 같은 요소들을 [구글 서치 콘솔](https://search.google.com/search-console)과 [네이버 서치 어드바이저](https://searchadvisor.naver.com/) 등에 등록하면 해당 검색 플랫폼에 내 사이트가 표시되도록 할 수 있습니다.
{: .prompt-info }

&nbsp;
&nbsp;
&nbsp;

### PageSpeed Insights 활용하기
- 웹사이트의 성능이 향상되면 검색 엔진에서는 해당 사이트를 더 긍정적으로 평가하고 사용자 경험을 향상시키기 위해 노력하는 것으로 간주합니다. 
  - 검색엔진은 웹사이트의 로딩 속도를 고려하여 순위를 결정하기에, 성능이 우수하다면 검색 결과에서 상위권에 노출될 가능성이 높아집니다.
  - 웹사이트가 빠르게 로딩되면 검색 엔진 크롤러가 효율적으로 웹 페이지를 크롤링할 수 있습니다. 
  - 빠른 웹사이트 성능은 사용자들이 사이트에서 머무는 시간을 늘릴 수 있습니다. 
  - 검색 엔진은 모바일 최적화된 웹사이트를 선호합니다.
- [**PageSpeed Insights**](https://pagespeed.web.dev/)를 활용하면 웹사이트 성능을 진단하고 개선 방법을 쉽게 알 수 있습니다.
  - 저는 1. 차세대 형식을 사용해 이미지 제공하기, 2. 이미지 크기 적절하게 설정하기 등의 개선 방법을 제공받았습니다.

![fig1](20231213-2.png){: width="700"}

&nbsp;
&nbsp;
&nbsp;

## Jekyll에서의 SEO
### jekyll-seo-tag 플러그인
- [`jekyll-seo-tag`](https://github.com/jekyll/jekyll-seo-tag) 플러그인은 다음 메타 태그들을 자동으로 생성해줍니다.
  - 페이지 제목/설명, Canonical URL, 오픈 그래프 제목, 설명, Twitter 요약 카드 메타 데이터 등

- `Gemfile`{: .filepath} 에 아래와 같이 추가합니다.

```
gem 'jekyll-seo-tag'
```

- `_config.yml`{: .filepath} 에 아래와 같이 추가합니다.

```yaml
plugins:
  - jekyll-seo-tag
```

### jekyll-sitemap 플러그인
- [`jekyll-sitemap`](https://github.com/jekyll/jekyll-sitemap) 플러그인은 자동으로 생성 날짜를 사용하여 태그를 채워줍니다.

- `Gemfile`{: .filepath} 에 아래와 같이 추가합니다.

```
gem 'jekyll-sitemap'
```

- `_config.yml`{: .filepath} 에 아래와 같이 추가합니다.

```yaml
url: "https://example.com" # the base hostname & protocol for your site
plugins:
  - jekyll-sitemap
```

&nbsp;
&nbsp;
&nbsp;

----------------
## References
1. [A Beginner's Guide to SEO optimization in a Jekyll static website](https://jsinibardy.com/optimize-seo-jekyll)

2. [Get your website on Google](https://developers.google.com/search/docs/fundamentals/get-on-google)
