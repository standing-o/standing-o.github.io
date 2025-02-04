---
title: "Chirpy Theme로 Jekyll 기반 깃허브 블로그 시작하기"
date: 2023-11-13 00:00:00 +/-TTTT
categories: [개발, 블로그]
tags: [jekyll, chirpy, ruby, github-blog]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
pin: false
image:
  path: 20231113-t.png
  alt: ""
description: 📚 Jekyll의 Chirpy Theme로 깃허브 블로그를 시작하는 방법을 공유합니다. 
---

--------------------

> **<u>KEYWORDS</u>**         
> 깃허브 블로그, Chirpy 블로그, Jekyll 블로그, Chirpy, Jekyll, Ruby, Github Blog
{: .prompt-info }

--------------------


&nbsp;
&nbsp;
&nbsp;


## **Introduction**
- Jekyll는 Ruby 언어로 개발된 정적 사이트 생성기 입니다. HTML 기반으로 커스텀이 용이하며 다양한 템플릿이 있습니다.

- 그중에서도 [`Chirpy`](https://chirpy.cotes.page/) 템플릿은 기능이 풍부하고 미니멀한 오픈소스 테마입니다. 깔끔하고 사용하기 쉬운 디자인이며 다양한 기능들을 제공하기에 이 템플릿을 선택했습니다.

&nbsp;
&nbsp;
&nbsp;

## **Chirpy 테마로 블로그 시작하기**

공식 페이지에서 소개하기를, [`chirpy-starter`](https://github.com/cotes2020/chirpy-starter)를 사용하는 경우 업그레이드가 용이하며 관련 없는 프로젝트 파일을 격리 가능하다는 장점이 있고, [`jekyll-theme-chirpy`](https://github.com/cotes2020/jekyll-theme-chirpy) 를 사용하면 맞춤형 개발에는 편리하지만 업그레이드가 어렵다는 특징이 있다고 합니다.

저는 jekyll에 익숙하지 않기에 [`chirpy-starter`](https://github.com/cotes2020/chirpy-starter) 를 사용하겠습니다.

&nbsp;
&nbsp;
&nbsp;

### 1. Chirpy Starter 레포지토리를 Clone 합니다.

```shell
git clone https://github.com/cotes2020/chirpy-starter.git
```

&nbsp;
&nbsp;
&nbsp;


### 2. 왼쪽 하단 소셜 아이콘의 설정을 수정합니다.
- `_data/contact.yml`{: .filepath} 파일에서 수정합니다. 파일에 지정된 연락처 옵션을 켜고 끌 수 있습니다.
  - 저는 linkedin 옵션을 추가했습니다.

```yaml
 - type: linkedin
   icon: 'fab fa-linkedin'   # icons powered by <https://fontawesome.com/>
   url:  'https://www.linkedin.com/in/seoyoung-oh-309a24200/'                  # Fill with your Linkedin homepage
```

&nbsp;
&nbsp;
&nbsp;

### 3. 스타일시트 사용자를 정의합니다.

- `jekyll-theme-chirpy`에서 `assets/css/jekyll-theme-chirpy.scss`{: .filepath}, `_sass/main.scss`{: .filepath}, `_sass/addon/variables.scss`{: .filepath}, `_sass/variables-hook.scss`{: .filepath}를 같은 경로로 복사합니다.
- `_sass/addon/variables.scss`{: .filepath}에서 블로그 페이지 구성요소 및 폰트 등을 수정해줍니다.

```scss
/* sidebar */
$sidebar-width: 260px !default; /* the basic width */
$sidebar-width-large: 300px !default; /* screen width: >= 1650px */
$sb-btn-gap: 0.8rem !default;
$sb-btn-gap-lg: 1rem !default;
...

/* fonts */
$font-family-base: 'Source Sans Pro', 'Microsoft Yahei', sans-serif !default;
$font-family-heading: Lato, 'Microsoft Yahei', sans-serif !default;
```

&nbsp;
&nbsp;
&nbsp;

### 4. Site Configuration을 설정합니다.
- `_config.yml`{: .filepath} 파일을 수정합니다.
  - `timezone` ㅣ Asia/Seoul
  - `title` ㅣ 블로그 타이틀
  - `tagline` ㅣ 왼쪽 타이틀 아래 부타이틀
  - `description` ㅣ SEO 키워드
  - `url` ㅣ 블로그 URL 주소
  - `social` ㅣ 왼쪽 하단 아이콘 링크 및 이메일 등을 입력
  - `avatar` ㅣ 왼쪽 상단의 아바타 이미지 설정

&nbsp;
&nbsp;
&nbsp;

### 5. 로컬에서 블로그를 빌드합니다.

- [Ruby](https://www.ruby-lang.org/ko/documentation/installation/#rubyinstaller)를 설치하여 활용하면 사이트를 배포하기 전 로컬에서 테스트 해볼 수 있습니다.

- **의존성 모듈 설치**

  ```shell
  bundle
  ```
- **블로그 빌드**
  - Localhost 4000 포트에서 실행됩니다.
  - `_config.yml`{: .filepath} 파일을 수정했을 경우 다시 실행해야 합니다.

  ```shell
  bundle exec jekyll serve
  ```

&nbsp;
&nbsp;
&nbsp;

### 6. 게시글을 작성합니다.
#### 머리말

- 게시글 파일은 `_posts`{: .filepath} 폴더 내에 위치합니다.
- 게시글은 **markdown** 형식을 기본으로 지원합니다.
  - 게시글의 파일 이름은 `YYYY-MM-DD-TITLE.md` 형태를 지켜야 합니다.

- 게시글의 상단에는 아래와 같은 **머리말**을 작성해야 합니다.
  - `categories`에는 최대 2개의 항목이 포함됩니다.
  - `toc` 는 게시글의 오른쪽에 표시되는 Table of Content 입니다. 
  - `math`는 수학 기능 활성화를 위한 옵션입니다.
  - `pin` 옵션을 통해 한개 또는 여러개의 게시글을 블로그 상단에 고정할 수 있습니다.
  - `image: path:` 에 썸네일 이미지의 경로를 넣어 썸네일을 추가할 수 있습니다. `1200 x 630` 해상도와 `1.91 : 1` 비율을 권장합니다.

```markdown
---
title: "Chirpy Theme로 Jekyll 블로그 시작하기"
date: 2023-11-12 17:00:00 +/-TTTT
categories: [Programming, Blogging]
tags: [jekyll, chirpy, blog]
math: true
toc: true
pin: true
image:
  path: thumbnail.png
  alt: image alternative text
---
```

&nbsp;
&nbsp;
&nbsp;

#### 수학 기능
- 수학 기능을 활성화 한 후 아래와 같이 **수학식**을 추가할 수 있습니다.
  - `$$ math $$` 와 같은 형식으로 활용합니다.
  - `$$ y = \frac{1}{x} $$`

$$ y = \frac{1}{x} $$

&nbsp;
&nbsp;
&nbsp;

#### 이미지 삽입
- 이미지를 추가하고 **너비와 높이**를 지정할 수 있으며, 이미지 아래 **캡션**도 아래와 같이 추가할 수 있습니다. 

- 이미지의 **위치**를 지정할 수 있으며, **다크/라이트 모드**와 **그림자 효과** 설정이 가능합니다.

- **크기 및 캡션 설정**

```markdown
![img-description](/path/to/image){: width="700" height="400" }
_Image Caption_
```

- **위치 설정**

```markdown
![Desktop View](/assets/img/sample/mockup.png){: .normal }
![Desktop View](/assets/img/sample/mockup.png){: .left }
![Desktop View](/assets/img/sample/mockup.png){: .right }
```

- **다크/라이트 모드**

```markdown
![Light mode only](/path/to/light-mode.png){: .light }
![Dark mode only](/path/to/dark-mode.png){: .dark }
```

- **그림자 효과**

```markdown
![Desktop View](/assets/img/sample/mockup.png){: .shadow }
```

&nbsp;
&nbsp;
&nbsp;

#### 구문 강조

- 아래의 문법으로 **파일 경로**를 강조할 수 있습니다.
```markdown
`/path/to/a/file.extend`{: .filepath}
```

- **프롬프트**를 추가하여 문구를 강조할 수 있습니다.
  - `tip` 대신 `info`, `warning`, `danger` 옵션도 사용 가능합니다.

> An example showing the `tip` type prompt.
{: .prompt-tip }

```markdown
> An example showing the `tip` type prompt.
{: .prompt-tip }
```

&nbsp;
&nbsp;
&nbsp;

#### 소스 코드
- 소스 코드는 ```로 감싸는 방식으로 작성하며 인라인 코드는 ``로 코드를 감싸서 작성합니다.
  - `Inline Code`

&nbsp;
&nbsp;
&nbsp;

### 7. 깃허브에 블로그를 배포합니다.

- `{github-username}.github.io`{: .filepath} 라는 이름의 레포지토리를 생성하고 위의 파일들을 모두 push 합니다.

  - github는 자동으로 해당 소스코드들을 build/deploy 합니다.

- `http://{github-username}.github.io`{: .filepath} 에 접속해서 블로그를 확인할 수 있습니다.

#### 사이트맵 플러그인으로 사이트맵 생성
- Gemfile에 `jekyll-sitemap` 플러그인을 추가합니다.

```text
gem "jekyll-sitemap"
```

- 아래 명령어로 플러그인을 설치합니다.

```terminal
bundle install
```

- `_config.yml` 파일에 해당 플러그인을 추가합니다.

```yaml
plugins:
  - jekyll-sitemap
```

- `블로그 주소/sitemap.xml` 로 접속하여 사이트맵을 확인할 수 있습니다.

&nbsp;
&nbsp;
&nbsp;


-------------------
## References
1. [Github:chirpy-starter](https://github.com/cotes2020/chirpy-starter)
2. [Github:jekyll-theme-chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)


