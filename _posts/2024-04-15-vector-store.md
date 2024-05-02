---
title: "벡터 스토어와 LLM 활용하기 | Vector Store"
date: 2024-04-15 00:00:00 +/-TTTT
categories: [AI Theory, Natural Language Processing (NLP)]
tags: [vector-store, vector-db, pinecone, weaviate, faiss, qdrant, llm]
math: true
toc: true
author: seoyoung
img_path: /assets/img/for_post/
pin: false
description: 벡터 스토어와 LLM 파이썬 실습 | Vector Store, Vector Database, LLM
---

> 벡터 스토어의 정의와 기능, 파이썬 실습 및 설명을 포함하고 있습니다.
{: .prompt-info }

벡터 데이터베이스는 벡터 임베딩을 색인화하고 저장하여 유사성 검색을 빠르게 수행하는 데이터베이스 유형입니다. 

ChromaDB, Weaviate, Qdrant, Pinecone, Faiss 등 다양한 벡터 데이터베이스가 있습니다.

챗봇 AI에서는 벡터 DB를 통해 외부 지식과 LLM 상태를 제공하며, 유사성 검색 및 RAG를 통한 정확한 결과 생성이 가능합니다.


&nbsp;
&nbsp;
&nbsp;


## **Vector Store, Vector Database란?**

-  빠른 검색 및 유사성 검색을 위해 벡터 임베딩을 색인화하고 저장하는 특정 유형의 데이터베이스
  - **벡터 임베딩**: 다차원 공간에서 객체 또는 데이터를 수학적으로 표현한 것
  - 데이터간의 관계와 유사성이 중요한 작업에 최적화하기 위함
- 각 벡터의 차원 수는 데이터의 복잡도에 따라 다르며, 텍스트, 이미지, 오디오 및 비디오 데이터를 다양한 프로세스를 사용하여 벡터로 변환함
  - 기존 데이터베이스처럼 특정 기준에만 의존하기보다는 의미적 또는 문맥적 관련성에 기초한 검색이 가능함
- 벡터 DB는 외부 메모리 역할을 하여 LLM의 기능을 향상시킬 수 있음

&nbsp;
&nbsp;
&nbsp;

### **주요 기능**

- 벡터 임베딩간의 거리를 기반으로 데이터 객체의 **유사성**을 계산함

  - Vector DB는 쿼리 시 더 빠른 검색을 가능하게 하기 위해 **벡터 인덱싱**을 사용하여 거리를 미리 계산함

  - 해싱 및 그래프 기반 검색과 같은 방법을 포함하는 **ANN <sup>Approximate Nearest Neighbor</sup>** 검색이라는 특수 검색 기술을 사용

#### **양자화 <sup>Quantization</sup>**
- 벡터를 벡터 공간의 유한한 참조점 집합에 매핑하여 벡터 데이터를 효과적으로 압축하는 작업
- 검색을 전체 데이터셋이 아닌 참조 지점의 하위 집합으로 제한하여 스토리지 요구 사항을 줄이고 검색 프로세스 속도를 높임
- 쿼리 속도와 정확성 사이의 균형이 허용되는 환경에서 탁월하므로 어느 정도의 정밀도 손실을 허용할 수 있는 속도에 민감한 애플리케이션에 이상적
  - 데이터 압축과 검색 정밀도 간의 본질적인 상충 관계로 인해 정확성과 최소한의 정보 손실을 요구하는 경우 부적합
    

- **HNSW <sup>Hierarchical Navigable Small World</sup>** 
  - 각 레이어가 데이터셋의 서로 다른 세분성을 나타내는 계층형 그래프를 구성하는 인덱싱 전략
  - 검색은 더 적고 더 먼 지점이 있는 최상위 레이어에서 시작하여 더 자세한 레이어로 이동
  - 데이터셋을 빠르게 탐색할 수 있으며 유사한 벡터의 후보 집합을 신속하게 좁혀 검색 시간을 크게 줄일 수 있음
  - 메모리 소비는 매우 큰 데이터 세트의 경우 제한이 될 수 있으므로 메모리 리소스가 제한되어 있거나 데이터 세트 크기가 실제 메모리 내 용량을 크게 초과하는 경우 부적합
    

- **IVF <sup>Inverted File Index</sup>**
  - k-평균과 같은 알고리즘을 사용하여 벡터 공간을 미리 정의된 수의 클러스터로 나눔
  - 각 벡터는 가장 가까운 클러스터에 할당되며 검색 중에는 가장 관련성이 높은 클러스터의 벡터만 고려
  - 검색 범위가 줄어들어 쿼리 속도가 향상
  - 과도한 분할 가능성으로 인해 저차원 데이터에 부적합
  - 추가 쿼리 시간이 발생할 수 있으므로 가능한 가장 낮은 대기 시간을 요구하는 애플리케이션에 대해서는 부적합

#### **최적화 <sup>Optimization</sup>**
- **차원 감소** : 인덱싱 전략을 적용하기 전에 벡터의 차원을 줄이는 것
- **병렬 처리** : 다중 코어가 있는 CPU 또는 GPU에서 많은 인덱싱 전략을 병렬화
- **동적 인덱싱** : 데이터를 자주 업데이트하는 데이터베이스의 경우 인덱스를 크게 재구성하지 않고도 벡터를 효율적으로 삽입하고 삭제


&nbsp;
&nbsp;
&nbsp;


## **Chatbot with Vector DB**

- **챗봇 AI의 기준 구성 요소**
  - 인간의 언어/추론을 생성해야 함
  - 올바른 대화를 위해서는 앞서 말한 내용을 기억해야 함 ➔ 벡터 DB 필요
  - 일반 지식 이외의 사실 정보를 쿼리할 수 있어야 함 ➔ 벡터 DB 필요
- **벡터 DB의 활용**
  - **LLM 상태 제공**
    - LLM은 상태를 저장하지 않음, 정적임
      - LLM을 Fine-tuning하여 추가 정보를 학습할 수 있지만, Fine-tuning이 완료되면 다시 정지상태가 됨
    - 벡터 DB에서 정보를 쉽게 생성하고 업데이트 할 수 있으므로 벡터 DB는 LLM 상태를 효과적으로 제공함
  - **외부 지식 DB 역할**
    - GPT-3의 **Hallucination** 문제
      - RAG <sup>Retrieval-Augmented Generation</sup>를 통한 정확한 결과 생성
        - 벡터 검색 엔진을 통해 관련 사실 지식을 검색하고 이를 LLM의 Context window에 연결


&nbsp;
&nbsp;
&nbsp;


## **Vector DB Variation**

![fig1](20240415-1.png){: width="700"}
_The landscape of vector databases [^ref1] [^ref2] [^ref3]_



### **ChromaDB**

- 오픈소스 벡터 데이터베이스이며, 쿼리, 필터링, 밀도 추정 등의 다양한 기능이 존재
  - 비용을 지불할 필요가 없기에 호스팅하는 것이 훨씬 저렴하지만, 인프라를 사용하려는 경우 인프라를 직접 관리해야 하는 오버헤드가 발생
- 사용이 단순하며 LLM 프로젝트에 쉽게 통합 가능
- LangChain 및 LlamaIndex 지원

#### 예시 [^ref6]
- 의미 검색 애플리케이션에 최적화를 위한 Sentence Transformer 사용
- 각 명령어 셋과 해당 컨텍스트에 대한 단어 임베딩을 생성하고 이를 벡터 데이터베이스인 ChromaDB에 통합
  - 각 데이터 세트 항목에 대해 LLM 프롬프트에서 검색을 위한 문서 역할을 하는 컨텍스트와 함께 결합된 명령 및 컨텍스트 필드의 임베딩을 생성하고 저장

```python
import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
client = chromadb.Client()
collection = self.client.create_collection(name=collection_name)

for i, item in enumerate(dataset):
    # combined_text = "instruction" + "context"
    embeddings = embedding_model.encode(combined_text).tolist()
    collection.add(embeddings=[embeddings], documents=["context"], ids=["id"])

query_embeddings = embedding_model.encode(query).tolist()
collection.query(query_embeddings=query_embeddings, n_results=n_results)
```

- 토크나이저의 주요 역할은 입력 텍스트를 모델이 이해할 수 있는 형식으로 변환하는 것
  - `Falcon-7B-Instruct` 모델에서 `AutoTokenizer.from_pretrained(model)` 호출은 이 모델과 함께 작동하도록 특별히 설계된 토크나이저를 불러와 모델이 학습된 방식에 맞춰 텍스트가 토큰화되도록 함

```python
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Falcon7BInstructModel:


model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

prompt = question if context is None else f"{context}\n\n{question}"
sequences = pipeline(
            prompt,
            max_length=500,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

generated_text = sequences['generated_text']
```

- 일반적이지 않은 정보로 학습되지 않았기에, 질문에 대해 보다 구체적이고 유용한 답변을 생성하려면 풍부한 컨텍스트가 필요함

```python
falcon_model = Falcon7BInstructModel()
answer = falcon_model.generate_answer(user_question)
```

- Context-aware Answer 생성
  - 임베딩을 생성하고 사용자 질문에서 컨텍스트를 가져오는 데 동일한 VectorStore 클래스를 사용중

```python
context_response = vector_store.search_context(user_question)

# Generate an answer using the Falcon model, incorporating the fetched context
enriched_answer = falcon_model.generate_answer(user_question, context=context)
```


&nbsp;
&nbsp;
&nbsp;



### **Weaviate**

- 오픈소스 벡터 데이터베이스이며, 프로토타입부터 대규모 생산까지 가능
- 빠른 벡터 검색, 추천, 요약, Neural Search 기능 제공
- 자체 호스팅하고 관리할 수 있는 옵션을 제공하는 동시에 Weaviate Cloud에서 서버리스, 자체 클라우드 또는 Azure, AWS 및 GCP 내부에서 사용할 수 있는 클라우드 솔루션을 제공
- OpenAI, Cohere, HuggingFace 제공

#### 예시 [^ref4]
- Weaviate 클라이언트를 벡터 데이터베이스 인스턴스에 연결하는 것으로 구성
  - OpenAI , Cohere 또는 Hugging Face의 임베딩 모델 또는 LLM을 사용하는 경우 이 단계에서 API 키를 제공하여 통합을 활성화함

```python
import weaviate

client = weaviate.Client(
  url = "<https://your-weaviate-endpoint>",
  additional_headers = {
    "X-OpenAI-Api-Key": "YOUR-OPENAI-API-KEY"
  }
)
```

- 자동 벡터화
  - 초기 설정 후 벡터 데이터베이스 내의 데이터 구조를 제공하는 데이터 컬렉션을 정의, 일괄적으로 데이터 개체를 불러옴

```python
class_obj = {
  "class": "MyCollection",
  "vectorizer": "text2vec-openai",
}

client.schema.create_class(class_obj)
client.batch.configure(batch_size=100) *

with client.batch as batch:
    batch.add_data_object(
    class_name="MyCollection",
    data_object={ "some_text_property": "foo", "some_number_property": 1 }
  )
```

- 검색 쿼리와의 유사성을 기반으로 데이터를 검색하도록 설정

```python
response = (
      client.query
      .get("MyCollection", ["some_text"])
      .with_near_text({"concepts": ["My query here"]})
      .do()
)
```

- 기술 스택과 통합
  - 의미 체계 검색 쿼리(`.with_generate()`)를 검색이 강화된 생성 쿼리로 확장
  - `.with_near_text()` 는 먼저 some_text 속성에 대한 관련 컨텍스트를 검색한 다음 `Summarize {some_text} in a tweet` 프롬프트에서 사용

```python
class_obj = {
  "class": "MyCollection",
  "vectorizer": "text2vec-openai",
  "moduleConfig": {
    "text2vec-openai": {},
    "generative-openai": {}
  }
}


response = (
  client.query
  .get("MyCollection", ["some_text"])
  .with_near_text({"concepts": ["My query here"]})
  .with_generate(
    single_prompt="Summarize {some_text} in a tweet."
  )
  .do()
)
```



&nbsp;
&nbsp;
&nbsp;


### **Qdrant**

- 오픈소스이며, 고차원 벡터 검색, 추천 등과 같은 포괄적인 기능 제공
- HNSW 알고리즘에 의한 빠르고 정확한 검색을 제공하며, 벡터 페이로드 기반 결과 필터링을 허용함
- Rust가 내장되어 있어 동적 쿼리 계획을 통해 리소스 사용을 최적화함
- OpenAPI v3와 다양한 언어에 대한 클라이언트 제공


#### 예시 [^ref5]
- Sentence Transformer를 이용한 임베딩 생성

```python
from sentence_transformers import SentenceTransformer

model_name = "neuralmind/bert-base-portuguese-cased"
encoder = SentenceTransformer(model_name_or_path=model_name)

sentence_embedding = encoder.encode(sentence)
```

- 임베딩 저장
  - 벡터 데이터베이스를 가리키는 클라이언트 객체를 만들어야함
  - 벡터 구성 매개변수는 컬렉션을 생성하는 데 사용됨
    - 이러한 매개변수는 벡터를 비교할 때 사용할 크기 및 거리 측정법 과 같은 벡터의 일부 속성을 Qdrant에 알려줌 (내적/유클리드거리)

```python
from qdrant_client import QdrantClient
client = QdrantClient(path="./qdrant_data")

from qdrant_client import models
from qdrant_client.http.models import Distance, VectorParams

client.create_collection(
    collection_name = "news-articles",
    vectors_config = models.VectorParams(
        size = encoder.get_sentence_embedding_dimension(),
        distance = models.Distance.COSINE,
    ),
)

print (client.get_collections())
```

- 데이터베이스를 최종적으로 채우기 전에 업로드할 적절한 객체를 생성
  - 다음 속성을 정의하는 데 사용할 수 있는 PointStruct 클래스를 사용하여 벡터를 저장

```python
from qdrant_client.http.models import PointStruct

points = PointStruct(
        id = "",
        vector = sentence_embedding,
        payload = {}
    )
```

- 모든 항목이 Point 구조로 변환된 후 데이터베이스에 청크로 업로드됨

```python
CHUNK_SIZE = 500
n_chunks = np.ceil(len(points)/CHUNK_SIZE)

for i, points_chunk in enumerate(np.array_split(points, n_chunks)):
    client.upsert(
        collection_name="news-articles",
        wait=True,
        points=points_chunk.tolist()
    )
```

- 컬렉션이 벡터로 채워졌으므로 데이터베이스 쿼리를 시작함
  - 입력 - 입력 텍스트, 입력 벡터 ID

```python
query_text = "Hello World."
query_vector = encoder.encode(query_text).tolist()

from qdrant_client.models import Filter
from qdrant_client.http import models

client.search(
    collection_name="news-articles",
    query_vector=query_vector,
    with_payload=["newsId", "title", "topics"],
    query_filter=None
)
```

- 입력 벡터 ID로 벡터 쿼리
  - 원하는 벡터 ID에 더 가깝지만 원하지 않는 벡터 ID와는 거리가 먼 항목을 추천하도록 벡터 데이터베이스에 요청
  - 원하는 ID와 원하지 않는 ID를 각각 `positive` 와 `negative` 라고 하며 시드로 입력됨

```python
client.recommend(
    collection_name="news-articles",
    positive=[seed_id],
    negative=None,
    with_payload=["newsId", "title", "topics"]
)
```

&nbsp;
&nbsp;
&nbsp;

### **Pinecone**

- 관리형 클라우드 기반 벡터 데이터베이스

- 완전관리형 서비스를 제공하며 확장성이 뛰어남
- 실시간 데이터 수집이 가능하며, 검색에 대한 지연시간이 짧음
- LangChain 지원
- 클라우드에 구애받지 않으므로 Microsoft Azure, AWS 및 Google Cloud와 함께 사용 가능

&nbsp;
&nbsp;
&nbsp;

### **Faiss**

- 오픈소스이며, Python/Numpy 통합을 완벽하게 지원함
- RAM 용량을 초과할 수 있는 벡터까지 다양한 크기에 대한 검색 알고리즘 지원
- 평가 및 매개변수 조정을 위한 보조 코드 제공
- Meta의 Fundamental AI Research에서 개발


&nbsp;
&nbsp;
&nbsp;

### **Azure AI Search**

- Microsoft Azure의 완전 관리형 클라우드 기반 AI 기반 정보 검색 플랫폼
- 확장성이 뛰어나 데이터의 양에 상관없이 쉽게 검색이 가능하며, 다량의 트래픽 로드를 지원
- Azure AI의 타 서비스와 원활하게 통합 가능

&nbsp;
&nbsp;
&nbsp;

#### References

[^ref1]: [Why You Shouldn’t Invest In Vector Databases?](https://blog.det.life/why-you-shouldnt-invest-in-vector-databases-c0cd3f59d23c)
    
[^ref2]: [The Top 5 Vector Databases](https://www.datacamp.com/blog/the-top-5-vector-databases)
    
[^ref3]: [Integrating Vector Databases with LLMs: A Hands-On Guide](https://www.qwak.com/post/utilizing-llms-with-embedding-stores)

[^ref4]: [From prototype to production: Vector databases in generative AI applications](https://stackoverflow.blog/2023/10/09/from-prototype-to-production-vector-databases-in-generative-ai-applications/)

[^ref5]: [Large Language Models and Vector Databases for News Recommendations](https://towardsdatascience.com/large-language-models-and-vector-databases-for-news-recommendations-6f9348fd4030)

[^ref6]: [Integrating Vector Databases with LLMs: A Hands-On Guide](https://www.qwak.com/post/utilizing-llms-with-embedding-stores)

    
