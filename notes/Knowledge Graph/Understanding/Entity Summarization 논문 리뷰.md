# Entity Summarization

사용자에게 빠르고 효율적인 정보를 제공하기 위한 지식그래프의 엔티티 요약 자동화



## Table of Contents

1. Introduction
   1. Motivation and Application for Entity Summarization
   2. Problem Statement
      1. Semantic Data
      2. Entity Description
      3. Entity Summary
      4. Entity Summarization
   3. Related Problems and Surveys
      1. Document Summarization
      2. Graph Summarization
      3. Ontology Summarization
2. Technical Features for Entity Summarization
   1. Generic Features
      1. Frequency and Centrality
         1. Frequency of Property
         2. Frequency of Property Value
         3. Centrality of Property Value
         4. Centrality of Triple
         5. LDA
         6. FCA
      2. Informativeness
         1. Statistical Informativeness
         2. Ontological Informativeness
      3. Diversity and Coverage
         1. Discrete Similarity
         2. Textual Similarity
         3. Semantic Similarity
   2. Specific Features
      1. Domain Knowledge
      2. Context Awareness
      3. Personalization
   3. Frameworks for Feature Combination
      1. Simple Frameworks
      2. Random Surfer Model
      3. Similarity-based Grouping
      4. MMR-like Re-Ranking
      5. Combinatorial Optimization
      6. Learning to Rank
   4. Evaluation Benchmarks for Entity Summarization
      1. Evaluation Metrics
         1. Quality
         2. F-measure
         3. Ranking-based Metrics
      2. Datasets
      3. Benchmark Results
   5. Conclusion and Furure Directions



## 1. Introduction

지식 그래프를 이루는 triple 구조는 다음과 같다.

- <entity, property, value>

- property는 다음과 같이 3가지로 구분 되어진다

  - type: 엔티티의 타입/클래스를 가지는 프로퍼티
    - ex) <이병헌, type, person>
  - attribute : 원시타입의 데이터를 가지는 프로퍼티
    - ex) <이병헌, 직업, 영화배우> 
  - relation: 다른 엔티티와의 관계를 가지는 프로퍼티
    - ex) <이병헌, 주연, 남산의 부장들>

  

### 1.1 Motivation and Application for Entity Summarization

하나의 엔티티에 연관된 트리플은 적게는 몇십개 많게는 몇백개가 존재한다. knowledge graph 사용하는 사용자들의 유저 인터페이스에 특히나 모바일같은 기기에는 다 담아 낼수 없다. 수많은 정보들 필요하거나 핵심정보를 빠르게 찾아보기에 힘들다. 해당 문제를 해결하기 위한 것은 엔티티의 모든 정보를 보여주기 보다는 요약본을 보여주는게 더 효율적이다. 구글에서 검색하면 오른쪽에 엔티티의 요약본을 볼수 있다. 자동으로 엔티티의 정보를 요약 하는 것을 **entity summarization** 이라고 한다.

entity summarization를 잘 활용 할수 있는 한 예는 브라우저에서 뉴스를 보고 있을때 NER을 통해 해당 엔티티에 마우스 포인트를 올렸을때 해당 엔티티의 요약된 정보를 작은 창으로 보여줄수 있을 것이다.



### 1.2 Problem Statement

#### Semantic Data

$$
\begin{align}
\mathbb{E}: & \space \text{set of all entities} \\
\mathbb{P}: & \space \text{set of all properties} 
\begin{cases}
\text{type} \space & \in \mathbb{P} \\
\text{attribute} \space \mathbb{A} & \in \mathbb{P} \\
\text{relation} \space \mathbb{R}& \in \mathbb{P}
\end{cases} \\
\mathbb{C}: & \space \text{set of all classes} \\
\mathbb{L}: & \space \text{set of all literals} \\
\end{align}
$$



#### Entity Description

$$
\begin{align}
e_t: & \space \text{entity in a triple t}  \\
p_t: & \space \text{property in a triple t} \\
v_t: & \space \text{value in a triple t} \\
\end{align} \\
T_e = \{t\in T: e_t = e \space \text{or} \space v_t = e \}
$$



#### Entity Summary

엔티티 $e$ 의 summary는 선택된 k개의 트리플들의 집합으로 size-contrained subset of triples(e)로 정의 한다. 
$$
S_e \subseteq T_e \space \text{with} \space \vert S_e\vert \leq k
$$

#### Entity Summarization

우리가 풀어야 하는 문제는 다음과 같다
$$
\begin{align}
& \underset{S_e}{\mathrm{argmax}} \space Score(S_e \mid T_e) \\ 
& \text{subject to} \space \vert S_e \vert \leq k
\end{align}
$$
해당 문제를 푸는 알고리즘을 **entity summarizer**라고 한다. entity summarizer 마다 Score 함수를 정의하는 방식이 다르다. 한가지 방식은 poinwise방식으로 개별 트리플을 각각 점수를 매겨 합산하는 방식으로 Score를 정의한다.
$$
Score(S_e\mid T) = \sum_{t\in S_e} score(t\mid T_e)
$$
$score(t\mid T_e)$ 는 트리플 $t \in S_e$ 의 퀄리티 스코어이다. 각 트리플의 점수를 어떻게 정할 수 있는지 알아보자.



### 1.3 Related Problems and Surveys

#### Document Summarization

다른 점은 트리플은 정형 데이터이고 텍스트는 비정형 데이터이다. 문서요약에서 사용되는 기법이 엔티티 요약에서도 응용되어 질 수 있다.

#### Graph Summarization

하나의 노드가 여러 노드의 대표노드가 되어 그래프를 요약 하는 방법이나 자주 발생하는 서브그래프를 대표노드로 만들어서 요약 할 수 있다. star-shaped 형태의 복잡한 관계를 가지는 지식그래프에서 엔티티 요약을 하는데 해당 패턴을 찾는 것은 쉽지 않는 방법이다.

#### Ontology Summarization

엔티티의 스펙인 온톨로지 스키마 자체를 요약하는 방법이다. 해당 요약 방법은 엔티티레벨이 아닌 스키마레벨에서 요약 하는 것이기에 엔티티가 가진 value들의 시맨틱을 담아내기 힘들다.



## 2. Technical Features for Entity Summarization

- Generic Features

- Specific Features

### 2.1 Generic Features

- frequency/centrality feature
  - 각 트리플의 특징을 측정한다.
  - Frequenty of Property
    - 프로퍼티의 빈도로 점수를 줌
  - Frequency of Property Value
    - 밸류의 빈도로 점수를 줌
  - Centrality of Property Value
    - 밸류도 까보면 어떤 엔티티로 존재하고 있는 것일수 있기 때문에 밸류노드의 centrality를 구할 수 있다. in-degree나 페이지랭크등을 사용 할 수 있다. 현재 크루즈 데이터는 sameas로 연결되어 있는 것들의 페이지랭크 점수를 가져와서 해당 프로퍼터의 점수를 매기기에는 연결이 안되어 있는 프로퍼티가 너무 많음
  - Cemtrality of Triple
    - 어떤 엔티티의 트리플들이 서로 다 연결되어 있지만 연결강도는 similarity로 정의함. similarity는 겹치는 단어나 단어 임베딩의 유사도를 사용하나? weighted pagerank를 사용한다. 
  - LDA
    - 통계적 생성모형이다. 모든 도큐먼트는 정해진 주제들의 mixture이고 각 토픽의 단어들을 주어진 확률로 샘플링 된다. 단어의 순서는 무시한다.
    - 엔티티의 정보를 하나의 코퍼스로 볼수 있고 위 LDA를 응용 할 수 있다. 어떻게 점수를 매기지?
  - FCA
    - 프로퍼티와 밸류들을 aggregate해서 계층적으로 만들어 준다.
- informativeness feature
  - 적은 엔티티에서 발생하는 단어들이 더 많은 정보를 제공하고 중요하다고 생각한다.
  - 두가지 measure 방식이 있다.
    - Statistical
      - 퍼로퍼티-배류 쌍이 적은 엔티티에서 발생하면 중요한다고 판단
      - 엔티티 e의 트리플 <e, p, v>은 확률 이벤트라고 가정한다. 엔티티 e에서 <p, v> 트리플을 self-information이라고 할수 있다. 다음과 같이 확률로 정의된다. TODO: 수식 채워넣기
      - self-information은 해당 트리플이 엔티티를 얼마나 잘 설명하는가로 생각 할 수 있다.
      - 엔티티 임베딩에서 트리플 임베딩이 얼마나 유사한지로 풀수도 있다.  
    - Ontological
      - 엔티티의 TYPE에 기반한 점수 measure 방식
      - <이병헌, TYPE, 사람> 보다는 <이병헌, TYPE, 영화배우>에 더 우선순위를 둔다. 영화배우가 TYPE 계층 구조상 사람이랑 같은 맥락 이지만 더 아래에 위치한 구체적이기 때문이다.
- diversity/coverage feature
  - summary가 있고 탑에 위치한 모든 트리플들이 같은 맥락이라면 좋은 summary라고 할 수 없다. 태그 추출도 마찬가지이다. generic summary가 주는 정보에서는 다양성이 있어야 한다. 최대한 비슷한 의미를 가진 트리플들의 중복을 피해야함.
  - Discrete Similarity
    - 같은 프로퍼티가 있으면 가장 점수가 좋은 트리플만 선택. 프로퍼티로 group by해서 max를 선택한다.
  - Textual Similarity
    - 워드 벡터를 통해서 유사도가 비슷하면 제거해주는 방식
  - Semantic Similarity
    - 온톨로지 시맨틱을 사용해서 제거해주는 방식. 그래프의 구조나 계층 구조를 고려해서 제거하는 방식
    - <이병헌, TYPE, 사람> 보다는 <이병헌, TYPE, 영화배우>을 선택



### 2.2 Specific Features

- 외부 지식을 사용해서 피쳐를 만든다
- Domain Knowledge
  - 도메인상 중요한 트리플들을 우선순위 준다.
- Context Awareness
  - 컨텍스트 기반 트리플들에세 우선순위를 준다.
  - 검색 시스템에서는 사용자 쿼리가 컨텍스트가 될수 있다. 아니면 해당 엔티티가 언급된 특정 주제를 가진 문서가 컨텍스트가 될수 있다.
- Personalization
  - 사용자 니즈가 context가 될수 있다. 사용자의 성향이 컨텍스트가 된다.

## 3. Frameworks for Feature Combination

- 피쳐들을 조합 할때 피쳐들의 objectives가 중복된 것일 수 있다. 여러 피쳐들을 어떻게 조합 할수 있는지 알아보자.
- Simple Frameworks
  - 각 피쳐로 트리플들을 랭킹 맥이고 각 피쳐의 탑 트리플들을 유니온한다. 하지만 해당 방법은 다양성와 커버리지를 위해서는 좋지 않다.
- Random Surfer Model
  - 랜덤워크 방식으로 하는 것 같은데 다양성과 커버리지를 좋게 하는 것을 목적으로 하는 방식은 아니다.
- Similarity-based Grouping
  - 유사한 프로퍼티로 group by해서 max하는 방식
- MMR-like Re-Ranking
  - IR에서 사용되는 방법인데 주어진 쿼리에 연관된 문서들을 가져오고 가져온 문서들을 다양성을 향상시키기 위해서 리랭킹 하는 방식이다.
  - TODO: 수식 추가
  - 그리디 알고리즘으로 풀기 때문에 optimal solution을 제공하지는 않는다.
- Combinatorial Optimization
  - MMR의 sub-optimality를 제거 하기 위해 사용 한다.
  - TODO: 수식 추가
  - quadratic knapsack 문제이다.
  - NP-hard 라서 휴리스틱하게 풀거나 MMR이랑 마찬가지로 그리디하게 풀기도한다.
- Learning to Rank
  - 각 프로퍼티에 마다 피쳐 벡터가 주어질 것이고 순위가 label되어서 supervised하게 학습해서 랭킹 모델을 만드는 방식이다. supervised이다 보니 label된 데이터가 있어야 하는 단점이 있다.



## 4. Evaluation Benchmarks for Entity Summarization

평가 방법에는 두가지 방식이 있다.

- Intrinsic method: 기계가 요약한 트리플이랑 사람이 만든 요약을 비교하는 방법
- extrinsic method: entity summarizer가 생성한 트리플 목록이 다운스트림 태스크에서 사용되면서 사용자 interaction을 통해서 평가되는 방법. A/B 테스트 생각 할수 있다.






$$
\begin{align}
\text{sim}(c_a, c_b) 
& = \frac{c_a \cdot c_b}{\Vert c_a \Vert_2 \Vert c_b \Vert_2} \\ \\
& = \frac{\frac{\sum_i^m w_{a, i}}{m} \cdot \frac{\sum_j^n w_{b, j}}{n}}
{\Vert \frac{\sum_i^m w_{a, i}}{m} \Vert_2 \Vert \frac{\sum_j^n w_{b, j}}{n} \Vert_2} \\ \\
& = \frac{\sum_i^mw_{a, i}\cdot \sum_j^n{w_{b, j}}}{\sqrt{\sum_i^mw_{a, i}^2}\sqrt{\sum_j^mw_{b, j}^2}}
\end{align}
$$


















