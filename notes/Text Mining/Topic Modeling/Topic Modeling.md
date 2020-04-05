# Topic Modeling

1. LDA (Latent Dirichelet Allocation)
2. ATM (Author Topic Model)
3. DMR (Dirichlet Multinomial Regression)
4. LDA의 단점
5. DP (Dirichelet Process)
6. Chinese Restaurant Process
7.  HDP (Hierarchical Dirichelet Process)
8. HDP-LDA
9. hLDA (Hierarchical LDA)
10. Nested Chinese Restautant Process
11. sLDA (Supervised LDA)
12. L-LDA (Labeled LDA)
13. JST (Joint Sentiment Topic Model)
14. TOT (Topic over Time)
15. hPAM (Hierarchical Pachinko Allocation)
16. PAM (Pachinko Allocation)
17. Topical N Gram
18. HPYLM (Hierarchical Pitman-You Language Model)
19. RTM (Relational Topic Model)
20. gRTM (Generalized Relational Topic Model)
21. LDA2Vec
22. CTM (Correlated Topic Model)



## LDA (Latent Dirichelet Allocation)

K개의 주제와 W종류의 단어가 들어가 있는 D개의 문헌이 있다고 가정한다. 하나의 문헌이 여러개의 주제들의 조합으로 이루어져 있고, 각각의 주제는 여러개의 단어가 포함된다고 가정한다.

![21436D3E59567B7905](/Users/kakao/Documents/dev/jupyter/notes/Text Mining/Topic Modeling/assets/21436D3E59567B7905.png)



## ATM (Author Topic Model)

LDA 기법에서는 놓치고 있는게 몇 개 있다. 문헌을 작성한 저자들이 여러명일 경우 저자에 따라서 사용하는 주제나 단어가 다를수 있다. 만약 그렇다면 저자에 따라 선호하는 주제의 차이를 발견할 수 있다. LDA는 문헌별 주제분포를 계산하는 반면, ATM은 저자별 주제분포를 계산한다.

![231CEF4059567AEC2E](/Users/kakao/Documents/dev/jupyter/notes/Text Mining/Topic Modeling/assets/231CEF4059567AEC2E.png)



## DMR (Dirichlet Multinomial Regression)

ATM 모형의 경우 단순히 문헌의 주제분포를 저자의 주제분포로 옮긴 것으로 생각 할 수 있다. 문헌의 주제 분포를 관장하는 하이퍼파라미터 $$\alpha$$ 가 문헌의 메타데이터(저자, 연도, 기관, 국가 등등)에 따라서 달라질 수 있다고 가정한 것이다. ATM 모형에서는 저자별로 달라지는 주제분포만을 파악할 수 있었지만, DMR에서는 저자뿐만 아니라 문헌에 할당된 명목변수라면 어떤 것도 사용 할수 있다. 예를 들어 문헌이 작성된 연도를 메타데이터로 사용한다면, 연도에 따라 달라지는 주제 분포 파라미터를 구할 수 있다.

![26139B35595682D33C](/Users/kakao/Documents/dev/jupyter/notes/Text Mining/Topic Modeling/assets/26139B35595682D33C.jpg)

## LDA의 단점

LDA 기법은 주제의 수 $$K$$ 를 설정 해주어야 한다. $$K$$ 값에 따라 LDA 토픽 모델링의 결과가 크게 달라지기 때문에 적절한 K값 선정은 중요하다. 하지만 얼만큼의 주제가 포함되어 있는지는 사전에 알기는 어렵다. 따라서 다양한 $$K$$ 값에 대해서 분석을 돌리고 perplexity 값을 기준으로 (또는 해석이 합리적인 정도를 기준으로) 적절한 $$K$$ 값을 선정한다. 

이것이 LDA의 주요한 약점중 하나이다. 하지만 우린 데이터에 따라 적절한 주제 개수를 찾아주도록 LDA 기법을 개선 할 수 있다. 디리클레 프로세스 (Dirichelet Process)와 이를 응용한 계층적 디리클레 프로세스 (Hierarchical Dirichelet Process)를 토픽 모델링에 적용하여 이 문제를 해결 한다.



## DP (Dirichelet Process)





## Chinese Restaurant Process



##HDP (Hierarchical Dirichelet Process)



##HDP-LDA



## hLDA (Hierarchical LDA)



## Nested Chinese Restautant Process



## sLDA (Supervised LDA)



## L-LDA (Labeled LDA)



##JST (Joint Sentiment Topic Model)



## TOT (Topic over Time)



## MG-LDA (Multi Grain LDA)



## hPAM (Hierarchical Pachinko Allocation)

계층적인 주제를 모델링하기 위한 모형으로 크게 파칭코 할당 모형과 계층적 LDA 모형, 그리고 이 둘을 합친 계층적 파칭코 할당 모형등이 있다.



## PAM (Pachinko Allocation)



## Topical N Gram

단어의 순서를 고려한 모형



## HPYLM (Hierarchical Pitman-You Language Model)



## RTM (Relational Topic Model)



## gRTM (Generalized Relational Topic Model)



## LDA2Vec



## CTM (Correlated Topic Model)

















