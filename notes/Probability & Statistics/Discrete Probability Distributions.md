# Discrete Probability Distributions



## Bernouli Distribution



## Binomial Distribution



## Category Distribution

카테고리 분포는 베르누이 분포의 확장판이다. 2가지 옵션을 가진 동전은 베르누이 분포로 묘사 할 수 있었다. 6가지의 옵션을 가진 주사위는 카테고리 분포를 사용하여 묘사 할 수 있다.

카테고리 분포는 1부터 $K$ 개의 정수 값중 하나가 나오는 확률 변수의 분포이다.  카테고리 분포를 가진 확률변수는 스칼라 값을 출력하는 확률 변수 이지만 다음과 같이 0과 1으로만 이루어진 다차원 벡터로 변형하여 사용한다.
$$
x = 1 \rightarrow = (1, 0,0,0,0,0) \\
x = 2 \rightarrow = (0, 1,0,0,0,0) \\
x = 3 \rightarrow = (0, 0,1,0,0,0) \\
x = 4 \rightarrow = (0, 0,0,1,0,0) \\
x = 5 \rightarrow = (0, 0,0,0,1,0) \\
x = 6 \rightarrow = (0, 0,0,0,0,1) \\
$$
이러한 인코딩 방식을 **One-Hot-Encoding** 이라고 한다. 다음과 같이 벡터로 표현 할 수 있다.
$$
x = (x_1, x_2, x_3, \cdots , x_{K-1}, x_K) \\
\quad \text{s.t.} \\
\quad x_i = 
\begin{cases}
0 \\
1
\end{cases}
\quad \text{and} \quad \sum_{k=1}^K x_k = 1
$$
각각의 원소 값 $x_k$는 일종의 베르누이 확률 분포로 볼 수 있다. 각각 1이 나올 확률을 나타내는 모수 $\mu_k$ 를 가진다. 전체 카테고리 분포의 모수는 다음과 같이 나타낼 수 있다.
$$
\begin{align}
& \mu = (\mu_1, \mu_2, \cdots, \mu_K) \\
& \text{s.t.} \\
& 0 \leq \mu_i \leq 1 \quad \text{and} \quad \sum_{k=1}^K \mu_k = 1 
\end{align}
$$
카테고리 분포는 다음과 같이 표기한다.
$$
\text{Cat}(x_1, x_2, \cdots, x_K \ \vert \ \mu_1, \mu_2, \cdots , \mu_K) \quad \text{or} \quad \text{Cat}(\bold x; \bold \mu)
$$
**확률 질량 함수** 는 다음과 같이 쓴다.
$$
\text{Cat}(\bold x; \mu) = 
\begin{cases}
& \mu_1 & \text{if} \ x = (1, 0, 0, \cdots, 0) \\
& \mu_1 & \text{if} \ x = (0, 1, 0, \cdots, 0) \\
& \vdots & \vdots \\
& \mu_K \quad & \text{if} \ x = (0, 0, 0, \cdots, 1)
\end{cases}
$$
위식을 다음과 같이 쓸 수 있다. One-Hot-Encoding을 사용한 덕분이다.
$$
\text{Cat}(\bold x, \mu) = \mu_1^{x_1}\mu_2^{x_2}\cdots\mu_K^{x_K} = \prod_{k=1}^K \mu_k^{x_k}
$$

### 카테고리 분포의 모멘트

- 기댓값
  $$
  \text{E}[x_k] = \mu_k
  $$

- 분산
  $$
  \text{Var}[x_k] = \mu_k(1-\mu_k)
  $$
  

## [Multinomial Distribution](#Multinomial Distribution)

베르누이 시도를 여러번 하여 얻은 총 성공 횟수합이 이항 븐포를 이루는 것처럼 독립적인 카테고리 분포를 여러번 시도하여 각 원소의 성공횟수 값은 **다항 분포**가 된다.

다항 분포는 카테고리 시도를 $N$ 번 반복하여 $K$개의 각각의 원소가 $x_k$ 번 나올  확률 분포를 말한다. 표본 값이 벡터 $x = (x_1, \cdots, x_K)$ 가 되는 확률 분포를 말한다.

예를 들어, $x=(1, 2, 1, 2, 3, 1)$ 은 6개의 숫자가 나올 수 있는 주사위를 10번 던져서 1인 면이 1번, 2인 면이 2번, 3인 면이 1번, 4인 면이 2번, 5인 면이 3 번, 6인 면이 1번 나왔다는 뜻이다.

다항 분포의 **확률 질량 함수** $\text{Mu}(\bold x; N, \mu)$ 를 사용하여 다음과 같이 표기한다. 

- [ ] TODO: 조합식이 의미 하는바 이해하기. 어떻게 derive 되었는지 생각해보자.

$$
\begin{align}
\text{Mu}(\bold x ; N, \mu) 
& = {N \choose \bold x} \prod_{k=1}^K \mu_k^{x_k} \\
& = {N \choose x_1, \cdots, x_K}\prod_{k=1}^K \mu_k^{x_k}
\end{align}
$$

조합 기호는 다음과 같이 정의된다.
$$
{N\choose x_1, x_2, \cdots, x_k} = \frac{N!}{x_1!\cdots x_k!}
$$
다른 방식으로 수식을 전개 해보자.
$$
\begin{align}
P(\bold X \ \vert \ \bold{\mu}) 
& = \prod_{n=1}^N \prod_{k=1}^K\mu_k^{x_{nk}} \\
& = \prod_{k=1}^K \mu_k^{\sum_{n=1}^N x_{nk}} \\
& = \prod_{k=1}^K \mu_k^{m_k} \quad \text{s.t.} \quad m_k = \sum_{n=1}^Nx_{nk}
\end{align}
$$
데이터가 주어졌으면 우리는 데이터를 가장 잘 설명 할 수 있는 $\bold \mu$ 를 어떻게 추론 할 수 있을까? MLE와 MAP 두가지 다 사용 할 수 있다.



### 다항 분포의 모멘트

- 기댓값
  $$
  \text{E}[x_k] = N\mu_k
  $$

- 분산
  $$
  \text{Var}[x_k] = N\mu_k(1-\mu_k)
  $$









































