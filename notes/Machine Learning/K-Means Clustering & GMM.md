# K-Means Clustering & Gaussian Mixture Model



GMM을 그래피컬 모델로 설명 할 수 있다. 그래서 베이지안 네트워크를 배움

## Objectives

- Understand the **clustering task** and the **K-menas algorithm**

  - Know what the **unsupervised learnin**g is
  - Understand the K-means interative process
  - Know the limitation of the K-means algorithm

- Understand the **Gaussian mixture model**

  - Know the multinomial distribution and the multivariate Gaussian distribution
  - Know why mixture models are useful
  - Understand how the parameter updats are derived from the Gaussian mixture model

- Understand the **EM algorithm**

  - Know the fundamentals of the EM algorithm
  - Know how to derive the EM updates of a model

  

## Unsupervised Learning

군집을 찾는 것이다. 잠재적인 어떤 factor가 있어서 군집들을 만들어낸다고 가정을 할 수 있다. 아래 그림을 참고해보자. 군집을 이루는 포인트들은 서로 latent factor를 공유 하고 있는 관측들이라고 짐작 할 수 있다.

<img src="/Users/kakao/Documents/notes/Machine Learning/assets/image-20190220085241537.png" width=500/>

## K-Means Algorithm

K개의 동력(latent factor)가 있어서 관측 포인트들이 생성이 되었다라는 가정을 한다. 그럼 관측이 된 포인트들은 어떻게 내부 동력에 optimal하게 assign 할 수 있을까?라는 것이 우리가 생각해야 하는 부분이다. 어떤 중심점 (Centroid)가 있고 그것을 기분으로 노이즈(?)를 포함한 포인트들이 관측이 되는 것이다. 우리는 이 중심점을 알아 내야 한다. 이 문제는 2단계의 문제로 구성 되어있다.

1. Setup K number of centroids
2. Cluster data points by the distance from the points to the neareast centroid

Objective function을 만들어 보자.
$$
\begin{align}
& \min \sum_{n=1}^N\sum_{k=1}^K r_{nk} \Vert x_n - \mu_k \Vert^2 \\
& \quad \quad \quad \quad \text{s.t.} \\
& r_{nk} = 0 \quad \text{or } \quad 1, \quad \text{the assignment of data points to clusters} \\
& \mu_k: \text{the location of k-th centroid}
\end{align}
$$
 $r_{nk}$ 와 $\mu_k$ 에대해서 우리는 잘 모른다. 우리가 어떤 알고리즘에 의해서 찾아야 하는  것들이다. 예전에는 찾아야 하는 parameter가 하나였다. 미분을 통해서 최적화를 했다. 이제 변수가 두개가 되어 버렸다. 반복적 최적화(Iterative Optimization)을 통해서 할 수 있다. $r_{nk}$ 를 최적화 하고 최적화된 $r_{nk}$ 를 활용하여 $\mu_k$ 에 대해서 최적화를 하고 최적화된 $\mu_k$ 를 활용해서 $r_{nk}$ 에 대해서 또 최적화를 반복해줄 수 있다.

### Expectation and Maximization

$$
J = \sum_{n=1}^N\sum_{k=1}^K r_{nk} \Vert x_n - \mu_k \Vert^2
$$

Latent variable에 대해서 반복적으로 최적화 해서 목적함수를 최소화 하는 것이 **Expectation and Maximization**이라고 한다. 

**Expectation**: $ r_{nk}$ 를 최적화하는 과정

- Expectation of the log-likelihood given the parameters
- Assign the data points to the nearest centroid

**Maximization**: 주어진 $ r_{nk}$ 를 기준으로 centroid들을 최적화하는 과정

- Maximization of the parameters with respect to the log-likelihood
- Update the centroid positions $ \mu_k$ given the assignments $r_{nk}$

$\mu_k$ 는 처음에 랜덤하게 주어진다. 랜덤하게 주어진 centroid들에 $r_{nk}$ 를 최적화 한다. 그리고 업데이트된 $r_{nk}$ 를 기준으로 $\mu_k$ 를 업데이트 한다. 이것을 반복한다.

$r_{nk}$ 은 불연속 변수이고 0 혹은 1의 값을 가진다. 하지만 딱 잘라서 0 혹은 1로 판단하기 어려울수 있다. Soft EM으로 풀수 있다. Hard EM으로 우선 시작해보자. 데이터 포인트 $x_n$ 은 가장 가까운 centroid $\mu_k$ 를 선택한다. $\mu_k$ 는 다음과 같이 최적화 할 수 있다. k  summation은 $\mu_k$ 의 k와 일치하지 않는 인덱스는 0이기에 제거 해줄수 있다.

TODO: Soft EM에 대해 알아보자.
$$
\begin{align}
\frac{d J}{d\mu_k} 
& = \frac{d}{d\mu_k} \sum_{n=1}^N \sum_{k=1}^K r_{nk} \Vert x_n - \mu_k \Vert ^2 \\
& = \frac{d}{d\mu_k} \sum_{n=1}^N r_{nk} \Vert x_n - \mu_k \Vert ^2 \\
& = \sum_{n=1}^N -2r_{nk}(x_n - \mu_k)  \\
& = -2(-\sum_{n=1}^N r_{nk}\mu_k + \sum_{n=1}^N r_{nk}x_n) \\
& = 0 \\
\end{align}
\\
\therefore \quad \mu_k = \frac{\sum_{n=1}^N r_{nk}x_n}{\sum_{n=1}^N r_{nk}}
$$
최적화된 $\mu_k$ 는 $\mu_k$ 에 해당하는 데이터 포인드들의 평균값이다.

### Properties of K-Means Algorithm

- K를 어떻게 정할까? Bayesian non-parametric을 하면 할수 있다. TODO: Bayesian non-parametric 알아보자.
- Centroid들의 초기위치를 어떻게 정할까? 평균에 위치 해줄수도 있고 램덤으로 해줄수도 있다. 초기 위치가 안 좋으면 local optima에 빠질수 있다.
- 두 포인트 사이의 **유클리드 거리**를 사용한다. 좋은 hypothesis는 아니다. 각 축의 중요도가 다를수 있다. 예를 들어 y축의 변화보다 x축의 변화가 더 중요 할 수 있다.
- $r_{nk} = 0 \ \text{or} \ 1 $ 인 Hard clustering의 문제가 있다. Soft clustering을 사용 할 수 있다. Gaussian Mixture Model을 사용 해서 극복할 수 있다. GMM은 고정된 K나 centroid들의 초기 위치 문제를 해결 해주지는 않는다.



##GMM (Gaussian Mixture Model)

GMM을 학습하기 전에 먼저 두가지 Multinomial Distribution과 Multivariate Gaussian Distribution에 대한 선수 지식이 필요하다. 분포들이 섞여있는 상황에서 하나의 분포가 선택되는 확룰을 다항 분포 그리고 각 분포를 가우시안 분포로 모델링 할 것이다.

---

### Multinomial Distribution

[Multinomial Distribution](../Probability & Statistics/Discrete Probability Distributions.md) 를 참고하자. $K​$ 개의 값을 가질수 있는 확률 변수에서 $N​$ 개의 데이터가 주어졌다.  다항분포는 다음과 같이 수식으로 정리 할 수 있다.
$$
\begin{align}
P(\bold X \ \vert \ \boldsymbol{\mu}) 
& = \prod_{n=1}^N \prod_{k=1}^K\mu_k^{x_{nk}} \\
& = \prod_{k=1}^K \mu_k^{\sum_{n=1}^N x_{nk}} \\
& = \prod_{k=1}^K \mu_k^{m_k} \quad \text{s.t.} \quad m_k = \sum_{n=1}^Nx_{nk}
\end{align}
$$
데이터가 주어졌으면 우리는 데이터를 가장 잘 설명 할 수 있는 $\bold \mu$ 를 어떻게 추론 할 수 있을까? MLE와 MAP 두가지 다 사용 할 수 있다.
$$
\begin{align}
& \max P(\bold X \vert \boldsymbol \mu) = \prod_{k=1}^K\mu_k^{m_k} \\
& \text{s.t.} \\
& \mu_k \geq 0, \quad \sum_{k=1}^K \mu_k = 1
\end{align}
$$
제약 조건이 주어진 수식을 MLE 통해서 local maximum을 계산 해보자. 제약 조건이 주어진 수식은 우선 [Lagragian Method (라그랑주 승수법)](../Mathematics/Lagrange Multiplier.md)로 변형해서 MLE 계산 할 수 있다. 참고로 더 자세히 알고 싶다면 [Continuous Optimization](../Optimization/Continuous Optimization.md) 를 통해 최적화 기법을 알아 볼 수 있다.

---

### MLE of Multinomial Distribution

1. Lagrangian method를 사용하여 수식을 변형 해주자.
   $$
   L(\mu, m, \lambda) = \prod_{k=1}^K \mu_k ^{m_k} + \lambda(\sum_{k}^K\mu_k - 1)
   $$
   미분 계산 편의를 위해 로그를 씌워주자.
   $$
   L(\mu, m, \lambda) = \sum_{k=1}^K m_k \ln \mu_k + \lambda(\sum_{k=1}^K \mu_k - 1)
   $$

2. $\mu_k$에 대해 1차 미분을 하고 0으로 만들어 주고 $\mu_k$ 에 대해 풀어 최적화 한다.
   $$
   \frac{d}{d\mu_k} L(\mu, m, \lambda) = \frac{m_k}{\mu_k} + \lambda = 0 \\
   \therefore \mu_k = -\frac{m_k}{\lambda}
   $$

3. $\lambda$ 를 찾자.
   $$
   \quad \quad \sum_{k=1}^K \mu_k = 1 
   \rightarrow 
   \sum_{k=1}^K -\frac{m_k}{\lambda} = 1
   \rightarrow 
   \sum_{k=1}^Km_k = -\lambda
   \rightarrow 
   \sum_{k=1}^K \sum_{n=1}^N x_{nk} = -\lambda \\
   \therefore \lambda = -N
   $$

정리하면 다음과 같다.
$$
\mu_k = \frac{m_k}{N}
$$
주어진 데이터를 가장 잘 설명 할 수 있는 각 원소의 최적화 $\mu_k​$ 는 해당 각 원소가 관측된 갯수 나누기 총 데이터의 갯수이다. 

---

### Multivariate Gaussian Distribution

1차원의 가우시안 분포는 다음과 같다.
$$
\mathcal{N}(x, \vert \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi\sigma^2}}\exp\bigg( -\frac{1}{2\sigma^2} (x- \mu)^2 \bigg)
$$
다변수 가우시안 분포는 다음과 같다.
$$
\boldsymbol{\mathcal {N}}(\bold x \vert \boldsymbol \mu, \Sigma) =
\frac{1}{(2\pi)^{D/2}} \frac{1}{\vert \boldsymbol \Sigma\vert^{1/2}}
\exp\bigg( -\frac{1}{2} (\bold x - \boldsymbol \mu)^\intercal \boldsymbol \sigma^{-1} (\bold x - \boldsymbol \mu) \bigg)
$$

---

### MLE of Multivariate Gaussian Distribution

로그를 씌워 MLE를 계산하여보자.
$$
\begin{align}
\ln \boldsymbol{\mathcal{N}}(\bold x \vert \boldsymbol \mu, \boldsymbol \Sigma) & = 
-\frac{1}{2} \ln \vert \boldsymbol \Sigma\vert - \frac{1}{2}(\bold x - \boldsymbol \mu)^\intercal \boldsymbol \Sigma^{-1} (\bold x - \boldsymbol \mu) + \bold C \\

\ln \boldsymbol{\mathcal{N}}(\bold X \vert \boldsymbol \mu, \boldsymbol \Sigma) & = 
-\frac{N}{2}\ln\vert \boldsymbol \Sigma \vert - \frac{1}{2} \sum_{n=1}^N(\bold x_n - \boldsymbol \mu)^\intercal \boldsymbol \Sigma^{-1}(\bold x_n - \boldsymbol \mu) + \bold C \\

& \propto -\frac{N}{2}\ln\vert \boldsymbol \Sigma \vert - \frac{1}{2} \sum_{n=1}^N(\bold x_n - \boldsymbol \mu)^\intercal \boldsymbol \Sigma^{-1}(\bold x_n - \boldsymbol \mu) \\

& = -\frac{N}{2}\ln \vert \boldsymbol \Sigma \vert - \frac{1}{2}\text{Tr}\bigg[ \boldsymbol \Sigma^{-1} \sum_{n=1}^N \big( (\bold x_n - \boldsymbol \mu)(\bold x_n - \boldsymbol \mu)^\intercal \big)\bigg]

\end{align}
$$
두가지 변수 $\boldsymbol \mu$ 와 $\boldsymbol \Sigma$ 에 대해 편미분하고 0으로 만들어서 풀어 최적화 하자. Trace Trick을 사용해야 한다.

> Trace Trick
>
> 1. $\frac{d}{dA}\log \vert A \vert  = A^\intercal$ 
> 2. $\frac{d}{dA}\text{Tr}[AB] = \frac{d}{dA}\text{Tr}[BA] = B^\intercal$

$$
\frac{d}{d\boldsymbol \mu} \ln \boldsymbol{\mathcal{N}}(\bold X \vert \boldsymbol \mu, \boldsymbol \Sigma) = 0 \longrightarrow -\frac{1}{2} \times 2 \times -1 \times \boldsymbol \Sigma^{-1}\sum_{n=1}^N(\bold x_n - \hat{\boldsymbol \mu}) = 0 \quad \therefore\quad \hat{\boldsymbol{\mu}} = \frac{\sum_{n=1}^N x_n}{N} \tag1
$$

$$
\frac{d}{d\boldsymbol \Sigma^{-1}}\ln \boldsymbol{\mathcal{N}}(\bold X \vert \boldsymbol \mu, \boldsymbol \Sigma) = 0 \quad \therefore \quad \hat{\boldsymbol{\Sigma}} = \frac{1}{N}(\bold x_n - \hat{\boldsymbol \mu})(\bold x_n - \hat{\boldsymbol \mu})^\intercal \tag2
$$

---

### Mixture Model

![image-20190223163729074](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190223163729074.png)

위의 그래프에서 하나의 가우시안으로 피팅하면 파란색 같이 피팅이 된다. 이것은 세가지의 봉우리가 있는 서로 근본이 달라 보이는 분포를 잘 묘사 하지 못한다. 서로 다른 분포라고 생각을 하고 각각 가우시안으로 피팅 할 수 있다. 그리고 아래와 같이 합칠 수 있다.
$$
P(x) = \sum_{k=1}^K \pi_k \mathcal N(x \ \vert \ \mu_k, \sigma_k)
$$
Mixing coefficients, $\pi_k$, 를 각 분포마다 할당해서 확률로 만들어 주는 것이다. 즉, K개의 선택지 중에서 하나가 선택될 확률이다.
$$
\sum_{k=1}^K \pi_k = 1 \quad \text{and} \quad 0 \leq \pi_k \leq 1
$$
다항분포로 mixing coefficients $\pi_k$ 를 모델링 하고 $\mathcal N$ 은 Multivatiate Gaussian 분포로 모델링 할 수 있다. 분포의 선택지를 $z$ 확률 변수를 사용해서 모델링 하자.
$$
P(x) = \sum_{k=1}^K P(z_k)P(x \ \vert \ z)
$$
조건 $z$ 는 선택된 분포를 의미한다.

---

### Gaussian Mixture Model

Gaussian Mixture Model은 $K​$ 개의 가우시안 분포중 하나를 고르고 골라진 분포에서 데이터 포인트 하나를 샘플링을 하는 식으로 모델링을 한다. 
$$
P(x) = \sum_{k=1}^K P(z_k) P(x \vert z) = \sum_{k=1}^K \pi_k \mathcal{N}(x \vert \mu_k, \Sigma_k)
$$
K-Means Clustering에서 $r_{nk}$ 는 0 혹은 1의 값을 가지는 hard-clustering이다. 하지만 여기서 $P(z_k)$ 는 선택은 hard하게 이루어지만 확률적(stochastic)으로 이루어진다. GMM의 두가지 부분을 모델링 해보자.

- Mixing Coefficients, or Selection Variable, $z_k$
  $$
  \begin{align}
  & z_k \in \{0, 1\} \\
  & \sum_{k=1}^Kz_k = 1 \\
  & P(z_k = 1) = \pi_k \\
  & \sum_{k=1}^K \pi_k = 1 , \quad 0 \leq \pi_k \leq 1 \\
  & P(Z) = \prod_{k=1}^K \pi_k^{z_k}
  \end{align}
  $$

- Mixture Component
  $$
  P(X \vert z_k = 1) = \mathcal{N}(x \vert \mu_k, \Sigma_k) \longrightarrow P(X \vert Z) = \prod_{k=1}^K \mathcal{N}(x\vert \mu_k, \Sigma_k)^{z_k}
  $$

**베이지안 네크워크**를 사용하여 GMM을 표현 해보자.

<img src="/Users/kakao/Documents/notes/Machine Learning/assets/diagram-20190223 (2)-0923414.png" width=200>

파란색은 parameter 형태로 표현 되었다. 하지만 베이지안 버전으로 바꾸서 생각하면 파란색들이 확률변수(random variable)가 되면서 부모노드로 prior distribution  노드가 붙는다. 주황색 노드는 관측이 된 상태인 것을 의미한다. $X$가 관측 되는데 영향은 미치는 확률 변수는 $Z$가 있고 그 과정에서 필요한 parameter들은  $\mu$, $\Sigma$ 가 있다. $N$개의 관측된 데이터가 있고 $N$ 개의 포인트마다 클러스터에 assignment해야되는 z가 존재한다.

**지금까지는 어떤 mixing component가 선택 됬으면  $x$가 얼마만큼의 확률로 샘플이 될까로 접근했었다. 이제 어떤 데이터 포인트 $x$ 가 주어지면 각 mixing component에 속할 확률은 무엇인지로 접근해보자.**
$$
\begin{align}
\gamma(z_{nk}) = p(z_k = 1 \ \vert \ x_n) 
& = \frac{p(x \vert z_k = 1)p(z_k = 1)}{\sum_{j=1}^Kp(x \ \vert \ z_j = 1)p(z_j = 1)} \\
& = \frac{\pi_k \mathcal{N}(x \ \vert \ \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x \ \vert \ \mu_j, \Sigma_j)}
\end{align} 
\tag 3
$$
K-Means clustering에서는 $K​$ 개의 centroid들에게 optimal 하게 배치하고 속한 centroid 중심부터의 Euclidean distance를 최소화하는 방식으로 최적화 하였다. 이번에는 확률 프레임워크를 상에서 최적화를 해보자. 관측된 데이터가 나타날 확률을 최대화 시켜줄수 파라미터 $\pi, \mu, \Sigma​$ 들을 찾을수 있다.
$$
\underset{\pi, \mu , \Sigma}{\arg \max} \ln P(\bold X \ \vert \ \pi, \mu, \Sigma) = \sum_{n=1}^N \ln \bigg\{ \sum_{k=1}^K  \pi_k \mathcal{N}(x \ \vert \ \mu_k, \Sigma_k) \bigg\}
$$

## Expectation and Maximization of GMM

GMM의 Expection 과 Maximization의 각각 스텝을 자세히 알아보자.

- Expectation은 데이터들을 클러스터들에게 배치하는 작업
- Maximization은 모수들, $\pi, \mu , \Sigma$, 을 업데이트 하는 작업

### Expectation of GMM

 K-means 클러스터링에서는 데이터를 가장 가까운 클러스터에 배치하였다. 하지만 GMM에서는 확률을 준다. **Assignment 확률**은 다음과 같다.
$$
\gamma(z_{nk}) \equiv p(z_k = 1 \ \vert \ x_n) 
= \frac{p(z \ \vert \ z_k = 1)p(z_k = 1)}{\sum_{j=1}^Kp(x \ \vert \ z_j = 1)p(z_j =1)} 
= \frac{\pi_k \mathcal{N}(x \ \vert \ \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x \ \vert \ \mu_j, \Sigma_j)} \tag4
$$
처음에 랜덤하게 $x, \pi, \mu, \Sigma$ 가 주어지면 우리는 $\gamma(z_{nk})$ 를 계산 할 수 있다. 그리고 새로이 계산된 $\gamma(z_{nk})$ 를 활용하여 $\pi, \mu, \Sigma$ 를 업데이트 할 수 있다. 그리고 업데이트된 모수들을 활용해서 다시 assignment probability $\gamma(r_{nk})​$ 를 업데이트 한다. 이것을 반복적으로 하는 것이 EM algorithm이다.

### Maximization of GMM

Expectation 스텝에서 구한 $\gamma(z_{nk})$ 를 사용해서 모수들($\pi, \mu, \Sigma $)을 업데이트하자.
$$
\ln P(\bold X \ \vert \ \pi, \mu, \Sigma) = \sum_{n=1}^N \ln \bigg\{ \sum_{k=1}^K  \pi_k \mathcal{N}(x \ \vert \ \mu_k, \Sigma_k) \bigg\}
$$


- $\mu_k​$: 가운시안 분포 평균
  $$
  \begin{align}
  \frac{d}{d\mu_k} \ln P(\bold X \ \vert \ \pi, \mu, \Sigma) 
  & = \sum_{n=1}^N\frac{\pi_k \mathcal{N}(x\vert\mu_k, \Sigma_k)}{\sum_{j=1}^{K}\pi_j \mathcal{N}(x\vert\mu_j, \Sigma_j)}\Sigma^{-1}(x_n - \hat{\mu_k}) = 0 \\
  & \rightarrow \sum_{n=1}^N \gamma(z_{nk})(x_n - \hat{\mu_k}) = 0 \\
  & \therefore \widehat{\mu_k} = \frac{\sum_{n=1}^N \gamma(z_{nk})x_n}{\sum_{n=1}^N\gamma(z_{nk})}
  \end{align}
  $$

- $\Sigma_k$: 가우시안 분포의 Covariance Matrix
  $$
  \begin{align}
  & \frac{d}{d\Sigma_k} \ln P(\bold X \vert \pi, \mu, \Sigma) = 0 \\
  & \therefore \widehat{\Sigma_k} = \frac{\sum_{n=1}^N\gamma(z_{nk})(x_n - \widehat{\mu_k})(x_n - \widehat{\mu_k})^\intercal}{\sum_{n=1}^N\gamma(z_{nk})}
  \end{align}
  $$

- $\pi_k$:  Mixing coefficients 
  $$
  \begin{align}
  & \frac{d}{d\pi_k} \ln P(\bold X \vert \pi, \mu, \Sigma) + \lambda\big(\sum_{k=1}^K \pi_k - 1\big) = 0 \\
  & \longrightarrow \sum_{n=1}^N \frac{\mathcal{N}(x\vert \mu_k, \Sigma_k)}{\sum_{j=1}^K\pi_j \mathcal{N}(x\vert \mu_j, \Sigma_j)} + \lambda = 0 \\
  & \longrightarrow \sum_{k=1}^K \bigg\{ \sum_{n=1}^N  \frac{\pi_k \mathcal{N}(x\vert \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x\vert \mu_j, \Sigma_j)} + \pi_k\lambda \bigg\} = 0
  \quad \text{with} \quad \sum_{k=1}^K \bigg\{ \sum_{n=1}^N  \frac{\pi_k \mathcal{N}(x\vert \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x\vert \mu_j, \Sigma_j)}\bigg\} = N \\ \newline
  & \quad \text{because of the summation of assignment probabilities over clueters over N data points}  \\ \newline
  & \therefore \widehat{\pi_k} = \frac{\sum_{n=1}^N\gamma(z_{nk})}{N} \
  \end{align}
  $$

### Pros and Cons of GMM

- Pros
  - Soft Clustering
  - Learn the latent distribution: unsupervised learning에서는 latent factor에 대해서 배우는것이 essence이다.
- Cons
  - Long computation time due to the covariance matrix modeling
  - Local maximum
  - Decide K: parametric 모델이기 때문에 K는 정해 질수 밖에 없다. TODO: GMM을 Bayesian GMM으로 만들고 bayesian non-parametric 모델의 접근 방법을 붙히면 해결 할 수 있다.

## Relation between K-Means and GMM



## Inference with Latent Variables



### Probability Decomposition

> **Jensen's Inequality**
>
> 



### Maximizing the Lower Bound (1)



> **KL-Divergence**
>
> 



### Maximizing the Lower Bound (2)

































