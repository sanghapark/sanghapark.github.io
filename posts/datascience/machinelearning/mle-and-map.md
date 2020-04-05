# MLE & MAP

**Maximum Likelihood Estimation &** **Maximum a Posteriori Estimation**

---

## Problem

MLE(Maximum Likelihood Estimation)와 MAP(Maximum a Posteriori)의 차이를 동전 던지기 예제를 통해 알아 보자. 동전을 던졌을 때 앞면이 나올 확률은 얼마인가? 동전이 하나 주어져서 우린 $$N$$번을 던졌고 $$x$$번 앞면이 나왔다. 각 시도는 독립적인 사건이라고 가정한다.

---

## Bernoulli Distribution

결과가 두 가지 중 하나로만 나오는 것을 **베르누이 시도**(Bernoulli trial)라고 한다. 베르누이 시도의 결과를 확률 변수 $$X$$로 나타낼 때는 일반적으로 성공을 정수 1 ($$X=1$$), 실패를 0 ($$X=0$$)으로 정한다. 이번 동전 던지기 예제에서는 앞면을 1, 뒷면을 0으로 정한다.

베르누이 확률 변수는 0, 1 두 가지 값 중 하나만 가지므로 **이산 확률 변수**이다. 따라서 **확률 질량 함수**(Probability Mass Function)과 누적 분포 함수(Cumulative Distribution Function)으로 정의 할 수 있다. 베르누이 확률 변수는 1이 나올 확률 $$\theta$$라는 하나의 **모수**(parameter)만을 가진다. 0이 나올 확률은 $$1-\theta$$로 정의한다. 베르누이 확률 분포의 확률 질량 함수는 다음과 같다.
$$
\text{Bern}(x;\theta)=\begin{cases}
    \theta & \text{if $$x = 1$$}, \newline
    1-\theta & \text{if $$x = 0$$}
  \end{cases}
$$
or,
$$
\text{Bern}(x;\theta) = \theta^x(1-\theta)^{(1-x)}
$$
어떤 확률 변수가 $$X$$가 베르누이 분포에 의해 발생된다면 다음과 같이 수식으로 쓴다.
$$
X \sim \text{Bern}(x;\theta)
$$

---

## Binomial Distribution

성공 확률이 $$\theta$$인 베르누이 시도를 $$N$$번 시도하는 경우를 생각 해보자. $$N$$번 중 성공한 횟수를 확률변수 $$X$$ 라고 한다면 $$X$$의 값은 0부터 $$N$$까지의 정수 중 하나가 될 것이다.
$$
X \sim \text{Bin}(x; N, \theta)
$$
이항 분포 확률 변수 $$X$$의 확률 질량 함수를 구해보자. 우선 베르누이 확률 분포를 따른 확률 변수 $$Y$$를 가정한다.
$$
Y \sim \text{Bern}(y; \theta)
$$
이 확률 변수의 N개의 표본을 $$y\_1, y\_2, \cdots, y\_N$$라고 하자. 이 값은 0(실패) 아니면 1(성공) 이라는 값을 가지기 때문에 $$N$$번 중 성공한 횟수는 $$N$$개의 표본 값의 합(sum)이다.
$$
x = \sum\_{i=1}^{N}y\_i
$$
베르누이 분포를 따르는 확률변수 $$Y$$의 확률 질량 함수를 대입하여 정리하면 이항 분포 확률변수 $$X$$의 확률 질량 함수는 다음과 같다.
$$
\text{Bin}(x;N,\theta) = \binom{N}{x}\theta^x(1-\theta)^{N-x}
$$

---

## How to make our hypothesis stronger

동전을 던져 나온 결과물을 더욱 잘 설명 하는 가정을 만드는 방법은 무엇이 있을까? \
    1. 이항분포보다 더욱 잘 설명 할 수 있는 분포를 찾아 사용한다. \
    2. 이항분포를 따른 다는 가정을 한다면 주어진 데이터를 가장 잘 설명 할 수 있는 이항분포의 모수를 찾는다.

---

## Maximum Likelihood Estimation

관측된 데이터들의 확률을 최대화 하는 $$\theta$$를 찾는 것이다.
$$
\hat{\theta} = \underset{\theta}{\arg\max}P(D \mid \theta)
$$
어떤 $$\theta$$가 주어졌을때 데이터가 관측 될 확률,$$P(D \mid \theta)$$, 을 최대화 하는 $$\theta$$를 찾아내는 수식이다.

그럼 이제 동전 던지기 예제에 위에 MLE를 적용해보자.
$$
\begin{equation} \label{eq1}
\begin{split}
\hat{\theta}
& = \underset{\theta}{\arg\max}\, P(D \mid \theta) \newline
& = \underset{\theta}{\arg\max}\, \theta^x(1-\theta)^{N-x}
\end{split}
\end{equation}
$$

수식에 로그를 씌워 미분 하기 쉽게 만들어 주자. 로그는 단조 함수 이므로 수식에 로그 변환을 적용해도 최고점/최소점 위치는 변하지 않는다.
$$
\begin{align}
\hat{\theta}
& = \underset{\theta}{\arg\max}\,\ln P(D\mid \theta) \newline
& = \underset{\theta}{\arg\max}\,\ln \big(\theta^x(1-\theta)^{N-x}\big) \newline
& = \underset{\theta}{\arg\max}\, \big(x\ln\theta + (N-x)\ln(1-\theta)\big)
\end{align}
$$

확률, $$P(D\mid\theta)$$,를 극대화하기 위하여 수식을 1차 미분 하고 0으로 세팅한다. 그리고 $$\theta$$에 대해서 풀면 확률을 극대화하는 모수 $$\theta$$를 찾을 수 있다.
$$
\begin{align}
0
& = \frac{d}{d\theta}\big(x\ln\theta + (N-x)\ln(1-\theta)\big) \newline
& = \frac{x}{\theta} - \frac{N-x}{1-\theta}
\end{align}
$$
$$\theta$$에 대해 풀면
$$
\begin{align}
\theta
& = \frac{x}{(N-x)+x}\newline
& = \frac{x}{N}
\end{align}
$$
즉, 우리가 선택한 이항 분포 모형의 모수 $$\theta$$가 $$\frac{x}{N}$$일 때, $$N$$번 시도에 $$x$$번 앞면이 나오는 데이터가 생성 될 가능성이 가장 크다. 이것이 MLE 관점에서 본 모수 추정 방법이다.


우리는 $$\theta$$를 처음부터 정해진 하나의 값으로 생각하지 않고 $$\theta$$에 대한 사전 정보(믿음)를 가지고 모수 $$\theta$$에 대한 추정을 시작 할 수 있습니다. 예를 들어, 사전 정보로 앞면이 나올 확률을 50:50으로 생각 할 수 있습니다. 처음 몇번 동전 던지기를 시도 해서 나온 확률을 사전 정보를 가미한 $$\theta$$를 구할 수 있다.

아래와 같은 수식을 사용 하여 구할 수 있다.

---

## Bayes's Theorem

$$
\begin{align}
& P(\theta \mid D) = \frac{P(D \mid \theta)P(\theta)}{P(D)} \newline \newline
& Posterior = \frac{Likelihood * Prior\,Knowledge}{Normalizing\,Constant}
\end{align}
$$

우리는 이미 $$P(D\mid\theta) = \theta^x(1-\theta)^{N-x}$$로 정의를 위에서 하였다. $$P(\theta)$$는 동전을 몇번 던져보고 계산한 앞면이 나올 확률이 50%일꺼라는 사전 정보이다. 분모는 단순히 Normalizing Constant이므로 무시 할 수 있다. 이제 $$P(D\mid\theta)$$를 구해보자.

$$
P(\theta\mid D) \propto P(D\mid\theta)P(\theta)
$$

그전에 우리는 우선 $$P(\theta)$$를 잘 정의 헤야 한다. $$P(D|\theta)$$를 이항분포로 잘 정의 하였듯이 $$P(\theta)$$도 어떤 확률 분포로 정의를 해주어야한다. 베타 분포를 사용하자.

---

## Beta Distribution

0에서 1사이의 값을 가지는 Cumulative Mass Function을 가지고 있어서 확률의 성격을 잘 묘사 할 수 있다.
$$
\begin{align}
P(\theta) & = \frac{\theta^{\alpha -1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)} \newline \newline
B(\alpha,\beta) & = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)} \newline \newline
\Gamma(\alpha) & = (\alpha - 1)!
\end{align}
$$

---

## Conjugate Prior

이제 사전정보와 likelihood를 사용하여 사후 분포를 구해보자. $$B(\alpha, \beta)$$는 $$\alpha$$와 $$\beta$$가 결정되어 있는 상황에서는 상수가된다. $$\theta$$에 의존하는 항이 아니기에 무시 할 수 있다.
$$
\begin{align}
P(\theta\mid D)
& \propto P(D\mid\theta)P(\theta) \newline
& \propto \theta^x(1-\theta)^{N-x}\theta^{\alpha -1}(1-\theta)^{\beta-1} \newline
& = \theta^{x + \alpha -1}(1-\theta)^{N-x+\beta-1}
\end{align}
$$

사전분포와 사후 분포가 동일한 분포족에 속하는 것을 볼수있다. 이때 사전분포를  **Conjugate Prior**라고 한다.

---

## Maximum a Posteriori Estimation

이제 주어진 사후 분포에서 $$\theta$$에 대해 극대화를 하자.
$$
\begin{align}
\hat\theta
& = \underset{\theta}{\arg\max}\, P(\theta\mid D) \newline
& = \underset{\theta}{\arg\max}\, \theta^{x + \alpha -1}(1-\theta)^{N-x+\beta-1}
\end{align}
$$

위의 식을 미분 하고 0을로 세팅하고 $$\theta$$에 대해서 풀면 다음과 같다.
$$
\begin{align}
\hat\theta
& = \frac{x+\alpha-1}{x+\alpha+(N-x)+\beta-2} \newline
& = \frac{x+\alpha-1}{\alpha+N+\beta-2}
\end{align}
$$

---

## Conclusion

$$
\begin{align}
\text{MLE:}\quad & \hat\theta = \frac{x}{N}\newline
\text{MAP:}\quad & \hat\theta = \frac{x+\alpha-1}{\alpha+N+\beta-2}
\end{align}
$$

MLE 방식으로 추론된 수식과 MAP 방식으로 추론된 수식이 많이 달라 보이지만 MAP 수식에서 $$N$$이 커질 수록 $$\alpha$$와 $$\beta$$의 영향은 소멸 할 것이고 MLE와 MAP의 값들은 같아 질 것이다. $$\alpha$$와 $$\beta$$와 같은 초모수(Hyperparameter)는 분석하는 사람의 주관적으로 정해질 수있다.
