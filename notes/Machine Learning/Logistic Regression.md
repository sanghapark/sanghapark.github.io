# Logistic Regression

Naive Assumption을 정의 하지 않고 Classification방식을 알아보자.

## Objectives

- Learn the logistic regression classifier
	- Understand why the logistic regression is better suited than the linear regression for classification tasks.
	- Understant the logistic function
	- Understand the logistic regression classifier
	- Understand the approximation approach for the open form solutions
- Learn the gradient descent algorithm
	- Know the taylor expansion
	- Understand the gradient descent/ascent algorithm
- Learn the difference between the naive Bayes and the logistic regression
	- Understand the similarity of the two classifiers
	- Understand the differences of the two classifiers
	- Understand the performance diffrences
# 1. Decision Boundary
## Optimal Classification and Bayes Risk
![1.1](/Users/kakao/Desktop/1.1.png)
Linear function vs Non-linear function of $P(Y\mid X)$

# 2. Introduction to Logistic Regression

## Sigmoid function

- Bounded, Differentiable, Real function, Defined for all real inputs, With positive derivative
- arctan, tanh, ...

간단한 로지스틱 함수를 사용하자.

$$f(x) = \frac{1}{1+e^{-x}}$$

## Logistic Function Fitting

선형회귀로 P(Y | X)를 추정 하려고 하면 01이여야 하는 확률 axiom에 맞지 않는다. 그래서 우리는 아래와 같이 변환을 한다.

$$f(x) = \log\frac{x}{1-x} \longrightarrow x = \log\frac{p}{1-p} \longrightarrow ax + b = \log\frac{p}{1-p} \longrightarrow X\theta = \log\frac{p}{1-p}$$

로지스틱 함수를 압축하거나 늘려주거나 왼쪽 오른쪽으로 시프트하기 위해서 x를 ax+b로 모델링 해주었다.

$$X\theta = \log\frac{P(Y\mid X)}{1-P(Y \mid X)}$$

## Logistic Regression

로지스틱 회귀는 probabilistic classifier이다. 

Bernoulli 시도에서
$$P(y\mid x) = \mu(x)^y(1-\mu(x))^{1-y}$$

$$\mu(x) = \frac{1}{1+e^{-\theta^Tx}} = P(y=1 \mid x)$$

$$X\theta = \log\frac{P(Y\mid X)}{1-P(Y \mid X)} \longrightarrow P(Y \mid X) = \frac{e^{X\theta}}{1 + e^{X\theta}} = \frac{1}{1 + e^{-X\theta}}$$


우리가 모르는 것은 theta이다. 이것을 알아 내는 것이 로지스틱 함수를 러닝하는 즉 파라미터를 인퍼런스 하는 작업이다. 

# 3. Logistic Regression Parameter Approximation 1

## Finding the Parameter $\theta$

선형회귀 분석에서는 미분하고 값을 0으로 해주고 모수를 최적화해줬다. 로지스틱 회귀도 이 방식으로 모수 $\theta$를 최적화 해보자.

### Maximum Likelihood Estimation (MLE) of $\theta$
$$\hat \theta = \underset{\theta}{\arg\max} P(D \mid \theta)$$


### Maximum Conditional Likelihood Estimation (MCLE)
$$
\begin{align}
\hat \theta 
& = \underset{\theta}{\arg\max} \\
& = \underset{\theta}{\arg\max} \prod_{1\leq i \leq N} P(Y_i \mid X_i; \theta) \\
& = \underset{\theta}{\arg\max} \log (\prod_{1\leq i \leq N}P(Y_i \mid X_i; \theta) \\
& = \underset{\theta}{\arg\max} \sum_{1\leq i \leq N} \log (P(Y_i \mid X_i; \theta)
\end{align}
$$

우리는 $P(Y_i \mid X_i; \theta)$에 대해서 아래와 같이 가정 할 수 있다.
$$
P(Y_i \mid X_i; \theta) = \mu(X_i)^{Y_i}(1-\mu(X_i))^{1-Y_i}
$$


아래 공식을 이용하여 
$$
P(y=1 \mid x) = \mu(x) = \frac{1}{1 + e^{-\theta^Tx}} = \frac{e^{X\theta}}{1 + e^{X\theta}}
$$

 로그를 씌워 풀면
$$
\begin{align}
\log(P(Y_i \mid X_i; \theta)
& = \log (\mu(X_i)^{Y_i}(1-\mu(X_i))^{1-Y_i}) \\
& = \cdots \\
& = Y_i X_i \theta + \log(1 - \mu(X_i)) \\
& = Y_iX_i\theta - \log(1 + e^{X_i\theta})
\end{align}
$$

편미분을 하고 0으로 세팅하고 $\theta​$에 대해 풀어 가정을 가장 강하게 만드는 모수를 찾아보자.
$$
\begin{align}
\frac{\partial}{\partial\theta\_j}\big\{ Y_iX_i\theta - \log(1 + e^{X_i\theta}) \big\}
& = \bigg\{ \sum_{1\leq i \leq N} Y_iX_{i, j} \bigg\} + \bigg\{ \sum_{1\leq i \leq N} - \frac{1}{1 + e^{X_i\theta}} \cdot e^{x_i\theta} \cdot X_{i, j} \bigg\}  \\
& = \sum_{1\leq i \leq N} X_{i, j} (Y_i - \frac{e^{X_i\theta}}{1 + e^{X_i\theta}}) \\
& = \sum_{1\leq i \leq N} X_{i, j} (Y_i - P(Y_i \mid X_i; \theta)) = 0
\end{align}
$$

위를 보면  $\theta$ 에 대해 풀기 쉽지 않게 되어있다. 우리는 이제  $\theta$를 approximate해야 한다.

# 4. Gradient Descent
etc에 있는 [Taylor Expansion & Gradient Descent](../../etc/Taylor Expansion & Gradient Descent-Ascent.md)를 참고하자.



# 5. Logistic Regression Parameter Approximation 2
로지스틱 리스레션에 gradient ascent 적용하기
리니어 리스레션에서 데이터셋이 너무 크면 클로즈드폼으로 계산 하지 말고 gradient descent로 찾기
$$
\hat \theta = \underset{\theta}{\arg\max} \sum_{1\leq i\leq N}\log(P)
$$


# 6. Naive Bayes to Logistic Regression
나이브 베이즈와 로지스틱 회귀의 차이점을 알아보자. 나이브 베이즈는 generative 로지스틱 회귀는 discriminative 한 pair이다. 우리는 나이브베이즈를 로지스틱회귀 함수로 도출 할 수 있을까? 가능 해서 서로 generative와 discriminative pair라고 불리운다.

Gaussian Naive Bayes의 각 Y 클래스들의 분산이 같다는 가정을 하면 GNU를 logistic regression 함수로 도출 할 수 있다.

# 7. Naive Bayes vs. Logistic Regression
나이브 베이즈 classifier는 찾아야 하는 모수(4d+1)가 많다. 하지만 로지스틱 리스레션의 경우 d+1개의 모수만 맞으면 된다. 그럼 누가 승자인가? 같은 데이터라면 로지스틱 회귀가 더 성능이 좋다. 그럼 이게 다인가? 아니다 나이브베이즈에는 우리가 사전 정보를 추가 할 수 있었다.

## Generative-Discriminative Pair
Generative model, $P(Y\mid X)= \frac{P(X, Y)}{P(X)} = \frac{P(X\mid Y)P(Y)}{P(X)}$
- Full probabistic model of all variables
	- Estimate the parameters of $P(X\mid Y), P(Y)$ from data
- Characteristics: Bayesian, Prior, Modeling the joint probability
- Naive Bayes Classifier
Discriminative model, $P(Y\mid X)$
- posterior를 바로 추정 하려고 하는 것이다.
- Do not need to model the distruribution of the observed variables
	- Estimate the parameters of $P(Y\mid X)$ from the data
- Characteristics: Modeling the conditional probability
- Logistic Regression
Pros and Cons
- Logistic regression is less biased
- Probably approximately correct learning: Naive Bayes learns faster













