## Data Modeling

![스크린샷 2019-07-27 오후 2.20.20](/Users/kakao/Desktop/스크린샷 2019-07-27 오후 2.20.20.png)

- Supervised vs. Unsupervised: Blocks #1 and #4
- Probabilistic vs. non-probabilistic: Primarily Block #2 (Some Block #3)
- Model development (Block #2) vs. Optimization techniques (Block #3)



## Gaussian Distribution (Multivariate)

Gaussian density in $d$ dimensions

- Block #1: Data $x_1, \cdots, x_n$. Each $x_i \in \R^d$
- Block #2: An indenpendent and identically distributed (i.i.d) Gaussian model
- Block #3: Maximum likelihood
- Block #4: Leave undefined

The density function is 
$$
p(x\mid \mu, \Sigma) = \frac{1}{(2\pi)^{\frac{d}{2}}\sqrt{\text{det}(\Sigma)}} \exp\big( -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) \big)
$$
The central moments are
$$
\begin{align}
\mathbb{E}[x] & = \int_{\R^d}x\cdot p(x \mid \mu, \Sigma)dx = \mu \\
\text{Cov}(x) & = \mathbb{E}[(x-\mathbb[x])(x-\mathbb{E}[x])^T]
= \mathbb[xx^T] - \mathbb[x]\mathbb[x]^T = \Sigma
\end{align}
$$


## BLOCK #2: A Probabilistic Model

#### Probabilistic Models

- A probabilistic model is a set of probability distributions, $p(x\mid \theta)$
- We pick the distribution family $p(\cdot)$, but don't know the parameter $\theta$.

Example: Model data with a Gaussian distribution $p(x|mid \theta), \theta = \{\mu, \Sigma\}$.

#### The i.i.d assumption

Assume data is independent and identically distributed (iid). This is written
$$
x_i \stackrel{iid}{\sim} p(x\mid \theta), \quad i = 1, \cdots, n.
$$
Writing the density as $p(x\mid \theta)$, then the joint density decomposes as
$$
p(x_1, \cdots, x_n\mid \theta) = \prod_{i=1}^{n}p(x_i \mid \theta)
$$

## BLOCK #3: Maximum Likelihood Estimation

#### Maximum Likelihood approach

We now need to find $\theta$. **Maximum Likelihood** seeks the value of $\theta$ that maximizes the likelihood function:
$$
\hat{\theta}_{\text{ML}} = \arg \max_\theta p(x_1, \cdots, x_n \mid \theta)
$$
가정한 모델의 $\theta$ 를 MLE를 통해서 찾은 $\hat{\theta}_{\text{ML}}$ 이 주어진 데이터를 가장 잘 설명한다.

#### Maximum Likelihood equation

iid 가정을 통해 다음과 같이 MLE를 구할 수 있다.
$$
\nabla_{\theta} \prod_{i=1}^n p(x_i \mid \theta) = 0
$$

#### Logarithm trick

위 식을 미분해서 푸는 것은 굉장히 복잡한 문제이다. 로그변환을 통해 문제를 좀더 아래와 같은 이유로 쉽게 만들수 있다. 

The logarithm is monotonically inscreasing on $\R_+$, and the equality
$$
\ln \bigg( \prod_i f_i\bigg) = \sum_i \ln(f_i)
$$
Taking the logarithm does not change the location of a maximum or minimum:
$$
\begin{align}
& \max_y \ln g(y) \neq \max_y g(y) & \quad \text{The value changes} \\
& \arg \max_y \ln g(y) = \arg \max_y g(y) & \quad \text{The location does not change}
\end{align}
$$

#### Analytic Maximum Likelihood

$$
\begin{align}
\hat{\theta}_{\text{ML}}
& = \arg \max_\theta \prod_{i=1}^n p(x_i \mid \theta) \\
& = \arg \max_\theta \ln \bigg( \prod_{i=1}^n p(x_i \mid \theta) \bigg)  \\
& = \arg \max_\theta \sum_{i=1}^n \ln p(x_i \mid \theta)
\end{align}
$$

To then solve for $\hat{\theta}_{\text{ML}}$, find
$$
\nabla_\theta \sum_{i=1}^n \ln p(x_i \mid \theta) = \sum_{i=1}^n \nabla_\theta \ln p(x_i \mid \theta) = 0
$$
Depending on the choise of the model, we will be able to solve this

1. analytically (via a simple set of equations)
2. numerically (via an iterative algorithm using different equations)
3. approximately (typically when #2 converges to a local optimal solution)



## Multivariate Gaussian MLE

#### Block #2: Multivatiate Gaussian data model

Model: Set of all Gaussian on $\R^d$ with unknown mean $\mu \in \R^d$ and covariance $\Sigma \in \mathbb{S}_{++}^d$ (positive definite $d \times d$ matrix)

We assume that $x_1, \cdots, x_n$ are i.i.d. $p(x \mid \mu, \Sigma)$, written $x_i \stackrel{iid}{\sim} p(x \mid \mu, \Sigma) $.



#### Block #3: Maximum likelihood solution

We have to solve the equation
$$
\sum_{i=1}^n\nabla_{\mu, \Sigma} \ln p(x_i \mid \mu \Sigma) = 0
$$
for $\mu$ and $\Sigma$.

First take the gradient with respect to $\mu$.
$$
\begin{align}
\vec0
& = \nabla_\mu \sum_{i=1}^n \ln \frac{1}{\sqrt{(2\pi)^2}\left|\Sigma\right|}
\exp\bigg(-\frac{1}{2}(x_i - \mu)^T\Sigma^{-1}(x_i-\mu)\bigg) \\
& = \nabla_{\mu} \sum_{i=1}^n - \frac{1}{2} \ln(2\pi)^d\left| \Sigma \right| - \frac{1}{2}(x_i - \mu)^T \Sigma^{-1}(x_i - \mu) \\
& = - \frac{1}{2} \sum_{i=1}^n \nabla_\mu \bigg(x_i^T\Sigma^{-1}x_i - 2\mu^T\Sigma^{-1}x_i + \mu^T\Sigma^{-1}\mu \bigg) \\
& = -\Sigma^{-1}\sum_{i=1}^n(x_i - \mu)
\end{align}
$$
Sing $\Sigma$ is positive definite, the only solution is
$$
\sum_{i=1}^n(x_i - \mu) = \vec0 \quad \Longrightarrow \quad \hat{\mu}_{\text{ML}} = \frac{1}{n}\sum_{i=1}^nx_i
$$
Since this solution is independent of $\Sigma$, it doesn't depedent on $\hat{\Sigma}_{\text{ML}}$.

Now take the gradient with respect to $\Sigma$. 
$$
\begin{align}
\vec0 
& = \nabla_\Sigma \sum_{i=1}^n -\frac{1}{2}\ln(2\pi)^d \left|\Sigma \right| - \frac{1}{2}(x_i - \mu)^T\Sigma^{-1}(x_i - \mu) \\
& = -\frac{n}{2}\nabla_\Sigma \ln \left| \Sigma \right| - \frac{1}{2}\nabla_{\Sigma}\text{trace}\bigg( \Sigma^{-1}\sum_{i=1}^n(x_i-\mu)(x_i-\mu)^T \bigg) \\
& = -\frac{n}{2}\Sigma^{-1} + \frac{1}{2}\Sigma^{-2}\sum_{i=1}^n(x_i-\mu)(x_i-\mu)^T
\end{align}
$$
Solving for $\Sigma$ and plugging in $\mu = \hat{\mu}_{\text{ML}}$,
$$
\hat{\Sigma}_{\text{ML}}= \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu}_{\text{ML}})(x_i - \hat{\mu}_{\text{ML}})^T
$$

#### Summary: Gaussian MLE

So if we have data $x_1, \cdots, x_n $ in $\R^d$ that we hypothesize is i.i.d. Gaussian, the maximum likelihood values of the mean and covariance matrix are 
$$
\begin{align}
\hat{\mu}_{\text{ML}}  & = \frac{1}{n}\sum_{i=1}^nx_i \\
\hat{\Sigma}_{\text{ML}} & = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu}_{\text{ML}})(x_i - \hat{\mu}_{\text{ML}})^T
\end{align}
$$
**Are we done?** There are many assumptions/issues with this approach that makes finding the "best" parameter values not a complete victory.

- We made a **model assumption** (multivariate Gaussian)

- We made an **i.i.d. assumption**.

- We **assumed that maximizing the likelihood is the best thing **to do.

  We often user $\theta_{\text{ML}}$ to make predictions about $x_{\text{new}}$. How does $\theta_{\text{ML}}$  generalize to $x_{\text{new}}$? If $x_{1:n}$ don't **capture the space** well, $\theta_{\text{ML}}$ can **overfit** the data.



## Linear Regression

#### Regression: Problem Definition

*Data*

**Input**: $x \in \R^d$ (i.e., measurements, covariates, features, indepden. variables)

**Output**: $y \in \R$ (i.e., response, dependent variable)



*Goal*

Find a function $f: \R^d \rightarrow \R$ such that $y \approx f(x; w)$ for the data pair $(x, y)$. $f(x; w)$ is called a a **regreession function**. Its free parameter are $w$.



*Definition of linear regression*

A regression method is called linear if the prediction $f$ is a linear function of the unknown parameter $w$.



#### Least Squares Linear Regression Model

*Model*
$$
y_i \approx f(x_i; w) = w_0 + \sum_{j=1}^dx_{ij}w_j
$$
*Model learning*

We first needn an objective function to tell us what a "good" value of $w$ is.



*Least squares*

The least squares objective tells us to pick the $w$ that minimizes the sum of squared errors
$$
w_{\text{LS}} = \arg \min_w \sum_{i=1}^n (y_i -f(x_i;w))^2 \equiv \arg \min_w \mathcal{L}
$$


We believe there is a linear relationship between $x_i$ and $y_i$,
$$
y_i = w_0 + \sum_{j=1}^d x_{ij}w_j + \epsilon_i
$$
We want to minimize the objective function
$$
\mathcal{L} = \sum_{i=1}^n \epsilon_i^2 = \sum_{i=1}^n \big(y_i - w_0 -\sum_{j=1}^d x_{ij}w_j\big)^2
$$
with respect to $(w_0, \cdots, w_d)$.



#### Notation: Regression

Usually, for linear regression we include an intercept term $w_0$ that doesn't interact with any element win the vector $x \in \R^d$.

It will be convenient to attach a 1 to the first dimension of each vector $x_i$ (which we indicate by $x_i \in \R^{d+1}$) and in the first column of the matrix $X$:
$$
x_i = 
\begin{bmatrix}
1 \\ x_{i1} \\ \vdots \\ x_{id}
\end{bmatrix}, 
\quad 
\bold{X} = 
\begin{bmatrix}
1 & x_{11} & \cdots & x_{1d} \\
1 & x_{21} & \cdots & x_{2d} \\
\vdots & & \vdots \\
1 & x_{n1} & \cdots & x_{nd} \\
\end{bmatrix}
=
\begin{bmatrix}
1 - x_1^T - \\
1 - x_2^T - \\
\vdots \\
1 - x_n^T - \\
\end{bmatrix}
$$
with $ w  = [w_0, w_1, \cdots, x_d]^T$ as $w \in \R^{d+1}$.

Assumptions for now:

- All features are treated as continuous-valued $(x\in\R^d)$
- We have more obervations than dimensions $(d < n)$

#### Least Squares in Vector Form

$$
\mathcal{L} =\sum_{i=1}^n(y_i - x_i^Tw)^2
$$

*Least squares solution*
$$
\nabla_w \mathcal{L} = 0 \quad\Longrightarrow\quad \sum_{i=1}^n \nabla_w(y_i^2 - 2w^Tx_iy_i + w^Tx_ix_i^Tw) = 0
$$
Solving gives,
$$
-\sum_{i=1}^n 2y_ux_i + \big( \sum_{i=1}^n2x_ix_i^T \big)w = 0 \quad\Longrightarrow\quad w_{\text{LS}} = \bigg( \sum_{i=1}^nx_ix_i^T \bigg)^{-1}\bigg( \sum_{i=1}^ny_ix_i \bigg)
$$

#### Least Squared in Matrix Form

$$
\mathcal{L} = \sum_{i=1}^n(y_i-x_i^Tw)^2 = \left\Vert y-Xw \right\Vert^2 = (y-Xw)^T(y-Xw)
$$

Take the gradient with respect to $w$,
$$
\nabla_w\mathcal{L} = 2X^TXw - 2X^Ty = 0 \quad\Longrightarrow\quad w_{\text{LS}} = (X^TX)^{-1}X^Ty
$$
Given $x_{\text{new}}$, the least squares prediction for $y_{\text{new}}$ is
$$
y_{\text{new}} \approx x_{\text{new}}^Tw_{\text{LS}}
$$


#### Least Squares Solution

**Potential issues**

Calculating $w_{\text{LS}} = (X^TX)^{-1}X^Ty$ assumes $(X^TX)^{-1}$ exits.

When doesn't it exist?

- Answer: When $X^TX$ is not a full rank matrix.

When is $X^TX$ (square matrix) full rank?

- Answer: When the $n \times (d+1)$ matrix $X$ has at leat $d+1$ **linearly independent** rows. This means that any point in $\R^{d+1}$ can be reached by a weighted combination of $d+1$ rows of $X$.

Obviously if $n < d+1$, we can't do least squares. if $(X^TX)^{-1}$ doesn't exist, there are am infinite number of possible solutions. We want $n \gg d$ (i.e., $X$ is *tall and skinny*). But it doesn't gurantee it works. 



## Polynomial Linear Regression

#### Recall: Definition of linear regression

A regression method is called linear if the prediction $f$ is a linear function of the unknown parameters $w$.

- A function such as $y = w_0 + w_1x+w_2x^2$ is linear in $w$. 

- For a $p$th-order polynomial approximation, construct the matrix
  $$
  X = 
  \begin{bmatrix}
  1 & x_1 & x_1^2 & \cdots & x_1^p \\
  1 & x_2 & x_2^2 & \cdots & x_2^p \\
  \vdots & & & \vdots \\
  1 & x_n & x_n^2 & \cdots & x_n^p \\
  \end{bmatrix}
  $$

- Then solve exactly as before: $w_{LS} = (X^TX)^{-1}X^Ty$.

## Geometry of Least Squares Regression

The LS solutions returns $w$ sucj that $Xw$ is as close to $y$ as possible in the Euclidean sense (i.e., intuitive "direct-line" distance).
$$
\arg \min_w \left\Vert y - Xw \right\Vert^2 \quad\Longrightarrow\quad w_{\text{LS}} = (X^TX)^{-1}XTy
$$
 The columns of $X$ define a $d+1$-dimendional subspace in the higher dimensional $\R^n$.

The closest point in that subspace is the *orthonormal projection* of $y$ into the $column space$ of $X$.

Suppose $y \in \R^3$ and data $x_i \in \R$. $X_1 = [1, 1, 1]^T$ and $X_2 = [x_1, x_2, x_3]^T$

The approximation is $\hat{y} = Xw_{\text{LS}} = X(X^TX)^{-1}X^Ty$ 

![스크린샷 2019-07-29 오전 12.36.47](/Users/kakao/Documents/assets/스크린샷 2019-07-29 오전 12.36.47.png)

Know the difference between two graphs above.



























