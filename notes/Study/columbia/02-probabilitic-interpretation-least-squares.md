# Probablitic View

- Least squares also has an insightful probabilitic interpretation that allows us to analyze its properties
- That is, given that we pick this model as reasonable for our problem, we can ask: What kinds of assumptions are we making? 확률적으로 해석하면서 least squares 문제에서 우리는 어떤 가정을 하고 있고 그 가정이 reasonable 한지 생각해 볼 수 있다.

### Recall: Gaussian density in $n$ dimentions

Assume a diagonal covariance matrix $\Sigma = \sigma^2I$. The density is
$$
p(y\mid \mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{\frac{n}{2}}}\exp\bigg( - \frac{1}{2\sigma^2}(y-\mu)^T(y-\mu) \bigg)
$$
What if we restrict the mean to $\mu=Xw$ and find the maximum likelihood solution for $w$?

### Maximum likelihood for Gaussian linear regression

Plug $\mu = Xw$ into the mutivariate Gaussian distribution and solve for $w$ using maximum likelihood.
$$
\begin{align}
w_{ML} & = \arg \max_{w} \ln p(y\mid \mu=Xw, \sigma^2) \\
	     & = \arg \max_{w} -\frac{1}{2\sigma^2} \Vert y - Xw \Vert^2 - \frac{n}{2}\ln(2\pi\sigma^2)
\end{align}
$$
Least squares (LS) and maximum likelihood (ML) share the same solution:
$$
\text{LS:}\quad \arg\min_w \Vert y - Xw \Vert^2 \iff \text{ML:} \quad \arg\max_w -\frac{1}{2\sigma^2}\Vert y-Xw \Vert^2
$$
Therefore, in a sense we are making an **independent Gaussian noise** assumption about the error, $\epsilon_i = y_i -x_i^Tw$ 

Other ways of saying this:

1. $y_i = x_i^Tw + \epsilon_i, \quad \epsilon \stackrel{iid}{\sim} N(0, \sigma^2), \quad \text{for} \quad i = 1,…,n$
2. $y_i \stackrel{iid}{\sim} N(x_i^Tw, \sigma^2), \quad \text{for} \quad i=1,…, n$
3. $y \sim N(Xw, \sigma^2I)$

Can we use this probabilistic line of analysis to better understand the maximum likehood (i.e., least squares) solution?

### Expected solution

**Given:** The *modeling assumption* that $y\sim N(Xw, \sigma^2I)$.

We can calculate the expectation of the ML solution under this distribution,
$$
\begin{align}
\mathbb{E}[w_{ML}] & = \mathbb{E}[(X^TX)^{-1}X^Ty] \quad{\bigg( = \int \big[(X^TX)^{-1}Xy\big] p(y\mid X, w)dy \bigg)} \\
& = (X^TX)^{-1}X^T\mathbb{E}[y] \\
& = (X^TX)^{-1}X^TXw \\
& = w
\end{align}
$$
Therefore $w_ML$ is an *unbiased* estimate of $w$, i.e., $\mathbb{E}[w_{ML}] = w$

### Review: An Equality from Probability

- Even though the "expected" maximum likelihood solution is the correct one, should we actually expect to get something near it? 가우시안 분포의 분산이 넓으면 좁은 분포보다 평균 가까운 값을 샘플링 하기 어렵다. 분산을 구해보자.

- We should also look at the covariance. Recall that if $y\sim N(\mu, \Sigma)$, then
  $$
  \text{Var}[y] = \mathbb{E}[(y-\mathbb{E})(y-\mathbb{E})^T] = \Sigma
  $$

- Plugging in $\mathbb{E}[y] = \mu$, this is equivalently written as
  $$
  \begin{align}
  \text{Var}[y] & = \mathbb{E}[(y-\mu)(y-\mu)^T] \\
  & = \mathbb{E}[yy^T - y\mu^T - \mu y^T + \mu \mu^T] \\
  & = \mathbb{E}[yy^T] - \mu\mu^T
  \end{align}
  $$

- Immediatly we also get $\mathbb{E}[yy^T] = \Sigma + \mu\mu^T$. This is an equality that we will find useful in the moment.

### Variance of the Solution

Returning to least squares linear regression, we wish to find
$$
\begin{align}
\text{Var}[w_{\tiny{\text{ML}}}] 
& = \mathbb{E}[(w_{\tiny{\text{ML}}} - \mathbb{E}[w_{\tiny{\text{ML}}}])(w_{\tiny{\text{ML}}} - \mathbb{E}[w_{\tiny{\text{ML}}}])^T] \\
& = \mathbb{E}[w_{\tiny{\text{ML}}}w_{\tiny{\text{ML}}}^T] - \mathbb{E}[w_{\tiny{\text{ML}}}]\mathbb{E}[w_{\tiny{\text{ML}}}]^T
\end{align}
$$
The sequence of equalities follows:
$$
\begin{align}
\text{Var}[w_{\tiny{\tiny{\text{ML}}}}] 
\quad = & \quad \mathbb{E}[(X^TX)^{-1}X^Tyy^TX(X^TX)^{-1}]-ww^T \\
\quad = & \quad (X^TX)^{-1}X^T\mathbb{E}[yy^T]X(X^TX)^{-1} - ww^T \\
\quad = & \quad (X^TX)^{-1}X^T(\sigma^2I + Xww^TX^T)X(X^TX)^{-1} - ww^T \\
\quad = & \quad (X^TX)^{-1}X^T\sigma^2IX(X^TX)^{-1} + (X^TX)^{-1}X^TXww^TX^TX(X^TX)^{-1} - ww^T \\
\quad = & \quad \sigma^2(X^Tx)^{-1}
\end{align}
$$

> For matrices $A$, $B$ and vector $c$, recall that $(ABc)^T = c^TB^TA^T$

- We've shown that, under the Gaussian assumption $y \sim N(Xw, \sigma^2I)$,
  $$
  \mathbb{E}[w_{\tiny{\text{ML}}}] = w, \quad 
  \text{Var}[w_{\tiny{\text{ML}}}] = \sigma^2(X^TX)^{-1}
  $$

- When there are very large values in $\sigma^2(X^TX)^{-1}$, the values of $w_{\tiny{\text{ML}}}$ are very sentitive to the measured data $y$ (more analysis later). 가중치들의 분산이 크다면 관측된 $y$ 에 따라 가중치도 엄청 크게 될수있다. 그래서 정규화를 통해 제어 할 수 있다. 

- That is bad if we want to analyze and predict using $w_{\tiny{\text{ML}}}$.

  

# Ridge Regression

#### Regularized Least Squares

- We saw how with least squares, the values in $w_{\tiny{\text{ML}}}$ may be huge.

- In general, when developing a model for data we often wish to **constrain** the model parameters in some way

- There are many models of the form
  $$
  w_{\tiny{\text{OPT}}} = \arg \min_w \Vert y - Xw \Vert ^2 + \lambda g(w)
  $$

- The added terms are 

  1. $\lambda > 0$: a regularization parameter
  2. $g(w) > 0$: a penalty function that encourages desired properties about $w$



#### Ridge Regression

Ridge regression is one $g(w)$ that addresses virance issues with $w_{\tiny\text{ML}}$.

It uses the squared penalty on the regression coefficient vector $w$,
$$
w_{\tiny{\text{ML}}} = \arg \min_w \Vert y - Xw \Vert^2 + \lambda \Vert w\Vert^2
$$
The term $g(w) = \Vert w \Vert^2$ penalizes large values in $w$.

However, there is a tradeoff between the first and second terms that is controlled by $\lambda$.

- Case $\lambda \rightarrow 0 \quad : \quad w_{\tiny\text{RR}} \rightarrow w_{\tiny\text{LS}}$ 

- Case $\lambda \rightarrow \infin \quad : \quad w_{\tiny\text{RR}} \rightarrow \vec{0}$



#### Ridge Regression Solution

**Objective:** We can solve the ridge regression problem using exactly the same procedure as for least squares,
$$
\begin{align}
\mathcal{L} 
& = \Vert y - Xw \Vert^2 + \lambda \Vert w\Vert^2 \\
& = (y - Xw)^T(y - Xw) + \lambda w^Tw
\end{align}
$$
**Solution:** First, take the gradient of $\mathcal{L}$ with respect to $w$ and set to zero,
$$
\nabla_{w}\mathcal L = -2X^Ty + 2X^TXw + 2\lambda w =0
$$
Then, solve for $w$ to find that
$$
w_{\tiny\text RR} = (\lambda I + X^TX)^{-1}X^Ty
$$


#### Ridge Regression Geometry



![스크린샷 2019-08-25 오후 4.30.57](/Users/kakao/Documents/notes/Study/columbia/assets/스크린샷 2019-08-25 오후 4.30.57.png)

There is a tradeoff between squared error and penalty on $w$

We can write both in terms of *level sets*: Curves where function evaluation gives the same number.

The sum of these gives a new set of levels with a unique minimum.

You can check that we cah write:
$$
\Vert y - Xw\Vert^2 + \lambda \Vert w\Vert^2 = (w-w_{\tiny\text{LS}})^T(X^TX)(w - w_{\tiny\text{LS}}) + \lambda w^Tw + \text{(const. w.r.t. $w$)}
$$

> **Data Prepprocessing**
>
> Ridge regression penalizes each dimension of $w$ equally. 각 피쳐 데이터의 스케일이 다르다면 스케일이 작은 피쳐에 더 많은 penalty가 적용 될 것이다.
>
> Ridge regression is one possible regularization scheme. For this problem, we first assume the following *preprocessing* steps are done:
>
> 1. The mean is substracted off of y:
>    $$
>    y \leftarrow y - \frac{1}{n}\sum_{i=1}^n y_i
>    $$
>
> 2. The dimensions of $x_i$ have been standardized before constructing $X$:
>    $$
>    x_{ij} \leftarrow (x_{ij} - \bar{x}_{.j})/\hat{\sigma}_j, \quad
>    \hat{\sigma}_j = \sqrt{\frac{1}{n}\sum_{i=1}^n(x_{ij}-\bar{x}_{.j})^2}
>    $$
>    i.e., substract the empirical mean and divide by the empirical standard deviation for each dimension
>
> 3. We can show that there is no need for the dimension of 1's in this case.



#### Ridge Regression vs. Least Squares

The solutions to least squares and ridge regression are clearly very similar,
$$
\begin{align}
w_{\tiny\text{LS}} & = (X^TX)^{-1}X^Ty \\
w_{\tiny\text{RR}} & = (\lambda I + X^TX)^{-1}X^Ty
\end{align}
$$

- We can use linear algebra and probability to compare the two.
- This requires the **singular value decomposition**, which we review next.

>**Singular Value Decomposition**
>
>- We can write any $n \times d$ matrix $X$ (assume $n > d$) as $X = USV^T$, where 
>
>  1. $U$: $n \times d$ and orthogonal in the columns, i.e. $U^TU = I$
>
>     즉, 각 칼럼은 다른 칼럼과 orthogonal하고 칼럼의 unit length는 1이다.
>
>  2. $S$: $d \times d$ non-negative diagonal matrix, i.e., $S_{ii} \geq 0$ and $S_{ij} = 0$ for $i \neq j$
>
>  3. $V$: $d \times d$ and orthogonal, i.e., $V^TV = VV^T = I$ 
>
>- From this we have the immediate equalities
>  $$
>  X^TX = (USV)^T(USV^T) = VS^2V^T, \quad XX^T = US^2U^T
>  $$
>
>- Assuming $S_{ii} \neq 0$ for all $i$ (i.e., "$X$ is full rank"), we also have that
>  $$
>  (X^TX)^{-1} = (VS^2V^T)^{-1} = VS^{-2}V^T
>  $$
>  Proof: Plug in and see that it satifies definition of inverse
>  $$
>  (X^TX)(X^TX)^{-1} = VS^2V^TVS^{-2}V^T = I
>  $$



#### Least Squares and the SVD

Using the SVD we can rewrite the variance,
$$
\text{Var}[w_{\tiny\text{LS}}] = \sigma^2(X^TX)^{-1} = \sigma^2VS^{-2}V^T
$$
This inverse becomes huge when $S_{ii}$ is very small for some values of $i$. (Aside: This happens when columns of $X$ are highly correlated.)



The least squares prediction for new data is
$$
y_{\small\text{new}} = x_{\small\text{new}}^Tw_{\small\text{LS}}
= x_{\small\text{new}}^T(X^TX)^{-1}X^Ty = x_{\small\text{new}}^TVS^{-1}U^Ty
$$
When $S^{-1}$ have very large values, this can lead to unstable predictions.



#### Ridge Regression vs. Least Squares I

**Mathematical relationship to least squares solutin**

Recall for two symmetric matrices,  $(AB)^{-1} = B^{-1}A^{-1}$.
$$
\begin{align}
w_{\scriptsize\text{RR}} 
& = (\lambda I + X^TX)^{-1}X^Ty \\
& = (\lambda I + X^TX)^{-1}(X^TX) \underbrace{(X^T)^{-1}X^Ty}_{w_{\scriptsize\text{LS}}} \\
& = [(X^TX)(\lambda(X^TX)^{-1} + I)]^{-1}(X^TX)w_{\scriptsize\text{LS}} \\
& = (\lambda(X^TX)^{-1}+I)^{-1}(X^TX)^{-1}(X^TX)w_{\scriptsize\text{LS}} \\
& = (\lambda (X^TX)^{-1}+I)^{-1}w_{\scriptsize\text{LS}}
\end{align}
$$
Can use this to prove that the solution shrinks toward zero: $\Vert w_{\scriptsize\text{RR}}\Vert_2 \leq \Vert w_{\scriptsize\text{LS}} \Vert_2$.



#### Ridge Regression vs. Least Squares II

Continue analysis with the SVD: $X = USV^T \rightarrow (X^TX)^{-1} = VS^{-2}V^T:$
$$
\begin{align}
w_{\scriptsize\text{RR}} 
\quad = & \quad  (\lambda(X^TX)^{-1} + I)^{-1}w_{\scriptsize\text{LS}} \\
\quad = & \quad  (\lambda VS^{-2}V^T + I)^{-1}w_{\scriptsize\text{LS}} \\
\quad = & \quad  V(\lambda S^{-2} + I)^{-1}V^Tw_{\scriptsize\text{LS}} \\
\quad := & \quad VMV^Tw_{\scriptsize\text{LS}}
\end{align}
$$
$M$ is a diagonal matrix with $M_{ii} = \frac{S_{ii}^2}{\lambda + S_{ii}^2}$. We can pursue this to show that 
$$
\begin{align}
w_{\scriptsize\text{RR}} \quad = & \quad VS_{\lambda}^{-1}U^Ty \\
S_{\lambda}^{-1} \quad = & \quad 
\begin{bmatrix}
\frac{S_{11}}{\lambda + S_{11}^2} & & 0 \\
& \ddots & \\
0 & & \frac{S_{dd}}{\lambda + S_{dd}^2}
\end{bmatrix}
\end{align}
$$
Compare with $w_{\scriptsize\text{LS}} = VS^{-1}U^Ty$, which is the the case where $\lambda = 0$ above. $\lambda$ 가 0이고 $S_{ii}$ 가 0에 가까워 지면 분모 제곱항때문에 무한대에 가까워진다. 하지만 $\lambda$ 가 0보다 크면 $S_{ii}$ 가 0에 가까워져도 해당 항은 0이 된다. $\lambda$ 가 분모가 가질수 있는 값에 바닥을 만들어서 더 작아지게 만들지 않기 때문이다.



#### Ridge Regression vs. Least Squares III

Ridge regression can also be seen as a special case of least squares.

Define $\hat{y} \approx \hat{X}w$ in the following way,
$$
\begin{bmatrix}
y \\
0 \\
\vdots \\
0
\end{bmatrix}
= 
\begin{bmatrix}
- & X & - \\
\sqrt{\lambda} & & 0 \\
& \ddots & \\
0 & & \sqrt{\lambda}
\end{bmatrix}
\begin{bmatrix}
w_1 \\ \vdots \\ w_d
\end{bmatrix}
$$
If we solved $w_{\scriptsize\text{LS}}$ for this regression problem, we find $w_{\scriptsize\text{RR}}$ of the original problem: 
$$
\begin{align}
(\hat y - \hat X w)^T(\hat y = \hat Xw)
\quad = & \quad (y - Xw)^T(y-Xw) + (\sqrt\lambda w)^T(\sqrt \lambda w) \\
\quad = & \quad \Vert y - Xw \Vert^2 + \lambda \Vert w \Vert^2
\end{align}
$$
Ridge regression is almost like **augmented** least square problem.



#### Selecting $\lambda$

It will show a way to understand how $\lambda$ changes our regularized least squares solutions.

![3tLxq](/Users/kakao/Documents/notes/Study/columbia/assets/3tLxq.jpg)

Degrees of freedom:
$$
\begin{align}
df(\lambda) 
& = \text{trace} \big[ X(X^TX + \lambda I)^{-1}X^T \big] \\
& = \sum_{i=1}^d \frac{S_{ii}^w}{\lambda + S_{ii}^2}
\end{align}
$$
As $\lambda$ gets bigger, degree of freedom of becomes smaller. We eill discuss methods for picking $\lambda$ later. 



### Review: Regression with/without regularization

**Given:**

A data set $(x_1, y_1), …, (x_n, y_n)$, where $x \in \mathbb R^d$ and $y\in \mathbb R$. We standardize such that each dimension of $x$ is zero mean unit variance, and $y$ is zero mean.

**Model**:

We define a model of the form
$$
y \approx f(x;w)
$$
We particulary focus on the case where $f(x;w) = x^Tw$.

**Learning:**

We can learn the model by minimizing the objective (aka, "loss") function
$$
\mathcal L = \sum_{i=1}^n (y_i - x_i^Tw)^2 + \lambda w^Tw 
\iff \mathcal L = \Vert y 0 Xw \Vert^2 + \lambda \Vert w \Vert^2
$$
We've focused on $\lambda = 0$ (least squares) and $\lambda > 0$ (ridge regression).









