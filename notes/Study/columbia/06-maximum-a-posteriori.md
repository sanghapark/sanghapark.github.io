# Maximum a Posteriori

### LIKELIHOOD MODEL

#### Least Squares and maximum likelihood

When we modeled data pairs $$(x_i, y_i)$$ with a linear model, $$y_i \approx x_i^Tw$$, we saw that the least squares solution,
$$
w_{\scriptsize \text {LS}} = \arg \min_ w (y - Xw)^T(y-Xw)
$$
was equivalent to the maximum likelihood solution when $$y \sim  N(Xw, \sigma^2 I)$$.

**The question now is whether a similar probabilistic connection can be made for the ridge regression problem.**



#### Ridge Regression and Bayesian Modeling

The likelihood model is $$y \sim N(Xw, \sigma^2I)$$. What about a prior for $$w$$?

Let us assume that the prior for $$w$$ is Gaussian, $$w\sim N(0, \lambda^{-1}I)$$. Then
$$
p(w) = \bigg(\frac{\lambda}{2\pi}\bigg)^{\frac{d}{2}}e^{-\frac{\lambda}{2}w^Tw}
$$
We can now try to find a $$w$$ that satisfies both the data likelihood, and our prior conditions abour $$w$$.



### MAXIMUM A POSTERIORI ESTIMATION

Maximum a posteriori (MAP) estimation seeks the most probable value $$w$$ under the posterior:
$$
\begin{align}
w_{\scriptsize\text{MAP}} 
\quad & = \quad
\arg \max_{w} \ln p(w \mid y, X) \\
\quad & = \quad
\arg \max_w \ln \frac{p(y\mid w, X) \space p(w)}{p(y \mid X)} \\
\quad & = \quad
\arg \max_{w} \ln p(y \mid w, X) + \ln p(w) - \ln p(y \mid X)
\end{align}
$$

- Contrast this with ML, which only focuses on the likelihood
- The normalizing constant term $$\ln p(y \mid X)$$ doesn't involve $$w$$. Therefore, we can maximize the first two terms alone. 상수를 더해줘서 최대값/최소값의 수치는 바뀌어도 위치는 변하지 않는다.
- In many models we don't know $$ln p(y \mid X)$$, so this fact is useful.



### MAP FOR LINEAR REGRESSION

MAP using our defined prior gives:
$$
\begin{align}
w_{\scriptsize\text{MAP}} 
\quad & = \quad
\arg \max_w \ln p(y \mid w, X) + \ln p(w) \\
\quad & = \quad
\arg \max_w - \frac{1}{2\sigma^2} (y - Xw)^T (y - Xw) - \frac{\lambda}{2}w^Tw + \text{const.}
\end{align}
$$
Calling this objective $$\mathcal L$$, then as before we find $$w$$ such that
$$
\nabla_w \mathcal L = 
\frac{1}{\sigma^2} X^Ty - \frac{1}{\sigma^2}X^TXw - \lambda w = 0
$$

- The solution is $$w_{\scriptsize \text{MAP}} = (\lambda\sigma^2 I + X^TX)^{-1}X^Ty$$

- Notice that $$w_{\scriptsize\text{MAP}} = w_{\scriptsize\text{RR}}$$ (modulo a switch from $$\lambda$$ to $$\lambda \sigma^2$$)

- RR maximizes the posterior, which LS maximizes the likelihood. RR에서 정규화 텀이 $$w$$ 에 대한 사전 믿음이다.

  































