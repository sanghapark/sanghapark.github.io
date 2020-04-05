# Bias-Variance Tradeoff

We will discuss the bias-variance tradeoff in the context of comparing least squares and ridge regression.

### Bias-Variance for Linear Regression

We can go further and hypothesize a generative model $y \sim N(Xw, \sigma^2I)$ and some true (but unknown) underlying value for the parameter vector $w$.

- We saw how the least squares solution, $w_{\scriptsize\text{LS}} = (X^TX)^{-1}X^Ty$, is unbiased but potentially has high variance:
  $$
  \mathbb E [w_{\scriptsize\text{LS}}] = w, \quad \text{Var}[w_{\scriptsize\text{LS}}] = \sigma^2(X^TX)^{-1}
  $$

- By contrast, the ridge regression solution is $w_{\scriptsize\text{LS}} = (\lambda I + X^TX)^{-1}X^Ty$. Using the same procedure as for least squares, we can show that
  $$
  \mathbb E[w_{\scriptsize\text{RR}}] 
  = (\lambda I + X^TX)^{-1}X^TXw, \quad
  \text{Var}[w_{\scriptsize\text{RR}}] = \sigma^2Z(X^TX)^{-1}Z^T,
  $$
  where $Z = (I + \lambda (X^TX)^{-1})^{-1}$.

 

The expectation and covariance of $w_{\scriptsize\text {LS}}$ and $w_{\scriptsize\text{RR}}$ gives insight into how well we can hope to learn $w$ in the case where our model assumption is correct.

- Least squares solution: unbiased, but potentially high variance
- Ridge regression solution: biased, but lower variance than LS

So which is preferable?

Ultimately, we really care about how well our solution for $w$ generalizes to new data. Let $(x_0, y_0)$ be future data for which we have $$x_0$$, but not $$y_0$$.

- Least squares predicts $$y_0 = x_0^Tw_{\scriptsize\text{LS}}$$
- Ridge regression predicts $$y_0 = x_o^Tw_{\scriptsize\text{RR}}$$ 

  

In keeping with the squared error measure of performance, we could **calculate the expected squared error of our prediction**:
$$
\mathbb E \big[ (y_0 - x_0^T\hat w)^2 \mid X, x_0 \big] =
\int_{\mathbb R}\int_{\mathbb R^n} 
(y_0 - x_0^T\hat w)^2p(y \mid X, w)p(y_0 \mid x_o, w) dydy_0
$$

- The estimate $\hat w$ is either $$w_{\scriptsize\text{LS}}$$ or $$w_{\scriptsize\text{RR}}$$

- The distributions on $$y, y_0$$ are Gaussian with the true (but unknown) $$w$$.
- We condition on knowing $$x_0, x_1, …, x_n$$.



In words this is saying:

- Image I know $$X, x_0$$ and assume some true underlying $$w$$.
- I generate $$y \sim N(Xw, \sigma^2 I)$$ and approximate $$w$$ with $$\hat w = w_{\scriptsize\text{LS}}$$ or $$ w_{\scriptsize\text{RR}}$$

- I then predict $$y_0 \sim N(x_0^Tw, \sigma^2)$$ using $$y_0 \approx x_0^T \hat w$$

What is the expected squared error of my prediction?



We can calculate this as follows (assume conditions on $$x_0$$ and $$X$$),
$$
\mathbb E \big[ (y_0 - x_0^T\hat w)^2 \big] =
\mathbb E[y_0^2] - 2\mathbb E[y_0]x_0^T\mathbb E[\hat w] + x_0^T \mathbb E[\hat w \hat w^T]x_0
$$

- Since $$y_0$$ and $$\hat w$$ are independent, $$ \mathbb E[y_0\hat w] = \mathbb E[y_0]\mathbb E[\hat w] $$

- Remember:
  $$
  \begin{align}
  \mathbb E[\hat w \hat w^T \quad & = \quad \text{Var}[\hat w] + \mathbb E[\hat w]\mathbb E[\hat w]^T \\
  \mathbb E[y_0^2] \quad & = \quad \sigma^2 + (x_0^Tw)^2
  \end{align}
  $$

Plugging these values in:
$$
\begin{align}
\mathbb E[(y_0 - x_0^T\hat w)^2]
\quad & = \quad
\sigma^2 + (x_0^Tw)^2 - 2(x_0^Tw)(x_0^T\mathbb E[\hat w]) + (x_0^T\mathbb E[\hat w])^2 + x_0^T\text{Var}[\hat w]x_0 \\
\quad & = \quad
\sigma^2 + x_0^T(w- \mathbb E[\hat w])(w - \mathbb E[\hat w])^Tx_0 + x_0^T\text{Var}[\hat w]x_0
\end{align}
$$


We have shown that if 

1.  $$y \sim N(Xw, \sigma^2I) $$ and $$y_0 \sim N(x_0^Tw, \sigma^2)$$, and 

2. we approximate $$w$$ with $$\hat w$$ according to some algorithm (LS or RR), then
   $$
   \mathbb E[(y_0 - x_0^T\hat w)^2 \mid X, x_0] = 
   \underbrace{\sigma^2}_{\text{noise}} +
   \underbrace{x_0^T(w-\mathbb E[\hat w])(w - \mathbb E[\hat w])^Tx_0}_{\text{squared bias}} +
   \underbrace{x_0^T\text{Var}[\hat w]x_0}_{\text{variance}}
   $$

Least squares solution의 경우 unbiased 이기 때문에 squared bias가 0이 된다. Ridge regression의 경우 biased 이기 때문에 squared bias 항은 0이 아니다. 하지만 variance는 LS solution 보다 작다.

We see that the **generalization error** is a combination of three factors:

1. Measurement noise - we can't control this given the model
2. Model bias - how close to the solution we expect to be on average.
3. Model variance - how sensitive our solution is to the data

We saw how we can find $$\mathbb E[\hat w]$$ and $$\text{Var}[\hat w]$$ for the LS and RR solutions.





























