# BAYESIAN LINEAR REGRESSION

### BAYESIAN LINEAR REGRESSION

#### Model

Have vector $$y\in \R^n$$ and covariates matrix $$X\in \R^{n\times d}$$. The $$i$$th row of $$y$$ and $$X$$ correspond to the $$i$$th observation $$(y_i, x_i)$$



In a Bayesian setting, we model this data as:
$$
\begin{align}
\bold{\text{Likelihood:}} \quad & y \sim N(Xw, \sigma^2 I) \\
\bold{\text{Prior:}} \quad & w \sim N(0, \lambda^{-1}I)
\end{align}
$$
The unknown model variable is $$w \in \R^d$$.

​	$$\blacktriangleright$$ The **likelihood model** says how well the observed data agrees with $$w$$.

​	$$\blacktriangleright$$ The **model prior**  is our prior belief (or contraints) on $$w$$.

This is called Bayesian linear regression because we have defined a prior on the unknown parameter and will try to learn its posterior.



### REVIEW: MAXIMUM  A POSTERIORI INFERENCE

#### MAP solution

Instead of finding the full posterior, we found MAP solution. MAP Inference returns the maximum of the log **joint likelihood**.
$$
\begin{align}
\quad p(w \mid y, X)
& = \quad \frac{p(w, y, X)}{p(y, X)} \\ \\
& = \quad \frac{p(y\mid w, X)p(w, X)}{p(y\mid X)p(X)} \\ \\
& = \quad \frac{p(y\mid w, X)p(w \mid X)p(X)}{p(y\mid X)p(X)} \\ \\
& = \quad \frac{p(y \mid w, X)p(w)}{p(y \mid X)}
\end{align}
$$
Using Bayes rule that this point also maximizes the *posterior* of $$w$$.

$$
\begin{align}
w_{\scriptsize\text{MAP}} 
\quad & = \quad \arg\max_w \space \ln p(w \mid y, X) \\
\quad & = \quad \arg\max_w \space \ln p(y\mid w, X)  + \ln p(w) \\
\quad & = \quad \arg\max_w \space -\frac{1}{2\sigma^2} (y - Xw)^T(y-Xw) - \frac{\lambda}{2}w^Tw + \text{const}
\end{align}
$$
We saw that this solution for $$w_{\scriptsize\text{MAP}}$$ is the same as for ridge regression:
$$
w_{\scriptsize\text{MAP}} = (\lambda \sigma^2I + X^TX)^{-1}X^Ty \iff  w_{\scriptsize\text{RR}}
$$


### POINT ESTIMATES VS BAYESIAN INFERENCE

#### Point estimates

$$w_{\scriptsize\text{RR}}$$ and $$w_{\scriptsize\text{ML}}$$ are referred to as *point estimates* of the model parameters.

Then find a specific value (point) of the vector $$w$$ that maximizes an objective function (MAP or ML)

​	$$\blacktriangleright$$ **ML**: Only consider data model: $$p(y\mid w, X)$$

​	$$\blacktriangleright$$ **MAP**: Takes into account model prior: $$p(y\mid w, X)p(w)$$

#### Bayesiam Inference

Bayesian inference goes one step further by characterizing uncertainty about the values in $$w$$ using Bayes rule. Bayesian inferene, instead of returning a specific $$w$$, is goring to return a distribution on $$w$$.



### BAYES RULE AND LINEAR REGRESSION

#### Posterior calculation

Since $$w$$ is a continuous-valued random variable in $$\R^d$$, Bayes rule says that the *posterior* distribution of $$w$$ given $$y$$, $$X$$ is
$$
p(w\mid y, X) = \frac{p(y\mid w, X)p(w)}{\int_{\R^d}p(y\mid w, X)p(w)dw}
$$
That is, we get an updated distribution on $$w$$ through the transition
$$
\text{prior} \longrightarrow \text{likelihood} \longrightarrow \text{posterior}
$$


### FULLY BAYESIAN INFERENCE

#### Bayesian linear regression

In this case, we can update the posterior distribution $$p(w \mid y, X)$$ analytically.

We work with the proportionality first:
$$
\begin{align}
p(w\mid y, X) 
\quad & \propto \quad p(y \mid w, X)\space p(w) \\ \\
\quad & \propto \quad \bigg[ \exp\bigg(-\frac{(y-Xw)^T(y-Xw)}{2\sigma^2}\bigg)\bigg]
\bigg [ \exp\bigg( -\frac{\lambda}{2} w^Tw \bigg) \bigg]
\\ \\
\quad &\propto \quad \exp\bigg( -\frac{1}{2}\big \{w^T (\lambda I + \sigma^{-2}X^TX)w - 2\sigma^{-2} w^TX^Ty \big\}\bigg)
\end{align}
$$
We defined the likelihood to be a multivariate gaussian and prior to be a zero-mean gaussian. Because we only work with proportion, we removed any premultiplied terms that does not involve $$w$$. The $$\propto$$ sign lets us multiply and divide this by anything as long as it doesn't contain $$w$$. 



### BAYESIAN INFERENCE FOR LINEAR REGRESSION

We need to normalize
$$
p(w \mid y, X) \quad \propto \quad \exp {{-\frac{1}{2} \bigg( w^T(\lambda I + \sigma^{-2}X^TX) w - 2\sigma^{-2} w^TX^Ty \bigg)}}
$$
There are two key terms in the exponent:
$$
\underbrace{w^T(\lambda I + \sigma^{-2}X^TX) w}_{\text{quadratic in} \space w}
-\underbrace{2w^TX^Ty/\sigma^2}_{\text{linear in} \space w}
$$
We can conclude that $$p(w \mid y, X)$$ is Gaussian. Why?

1. We can multiply and divide by anything not involving $$w$$.
2. A Gaussian has $$(w - \mu)^T\Sigma^{-1}(w - \mu)$$ in the exponent.
3. We can "comlete the square" by adding terms not involviign $$w$$.



**Compare**: In other words, a Gaussian looks like:
$$
p(w \mid y, \Sigma) =
\frac{1}{(2\pi)^{\frac{d}{2}} |\Sigma|^{\frac{1}{2}}} 
e^{-\frac{1}{2}(
w^T \Sigma^{-1}w-2w^T\Sigma^{-1} \mu + \mu^T \Sigma^{-1} \mu
)}
$$
and we've shown for some setting of $$Z$$ that
$$
p(x \mid y, X) = 
\frac{1}{Z}
e^{-\frac{1}{2}(w^T (\lambda I + \sigma^{-2}X^TX)w -2w^TX^Ty/\sigma^2)}
$$
**Conculde** What happens if in the above Gaussian we define:
$$
\Sigma^{-1} = (\lambda I + \sigma ^{-2} X^TX), \quad
\Sigma^{-1} \mu = X^Ty/\sigma^2 
$$
Using these specific values of of $$\mu$$ and $$\Sigma$$ we only need to set
$$
Z = (2\pi) ^{\frac{d}{2}} |\Sigma| ^{\frac{1}{2}}
e^{\frac{1}{2}\mu^T \Sigma^{-1} \mu}
$$

#### The posterior distribution

Therefore, the posterior distribution of $$w$$ is:
$$
\begin{align}
p(w \mid y, X)
& \quad =  \quad N(w \mid \mu, \Sigma) \\
\Sigma  & \quad = \quad (\lambda I + \sigma^{-2}X^TX)^{-1} \\
\mu & \quad = \quad (\lambda \sigma^2I + X^TX)^{-1}X^Ty 
\quad \Longleftarrow \quad w_{\scriptsize\text{MAP}}

\end{align}
$$
Things to notice:

$$\blacktriangleright$$ $$\mu = w_{\scriptsize\text{MAP}}$$ after a redefinition of the regularization parameter $$\lambda$$.

$$\blacktriangleright$$ $$\Sigma$$ captures uncertainty about $$w$$ as $$\text{Var}[w_{\scriptsize\text{LS}}]$$ and $$\text{Var}[w_{\scriptsize\text{RR}}]$$ did before.

$$\blacktriangleright$$ However, now we have a full probability distribution on $$w$$.



### USES OF THE POSTERIOR DISTRIBUTION

#### Understanding $$w$$

We saw how we could calculate the variance of $$w_{\scriptsize\text{LS}}$$ and $$w_{\scriptsize\text{RR}}$$. Now we have an entire distribution. Some question we can ask are:

​	**Q**: Is $$w_i > 0$$ or $$w_i < 0$$? Can we confidently say $$w_i \neq 0$$?

​	**A**: Use the *marginal posterior distribution*: $$w_i \sim N(\mu_i, \Sigma_{ii})$$.



​	**Q**: How do $$w_i$$ and $$w_j$$ relate?

​	**A**: Use their joint marginal posterior distribution:
$$

$$


 

































































