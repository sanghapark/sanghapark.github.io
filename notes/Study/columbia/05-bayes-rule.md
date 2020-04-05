# Bayes Rules

## Prior Information/Belief

### Motivation

We've discussed the ridge regression objective function
$$
\mathcal L = \sum_{i=1}^n(y_i - x_i^Tw)^2 + \lambda w^Tw
$$
The regularization term $$\lambda w^Tw$$ was imposed to penalize values in $$w$$ that are large. This reduced potential high-variance predictions from least squares.



In a sense, we are imposing a "prior belief" about what values of $$w$$ consider to be good.

- Question: Is there a mathematical way to formalize this?
- Answer: Using probability we can frame this via **Bayes** rule.



## Bayes Rule

Bayes rule lets us quantify what we don't know. Imagine we want to say something about the probability of $$B$$ given that $$A$$ happened.



Bayes rule says that the probability of $$B$$ after knowing $$A$$ is:
$$
\underbrace{P(B\mid A)}_{posterior} = 
\underbrace{P(A\mid B)}_{likelihood} \space\underbrace{P(B)}_{prior}/\underbrace{P(A)}_{marginal}
$$
Notice what with this perspective, these probabilities take on new meanings.

$$P(B\mid A)$$ and $$P(A\mid B)$$ are both **conditional probabilities** but they have different significance.

### Bayes Rule with Continuous Variables

Bayes rule generalizes to continuous-valued random variables as follows. However, instead of *probabilities* we work with *densities*.

- Let $$\theta$$ be a continuous-valued model parameter

- Let $$X$$ be data we possess. Then by Bayes rule,
  $$
  p(\theta \mid X)  =
  \frac{p(X\mid \theta)\space p(\theta)}{\int p(X\mid \theta) \space p(\theta)\space d\theta} = 
  \frac{p(X\mid \theta)p(\theta)}{p(X)}
  $$

In this equation,

- $$p(X\mid \theta)$$ is the likelihood, known from the model definition.
- $$p(\theta)$$ is a prior distribution that we define.
- Given these two, we can (in principle) calculate $$p(\theta \mid X)$$.



## Example: Coin Bias

We have a coin with bias $$\pi$$ towards "heads". (Encode: head = 1, tail = 0)

We flip the coin many times and get a sequence of n numbers $$(x_1, x_2, …, x_n)$$.

Assume the flips are independent, meaning
$$
p(x_1, ..., x_n \mid \pi) = \prod_{i=1}^{n} p(x_i \mid \pi)
= \prod_{i=1}^{n} \pi^x_i(1-\pi)^{1-x_i}
$$
We choose a prior for $$\pi$$ which we define to be a beta distributionm
$$
p(\pi) = Beta(\pi\mid a, b) = \frac{\Gamma (a+b)}{\Gamma(a)\Gamma(b)}\pi^{a-1}(1-\pi)^{b-1}
$$
What is the posterior distribution of $$\pi$$ given $$x_1, x_2, …, x_n$$?



From Bayes rule, 
$$
p(\pi \mid x_1, ...x_n) 
= \frac{p(x_1,...,x_n\mid \pi)\space p(\pi)}{\int_0^p(x_1,...x_n\mid \pi)\space p(\pi)\space d\pi}
$$
There is a tric that is often useful:

- The denominator only normalized the numerator, doesn't depend on $$\pi$$.

- We can write $$p(\pi \mid x) \propto p(x\mid \pi )\space p(\pi)$$

- Multiply the two and see if we recognize anything:
  $$
  \begin{align}
  p(\pi \mid x_1,...,x_n) 
  \quad \propto & \quad 
  \bigg[ \prod_{i=1}^n \pi^{x_i}\space (1-\pi)^{1-x_i} \bigg]
  \bigg[ \frac{\Gamma(a+b)}{\Gamma(a) + \Gamma(b)} \pi^{a-1}(1-\pi)^{b-1} \bigg] \\
  \quad \propto & \quad
  \pi^{\sum_{i=1}^n x_i + a- 1}(1-\pi)^{\sum_{i=1}^n(1-x_i) + b -1}
  \end{align}
  $$

We recognize this $$p(\pi \mid x_1,…,x_n) = Beta(\sum_{i=1}^nx_i + a, \sum_{i=1}^n (1-x_i) + b)$$







































