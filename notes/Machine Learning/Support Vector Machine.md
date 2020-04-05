# Support Vector Machine

## Prerequisite

### Vector

Vector는 두가지의 요소로 이루어진다.

- norm
- direction

### Norm

벡터 $\bold x​$ 는 크기를 가지고 $\Vert \bold x \Vert​$ 로 크기로 표시하고 norm한다.

### Direction

The direction of a vector $\bold x = (x_1, x_2, \cdots, x_d)$ is the vector $\bold u = (\frac{x_1}{\Vert \bold x \Vert}, \frac{x_2}{\Vert \bold x \Vert}, \cdots. \frac{x_d}{\Vert \bold x \Vert})$. 

![image-20190207104122220](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190207104122220.png)
$$
\cos(\theta) = \frac{x_1}{\Vert \bold x \Vert}
$$

### Dot Product

It is the product of the Euclidian magnitudes of the two vectors and the cosine of the angle between them.
$$
\bold x \cdot \bold y = \Vert \bold x \Vert \Vert \bold y \Vert \cos(\theta)
$$
$\theta$ 는 두 벡터 사이의 각이다. 왜 위의 공식이 성립되는지 알아보자. $\theta$를아래와 같이 $\beta$ 와 $\alpha$로 분해 할 수 있다. 

| $\theta$                                                     | $\beta$                                                      | $\alpha$                                                     | $\theta = \beta - \alpha$                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![dot product](https://i0.wp.com/www.svm-tutorial.com/wp-content/uploads/2014/11/11-dot-product-e1415553317776.png?zoom=2&resize=320%2C275&ssl=1) | ![dot product](https://i0.wp.com/www.svm-tutorial.com/wp-content/uploads/2014/11/13-dot-product-cosx-e1415553344960.png?zoom=2&resize=250%2C324&ssl=1) | ![dot product](https://i2.wp.com/www.svm-tutorial.com/wp-content/uploads/2014/11/12-dot-product-cosy.png?zoom=2&resize=409%2C191&ssl=1) | ![dot product](https://i1.wp.com/www.svm-tutorial.com/wp-content/uploads/2014/11/11-dot-product-angles.png?zoom=2&resize=350%2C273&ssl=1) |

*Difference identity for cosine* 이론에 따르면 다음과 같은 공식이 성립된다. 증명은 찾아보자.
$$
\cos(\beta - \alpha) = \cos(\beta)\cos(\alpha) + \sin(\beta)\sin(\alpha)
$$
다음 공식들을 (3) 식에 대입 해주자.
$$
\cos(\beta) = \frac{x_1}{\Vert \bold x \Vert} \\
\sin(\beta) = \frac{x_2}{\Vert \bold x \Vert} \\
\cos(\alpha) = \frac{y_1}{\Vert \bold y \Vert} \\
\sin(\alpha) = \frac{y_2}{\Vert \bold y \Vert} \\
$$
아래와 같이 정리 할 수 있다.
$$
\begin{align}
\cos(\theta) 
& = \cos(\beta - \alpha) \\
& = \cos(\beta)\cos(\alpha) + \sin(\beta)\sin(\alpha) \\
& = \frac{x_1}{\Vert \bold x \Vert}\frac{y_1}{\Vert \bold y \Vert} + \frac{x_2}{\Vert \bold x \Vert}\frac{y_2}{\Vert \bold y \Vert} \\
& = \frac{x_1y_1 + x_2y_2}{\Vert \bold x \Vert \Vert \bold y \Vert} \\
\end{align}
$$

$$
\begin{align}
\Vert \bold x \Vert \Vert \bold y \Vert \cos(\theta) 
& = x_1y_1 + x_2y_2 \\
& = \sum_{i=1}^2 x_iy_i \\
& = \bold x \cdot \bold y
\end{align}
$$

$$
\therefore \quad \bold x \cdot \bold y = \Vert \bold x \Vert \Vert \bold y \Vert \cos(\theta)
$$

### Orthogonal Projection of a Vector

Given two vectors $\bold x, \bold y$, let's find the **orthogonal projection of $\bold x$ onto $\bold y$**.

![z is the projection of x onto y](https://i1.wp.com/www.svm-tutorial.com/wp-content/uploads/2014/11/14-projection-2-e1415553135545.png?zoom=2&resize=350%2C245&ssl=1)

$\bold z$ is the projection of $\bold x$ onto $\bold y$. By definition,
$$
\cos(\theta) = \frac{\Vert \bold z \Vert}{\Vert \bold x \Vert} \\
\Vert \bold z \Vert = \Vert \bold x \Vert \cos(\theta)
$$
(12) 공식과 합하면 
$$
\frac{\bold x \cdot \bold y}{\Vert \bold x \Vert \Vert \bold y \Vert} = \frac{\Vert \bold z \Vert}{\Vert \bold x \Vert} \\
\Vert \bold z \Vert = \Vert \bold x \Vert \frac{\bold x \cdot \bold y}{\Vert \bold x \Vert \Vert \bold y \Vert} \\
\therefore \quad\Vert \bold z \Vert = \frac{\bold x \cdot \bold y}{\Vert \bold y \Vert}
$$
**이제 우리는 $\bold y​$ 를 지나는 선과 $\bold x​$ 까지의 가장 짧은 거리를 계산 할 수 있다.**

### Understanding the equation of the hyperplane

라인은 $ax + b = y$ 로 표현한다. 벡터로 표현하면 다음과 같다.
$$
y - ax - b = 0 \\
1y - ax - b1 = 0 \\
\text{Given, }\quad \bold w = 
\begin{bmatrix} 
1 \\ -a \\ -b
\end{bmatrix}, 
\bold x = 
\begin{bmatrix}
y \\ x \\ 1
\end{bmatrix}
\\
\bold w^\intercal \cdot \bold x = y -ax - b = 0
$$
벡터 표형 방식이 좋은 이유는

1. 다차원일 경우 벡터로 표현 하기 편하다.
2. 벡터 $\bold w$ 은 hyperplane과 normal(수직)이다.

### Compute the distance from a point to the hyperplane

벡터 $\bold a​$ 와 hyperplane까지의 거리를 계산 해보자. 벡터 $\bold p​$ 는 $\bold a​$ 의 $\bold w​$ 에대한 프로젝션이다.

공식 (14)를 사용 하여 풀어보자.![p is the projection of a onto w](https://i0.wp.com/www.svm-tutorial.com/wp-content/uploads/2014/11/18-svm-hyperplane.png?resize=571%2C390&ssl=1)
$$
\Vert \bold p \Vert = \frac{\bold a \cdot \bold w}{\Vert \bold w \Vert} \\
\Vert \bold p \Vert = \frac{3*2 + 4*1}{\sqrt{2^2 + 1^1}} \\
\Vert \bold p \Vert = \frac{10}{\sqrt{5}}
$$
**어떤 hyperplane으로 부터 어떤 포인트 $\bold a$ 까지의 거리를 구하고 싶으면 hyperplane과 orthogonal한 벡터  $\bold w$의 유닛벡터 $\frac{\bold w}{\Vert \bold w \Vert}$ 와 $\bold a$ 의 Dot Product 를 구하면 된다.**
$$
\text{Distance between a point $\bold a$ and a hyperplane $\bold w^\intercal\bold x$} = \frac{\bold a \cdot \bold w}{\Vert \bold w \Vert}
$$



### Compute the margin of the hyperplane

마진은 곱하기 2를 해줘야 한다.
$$
\text{margin} = 2\Vert \bold p \Vert = \frac{20}{\sqrt{5}}
$$
마진을 구하는 방법을 찾았다. **이제 우리는 마진을 최대화 할 수있는 hyperplane을 찾아보자.**



## SVM - Understanding the math - the optimal hyperplane

우리가 생각해봐야 할 것은 아래 세가지이다.

1. 어떻게 optimal hyperplane을 찾을 것인가?
2. 두개의 hyperplane의 거리는 어떻게 계산 하나?
3. SVM optimization 문제는 무엇인가?

우리가 전에 찾은 마진 $2\Vert \bold p \Vert$ 를 가진 decision boundary는 optimal hyperplane은 아니였다. Optimal hyperplane은 트레이닝 데이터의 마진을 최대화 하는 것이다. **Finding the biggest margin, is the same thing as finding the optimal hyperplane.**



어떻게 가장 큰 마진을 찾을수 있을까? 

1. 트레이닝 데이터를 모은다.

2. 두개의 hyperplane 사이에 어떠한 데이터도 없게 hyperplane 2개를 만든다.

3. hyperplane 사이의 거리가 최대가 되도록 한다.

   

자세히 들여다보자.

1. 트레이닝 데이터를 모은다.

   My dataset $D$ is the set of n couples of element $(\bold x_i, y_i)$
   $$
   D = \{ (\bold x_i, y_i) \vert \bold x_i \in \Re^d, y_i \in \{-1, 1 \} \}_{i=1}^n
   $$

2. 두개의 hyperplane 사이에 어떠한 데이터도 없게 hyperplane 2개를 만든다. 여기서 우리는 linearly separable data에 대해서만 생각한다. 

   Given a hyperplane $H_0$ separating the dataset and satisfying:
   $$
   \bold w \cdot \bold x + b = 0 \\
   $$
   We can select two other hyperplanes $H_1$ and $H_2​$ which also seperate the data and have the following equations:
   $$
   \bold w \cdot \bold x + b = \delta \\
   \bold w \cdot \bold x + b = -\delta
   $$
   $H_0$ is equidistant from $H_1$ and $H_2$. To simplify the problem, let $\delta$ be 1.
   $$
   \bold w \cdot \bold x + b = 1 \\
   \bold w \cdot \bold x + b = -1
   $$
   For each vector $\bold x_i$, either
   $$
   \bold w \cdot \bold x_i + b \geq 1 \quad \text{for $x_i$ having the class $1$} \\
   \text{or} \\
   \bold w \cdot \bold x_i + b \leq -1 \quad \text{for $x_i$ having the class -1} \\
   $$
   Let's combine the two constaints into one:
   $$
   y_i(\bold w \cdot \bold x_i + b) \geq 1 \quad \forall i
   $$
   We now have a unique constraint (8) instead of two (23).

3. hyperplane 사이의 거리가 최대가 되도록 한다.

   Before trying to maximize the distance between the two hyperplanes, how do we compute the distance between the two? 

   Let:

   - $H_{0}$ be the hyperplane having the equation $\bold w \cdot \bold x + b = -1$
   - $H_{1}$ be the hyperplane having the equation $\bold w \cdot \bold x + b = 1$
   - $\bold x_0 $ be a point in the hyperplane $H_{0}$

   | ![Figure 9: m is the distance between the two hyperplanes](https://i2.wp.com/www.svm-tutorial.com/wp-content/uploads/2015/06/svm_margin_demonstration_1.png?resize=584%2C526&ssl=1) 1 | ![Figure 10: All points on the circle are at the distance m from x0](https://i1.wp.com/www.svm-tutorial.com/wp-content/uploads/2015/06/svm_margin_demonstration_2.png?resize=584%2C428&ssl=1)2 | ![Figure 11: w is perpendicular to H1](https://i0.wp.com/www.svm-tutorial.com/wp-content/uploads/2015/06/svm_margin_demonstration_3.png?resize=584%2C521&ssl=1)3 |
   | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | ![Figure 12: u is also is perpendicular to H1](https://i0.wp.com/www.svm-tutorial.com/wp-content/uploads/2015/06/svm_margin_demonstration_4.png?resize=584%2C512&ssl=1)4 | ![Figure 13: k is a vector of length m perpendicular to H1](https://i2.wp.com/www.svm-tutorial.com/wp-content/uploads/2015/06/svm_margin_demonstration_5.png?resize=584%2C522&ssl=1)5 | ![Figure 14: z0 is a point on H1](https://i2.wp.com/www.svm-tutorial.com/wp-content/uploads/2015/06/svm_margin_demonstration_7.png?resize=584%2C446&ssl=1)6 |

   $m$ is the shortest distance from $x_0$ to $H_1$. $m$ is called **the margin**. m 계산하는 방법을 알아보자.

   Let $m​$ have a direction by  giving a unit vector $\frac{\bold w}{\Vert \bold w \Vert }​$ and call it $\bold k​$:
   $$
   \bold k = m\frac{\bold w}{\Vert \bold w \Vert}
   $$
   Let
   $$
   \bold z_0 = \bold x_0 + \bold k \quad \text{which is in the hyperplane $H_1$}
   $$
   which implies
   $$
   \bold w \cdot \bold z_0 + b = 1 \\
   \bold w \cdot (\bold x_0 + \bold k) + b = 1 \\
   \bold w \cdot (\bold x_0 + m\frac{\bold w}{\Vert \bold w \Vert}) + b = 1 \\
   \bold w \cdot \bold x_0 + m \frac{\bold w \cdot \bold w}{\Vert \bold w \Vert}) + b = 1 \\
   \bold w \cdot \bold x_0 + m \frac{\Vert \bold w \Vert^2}{\Vert \bold w \Vert}) + b = 1 \\
   \bold w \cdot \bold x_0 + m \Vert \bold w \Vert + b = 1 \\
   \bold w \cdot \bold x_0 + b = 1 - m\Vert \bold w \Vert
   $$
   As $\bold x_0$ is in $H_0$ the, $\bold w \cdot \bold x_0 + b = -1$
   $$
   -1 = 1 - m\Vert \bold w \Vert \\
   \therefore \quad m = \frac{2}{\Vert \bold w \Vert}
   $$
   We now need to maximize the margin $m$. $m$ gets bigger as $\Vert \bold w \Vert$ gets smaller. To maximize $m$, we need to minimize $\Vert \bold w \Vert$. Thus,
   $$
   \min_{\bold w, b} \quad \Vert \bold w \Vert \quad \text{s.t.} \quad y_i(\bold w \cdot \bold x_i + b) \geq 1 \quad \forall i
   $$
   Once we have solved it, we will have found the $(\bold w, b)$ for which $\Vert \bold w \Vert$ is the smallest possible and the constraints satisfied.

   

   ## Hard-margin SVM

### Solving the Optimization Problem

 [Lagrange Multiplier](../Mathematics/Lagrange Multiplier.md)를 참고하자.
### The SVM Lagrangian Problem

$$
\min_{\bold w, b} \quad \frac{1}{2}\Vert \bold w \Vert^2 \quad \text{s.t.} \quad y_i(\bold w \cdot \bold x_i + b) - 1\geq 0 \quad \forall i
$$

The factor $\frac{1}{2}$ has been added for later convenience, when we will use QP solver to solve the problem and suqaring the norm has the advantage of removing the square root. 

We now have **Convex Quadratic Optimization Problem** which, although not obvious yet, is much simpler to solve.

 We introduce the **Lagrangian function**:
$$
\begin{align}
\mathcal{L}(\bold w, b, \alpha) 
& = f(\bold w) - \sum_{i=1}^{m}\alpha_ig_i(\bold w, b) \\
& = \frac{1}{2}\Vert \bold w \Vert^2 - \sum_{i=1}^m\alpha_i\big[ y_i(\bold w\cdot \bold x_i + b)- 1 \big] 
\end{align}
$$
$m$ is the number of constraints. We also introduced **Lagrangian multiplier** $\alpha_i$ for each contraint function. 여기서 우리는 이미 $\nabla \mathcal{L}(\bold w, b, \alpha)  = 0 ​$을  풀려고 시도 해도 되지만 데이터수가 너무 작으면 오직 analytically 하게 풀수있다. 우리는 **duality principle** 로 문제를 재정의 해보자. Dual optimization은 다음 [노트](../Mathematics/Optimization/Continuous Optimization.md)를 참고 하자.
$$
\min_{\bold w, b} \max_\alpha \mathcal{L}(\bold w, b, \alpha) \quad \text{s.t.}\quad \alpha_i \geq 0, \quad i=1,..., m
$$
The method of Lagrange multipliers is used for solving problems with equality constraints, and here we are using them with inequality constraints. This is because the method still works for inequality constraints if **KKT conditions** are satisfied. 우리는 조건이 맞족된다는 조건하에 진행 할 것이다.

We are trying to solve a convex optimization problem, and **Slater's condition** holds for affine constraints. Thus, **strong duality** holds. This means that the maximum of the dual problem is equal to the minimum of the primal problem. Solving the dual is the same thing as solving the primal, except it is easier.

Solving the minimization problem involves taking the partial derivatives of $\mathcal{L}$ w.r.t. $\bold w$ and $b$.
$$
\begin{align}
\nabla_{\bold w}\mathcal{L} & = \bold w - \sum_{i=1}^m \alpha_i y_i\bold x_i = \bold 0  \tag 1\\
\frac{\partial\mathcal{L}}{\partial b} & = - \sum_{i=1}^m \alpha_iy_i = 0 \tag 2
\end{align}
$$



(1)번 식을 풀면
$$
\bold w = \sum_{i=1}^m \alpha_i y_i \bold x_i
$$
$\mathcal{L}(\bold w, b, \bold \alpha)​$ 에 대입 해서 풀어보자.
$$
\begin{align}
\mathcal{L}(\bold w, b, \alpha) = \bold{W}(\alpha, b) & = \frac{1}{2}\bigg (\sum_{i=1}^m \alpha_i y_i \bold x_i\bigg)\cdot \bigg( \sum_{j=1}^m\alpha_j y_j \bold x_j\bigg)
- \sum_{i=1}^m\alpha_i \bigg[ y_i\bigg ( (\sum_{j=1}^m\alpha_j u_j \bold x_j\cdot \bold x_i + b \bigg) - 1\bigg] \\
& = \cdots \\
& = \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha_i \alpha_j y_i y_j \bold x_i \cdot \bold x_j - b\sum_{i=1}^m\alpha_i y_i \\
\bold W(\alpha) & = \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \bold x_i \cdot \bold x_j
\end{align} \\
$$
(2) 식에 의하면 마지막 텀은 0이라서 제거 할 수 있다. 이제 문제가 dual problem이다. **Wolfe dual problem** 이라고 한다.
$$
\begin{align}
&\max_\alpha \ \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_j \alpha_j y_i y_j \bold x_i \cdot \bold x_j \\
&\text{s.t.} \quad  \alpha_i \geq 0, \quad i=1,..., m \\
& \quad\quad \sum_{i=1}^m \alpha_i y_i = 0
\end{align}
$$
Inequality contraints 때문에 솔루션은 **Karush-Kuhn-Tucker(KKT) conditions**을 만족해야 한다.

**Karush-Kuhn-Tucker(KKT) conditions**

1. Stationary condition
   $$
   \nabla_{\bold w} \mathcal{L} = \bold w - \sum_{i=1}^m\alpha_i y_i \bold x_i = \bold 0 \\
   \frac{\partial \mathcal{L}}{\partial b} = - \sum_{i=1}^m \alpha_i y_i = 0
   $$

2. 

$$
y_i (\bold w \cdot x_i + b) - 1 \geq 0 \quad \text{for all} \quad i = 1, ..., m
$$

3. Dual feasibility condition (이해 안되면 [Continuous Optimization](../Matehmatics/Optimization/Continuous Optimization.md) 참고)

   objective function과 constraint funciton의 그레디언트 방향을 갖게 하기 위해서
   $$
   \alpha_i \geq 0 \quad \text{for all} \quad i = 1,...,m
   $$

4. Complementary slackness condition

   Support vector들은 모두 양의 Lagrange multiplier $\alpha_i$ 를 가지고 있습니다. 즉 $y_i(\bold w \cdot \bold x_i + b)  - 1 = 0$ 입니다. 
   $$
   \alpha_i[y_i(\bold w \cdot \bold x_i + b) - 1] = 0 \quad \text{for all} \quad i = 1,...,m
   $$

SVM 문제를 푸는 것은 KKT 조건을 만족하는 솔루션을 찾는 것과 같다.



Wolfe dual problem을 풀면서 우리는 모든 Lagrange multiplier $\alpha_i$ 을 찾을 수 있다. 우리의 처음 목적은 $\bold w$ 와 $b$ 를 찾는 것이다. 위에서 primal function을 풀면서 $\bold w$ 는 찾았다. $b$ 를 찾아보자. primal problem의 constraints중 하나를 선택해서 풀자. 서포트 벡터는 다음을 만족한다.
$$
\begin{align}
& \quad y_i (\bold w \cdot \bold x_i + b) - 1 = 0 \\
\Longrightarrow & \quad  y_i(\bold w \cdot \bold x_i + b)  = 1 \\
\Longrightarrow & \quad y_i^2(\bold w \cdot \bold x_i + b) = y_i \\
\Longrightarrow & \quad \bold w\cdot x_i + b = y_i \\
\quad \therefore & \quad b = y_i - \bold w \cdot \bold x_i
\end{align}
$$
*Pattern Reconition and Machine Learning(Bishop, 2006)* 에 따르면 랜덤한 서포트 벡터 하나를 선택해서 $b$ 를 계산 하는 것보다는 서포트 벡터 집합 $S$ 의 평균을 구한 것이 더 stable한 솔루션을 준다고 한다.
$$
b = \frac{1}{2}\sum_{i=1}^S(y_i - \bold w \cdot x_i)
$$
다른 저자들은 (Cristianini & Shawe-Taylor, 2000) and (Ng) 다음 과 같은 식을 사용하기도 한다.
$$
b = \frac{\max_{y_i = -1}(\bold w\cdot \bold x_i) + \min_{y_i = 1}(\bold w \cdot \bold x_i)}{2}
$$
$\bold w​$ 와 $b​$ 를 찾았으면 우리는 이제 Hypothesis function을 만들수 있다. 퍼셉트론과 같은 함수를 사용한다.
$$
h(\bold x_i) = \text{sign}(\bold w \cdot x_i + b)
$$

## Soft-margin SVM

Hard-margin SVM의 문제는 데이터가 linearly seperable한 것을 요구한다. 하지만 현실 세계의 데이터는 linearly seperable 하던 안하던 노이즈가 껴있다. Hard-margin의 constraints를 아래와 같이 변경 해서 학습시킨다.
$$
y_i(\bold w \cdot x_i + b) \geq 1 \quad\Longrightarrow\quad y_i(\bold w \cdot \bold x_i + b) \geq 1 - \zeta_i
$$
predicted 값과 똑같이 분류된 포인트의 **slack variable $\zeta$** 는 0이다. 하지만 최전선을 지나가면서 0보다 커지기 시작 하면서 0커지기 시작하고 decision boundary를 넘어가면서 1보다 커지기 시작한다. 잘 못 분류된 포인트의 slack variable의 값을 커진다. 아래와 같이 penalty를 추가한 objective function을 정의 할 수 있다.
$$
\begin{align}
& \min_{\bold w, b, \zeta} \ \ \frac{1}{2}\Vert \bold w \Vert^2 + C\sum_{i=1}^m \zeta_i \\
& \text{s.t.}\quad y_i (\bold w\cdot \bold x_i + b) \geq 1 -\zeta_i \\ 
& \quad \quad \ \zeta_i \geq 0 \quad \text{and} \quad \text{for any} \quad  i = 1,..., m
\end{align}
$$

### 1-Norm Regularized Soft-margin SVM

$$
\begin{align}
& \min_{\bold w, b, \zeta} \ \ \frac{1}{2}\Vert \bold w \Vert^2 + C\sum_{i=1}^m \zeta_i\\
& \text{s.t.}\quad y_i (\bold w\cdot \bold x_i + b) \geq 1 -\zeta_i \\ 
& \quad \quad \ \zeta_i \geq 0 \quad \text{and} \quad \text{for any} \quad  i = 1,..., m
\end{align}
$$

위 문제의 솔루션은 마진을 최대화 하고 에러는 가능한 최소화 하는 hyperplane이다. $C$ 는 우리는 이 regularized 문제를 일반화할 수 있다. Hard-margin 문제로 바꾸고 싶을 때 $C$ 를 작게 만들어 $\zeta$의 영향을 줄여 만들 수 있다.  



Hard-margin 문제를 풀 때랑 같은 방식으로 아래와 같은 문제 Wolfe dual problem로 축약 할 수 있다. C조건은 **Box Constraint**로 변경되었다.
$$
\begin{align}
& \max_\alpha \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \bold x_i \cdot \bold x_j \\
& \text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \text{for any} \quad i = 1,...,m \\
&\quad \quad \ \sum_{i=1}^m \alpha_i y_i = 0
\end{align}
$$
최적화된 $C$ 를 찾는 방법은 **grid search** 와 **cross-validation** 을 통해 찾을 수 있다.

 ### 2-Norm Regularized Soft-margin SVM

$$
\begin{align}
& \min_{\bold w, b, \zeta} \ \ \frac{1}{2}\Vert \bold w \Vert^2 + C\sum_{i=1}^m \zeta_i^2\\
& \text{s.t.}\quad y_i (\bold w\cdot \bold x_i + b) \geq 1 -\zeta_i \\ 
& \quad \quad \ \zeta_i \geq 0 \quad \text{and} \quad \text{for any} \quad  i = 1,..., m
\end{align}
$$

위의 식은 box constraint 없는 Wolfe dual problem으로 풀린다.

## Kernel

[Kernels](./Kernels.md) 노트를 참고하자.

## Kernel SVM

Soft-margin dual 문제를 커널을 사용하여 정의해보자.
$$
\begin{align}
& \max_\alpha \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_j y_i y_j K(\bold x_i, \bold x_j) \\
& \text{s.t.} \quad 0\leq\alpha_i \leq C, \quad \text{for any} \quad i=1,...,m \\
& \quad \quad \ \sum_{i=1}^m\alpha_i y_i = 0
\end{align}
$$
커널을 사용하는 Hypothesis 함수는 다음과 같다.
$$
h(\bold x_i) = \text{sign}\bigg( \sum_{j=1}^S \alpha_j y_j K(\bold x_j, \bold x_i) + b \bigg)
$$
$S$ 는 서포트 벡터들 집합이다. Bishop(2006)에 따르면 다른 kernel methods와는 다르게 오직 서포트 벡터들 집합으로만 커널 함수를 계산 하기 때문에 SVM은 **sparse kernel machines**라고 불린다.






































