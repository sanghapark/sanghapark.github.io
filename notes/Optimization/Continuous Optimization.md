# Continuous Optimization I: unconstrained

## Contents

- Foundations
- Categorization
- Unconstrained Optimization
  - Regression
  - Universally applied methods
  - Line search
  - Steepest descent method
  - Conjugate gradient method
  - Newton's method (and Quasi-Newton method)
  - Newton's method for nonlinear least square problems
  - Lavenberg-Marquardt algorithm

## Foundation

- Infimum and Supremum
  - Consider $\theta(u) = u - \sqrt{1 + u^2}$
  - As $u \rightarrow \infty, \quad \theta(u) \rightarrow 0$
  - Hence, $\sup {\theta(u): u \geq 0} = 0$
  - But a maximizing solution $u^*$ does not exist

- Differentiability

- Taylor Series

- Positive Semidefinite

  - $ax^2 + bx + c = 0$ 에서 $b$ 와 $c$ 는 parabola의 위치에 영향을 주고 $a$ 는 얼마나 벌어져있고 위로 벌어져있는지 아래로 벌어져있는지를 결정합니다. 고차원에서는 $a$ 는 행렬 이다. 

  - 행렬은 여러 개의 숫자로 이루어져 있으므로 실수와 같이 부호 혹은 크기를 정의하기 어렵가. 하지만 부호/크기와 유사한 개념들을 행렬에서도 정의 할 수 있다.

  - 영 벡터가 아닌 모든 벡터 $\bold x$에 대해 다음 부등식이 성립하면 행렬 $A$가 **양의 정부호(positive definite)**라고 한다.
    $$
    \bold x^\intercal \bold A \bold x \gt 0
    $$
    등호를 포함한다면 **양의 준정부호(positive semi-definite)**라고 한다.
    $$
    \bold x^\intercal \bold A \bold x \geq 0
    $$

  - $ax^2$ 이 0보다 크면 a가 0보다 크다고 말 할 수 있듯이  $\bold x^\intercal \bold A \bold x$ 는 실수가 나오기 때문에 $\bold A$에 대한 부호와 같은 유사한 성질을 알 수 있다. 0보다 크면 위로 팔을 벌리고 있는 parabola라고 생각 할 수 있다.

## Categorization

- Categorization of continuous optimization
  - Unconstrained optimization
    - ~~Linear~~
    - Nonlinear
  - Constrained Optimization
    - Linear programming
    - Nonlinear programming

- Unconstrained optimization
  $$
  \begin{align}
  & \bold x^* = \min f(\bold x) \\
  & \text{subject to} \quad \bold x \in \Omega \\
  & \text{where} \quad \Omega = \mathcal{R}^n
  \end{align}
  $$

- Constrained optimization

  - Feasible set is in the form of functional constraints.

  $$
  \Omega = \{ \bold x: h(\bold x) = 0, \ g(\bold x) \leq 0 \}
  $$

  

## Unconstrained Optimization I

일차 미분을 기반한 최적화 방식을 알아보자.

- Regression
- Universally applied methods
- Line search
- Steepest descent method
- Conjugate gradient method

### Regression: Least squared method

$$
\min \Vert Ax - y \Vert^2
$$

아래와 같은 quadratic form이다.
$$
f(x) = \frac{1}{2} x^\intercal Q x - c^\intercal x + d
$$
일차 미분하고 0으로 세팅하고 $x$에 대해서 풀면 함수를 최적화하는 $x$ 를 찾을 수 있다.
$$
\frac{\partial}{\partial x} f(x) = 0
$$

### Universally applied methods

- Iterative multidimensional optimization

  - Consists of two steps
    1. 어느 방향으로 갈지 정한다.
    2. 1번에서 결정한 방향으로 얼만큼 이동 할지 정한다.

  $$
  x_{k+1} = x_k + \alpha_kd_k,\quad \alpha_k \ \text{is a step size}
  $$

- Line Search
  - 방향을 정했으니 이제는 선따라 이동하면서 optimal point를 찾아 나선다.
  - 찾는 방식에는 
    - Golden section search
    - Fibonacci search
    - Newton's method

### Steepest descent method

Conduct two steps starting at $x_0$

1. Find search direction:

$$
-\nabla f(x_0)
$$

2. Line search $\alpha_k = \underset{\alpha \geq 0}{\arg\max} f(x_t - \alpha \nabla f(x_t))$
   $$
   x_{t+1} = x_t + \alpha_td_t
   $$

Steepest descent method의 문제점은 항상 내가 전에 이동한 방향으로 항상 움직이게 되는 것이 아니다. 컨투어가 찌그러져 있으면 지그재그로 이동한다. 비효율 적일 수 있다. 이걸 해결 하려고 나온 것이 Conjugate Descent Method이다

### Conjugate descent method

컨투어가 정확한 원형이면 최소점을 향해 이동한다. 하지만 찌그러진 원형 이라면 지그재그로 이동하게 된다. 선형 변환을 통해 좀 바꿔주는 방식이다. 다른 방식으로는 피쳐들을 nomarlization을 통해 원형으로 만들어 줄 수 있다.



## Unconstrained Optimization II

2차 미분을 기반한 최적화 방식을 알아보자. Gradient descent는 지그재그로 움직여서 효과적이지 못해서 더 좋은 방안을 찾다가 나온 2차미분을 활용한 최적화 방법론이다. 

- **Newton's method**

  - Quadratic approximation using the Talyor series expansion of $f(x)$ at the current point $x_t$

  - 테일러 급수를 사용하여 함수를 2차 함수 형태로 $x_t$ 위치에서 근사 하는 방법이다.  현재 위치 $x_t$ 에서 테일러 급수로 근사된 2차 함수를 미분하고 minimum에 해당하는 $x_{t+1}$로 이동한다. 항상 주어진 상황에서 최저점을 향해 이동 하기 때문에 수렴 속도가 빠르다.

    

    > **Taylor Series Expansion** 
    >
    > 함수 $f(x)$ 를 $x=a$ 일 때 테일러 급수를 이용하여 근사 할 수 있다.
    > $$
    > \begin{align}
    > f(x) = f_{taylor}(x) 
    > & = f(a) + \frac{f^{'}(a)}{1!}(x-a) + \frac{f^{''}(a)}{2!}(x-a)^2 + \frac{f^{'''}(a)}{3!}(x-a)^3 + \cdots  \\
    > & = \sum_{n=0}^\infty\frac{f^{(n)}(a)}{n!}(x-a)^n 
    > \end{align}
    > $$

    

    $\bold x=\bold x_t$ 일 때 함수 $f(\bold x) $를 테일러 급수 2차 함수까지만 사용해서 근사한다.
    $$
    f(\bold x) = f_{taylor}(\bold x) \approx q(\bold x) = f(\bold x_t) + (\bold x- \bold x_t)^\intercal\nabla f(\bold x_t) + \frac{1}{2}(\bold x- \bold x_t)^\intercal \bold H(x_t)(\bold x-\bold x_t)
    $$
    $q(\bold x)$ 를 최소화/최대화하는 minimizer/maximizer를 찾는다.
    $$
    \nabla q(\bold x) = \nabla f(\bold x_t) + \bold H(\bold x_t)(\bold x-\bold x_t) = 0 \\
    \bold x^* = \bold x_t - \bold H(\bold x_t)^{-1}\nabla f(x_t)
    $$
    만약 $\bold H(\bold x_t)$ 가 0보다 크면, $q(\bold x)$ 는 $\bold x^*$가 minimizer 이다. 즉, 다음 이동할 위치는 $\bold x_{t+1} = \bold x^*$ 이다.
    $$
    \bold x_{t+1} = \bold x_t - \bold H(\bold x_t)^{-1}{\nabla f(\bold x_t)}
    $$

  - 근사된 함수에서는 하강하게 minimizer를 찾았지만 실제 함수는 그부분에서 상승 하고 있을 수 있다. 그러면 새로운 시작점은 전 시점 보다 더 높은 곳에서 시작된다. 그래서 아래와 같이 $\alpha$ 변수를 추가해서 상승은 안되도록 막아줄수 있다. $\bold H(\bold x_t) \gt 0$ 과 $\nabla f(\bold x_t) \neq \bold 0$ 만족 된다면 $f(\bold x_{t+1}) < f(\bold x_t)$이 성립된다. (헤시안인 플러스 라는 것은 컨벤스 한 것이고 함수 $f(\bold x)$가 2차 미분이 가능해야함)

  $$
  \begin{align}
  & x_{\bold t+1} = \bold x_t - \alpha_t \bold H(\bold x_t)^{-1}{\nabla f(x_t)} \\
  & \text{where} \quad \alpha_t =  \underset{\alpha \geq 0}{\arg\min} \ \ f\bigg(\bold x_t - \alpha \bold H(\bold x_t)^{-1}\nabla f(\bold x_t)\bigg) 
  \end{align}
  $$

- **Quasi-Newton method**

  - 단점은 차원이 높아질 수록 Hessian 행렬($d\times d$)과 역행렬을 구하기 위한 연산량이 기하급수적으로 늘어난다. 이를 보완하기 위해 Quasi-Newton method 는 Hessian 역행렬을 근사적으로 푼다. 하지만 매번 Hessian 역행렬을 업데이트 하기 위해서 기존 Hessian을 저장 하고 있어 메모리 사용량이 크다.

- **Lavenberg-Marquardt(LM) modification**

  - 최소화 문제를 풀 때 현재 위치가 concave 스러운 곳에 있다면 Hessian 행렬이 positive definite 하지 못하게 계산 된다. 즉, 근사한 2차 함수가 concave 할 것이고 최소점이 아닌 최대점으로 이동 할 것이다.  이것을 보완하기 위해 Lavenberg-Marquardt(LM) modification을 적용 할 수 있다.
    $$
    \bold x_{x+1} = \bold x_t - (\bold H(\bold x_t) + \mu_t \bold I)^{-1}\nabla f(\bold x_t), \quad \mu_t \gt 0
    $$
    행렬이 positive definite한지 알수 있는 방법은 양쪽에 어떠한 벡터를 곱해도 결과값이 양수이면 positive definite하다고 할 수 있다. 또한 행렬의 모든 eigenvalue가 0보다 크면 positive definite 하다고 할 수 있다.  그래서 $\bold H(\bold x_t) $ 의 모든 eigenvalue들을 **강제적**으로 크게 만들어 주기 위해서 $\mu_k \bold I$를 더해 주는 것이다.

- **Final Newton's method**
  $$
  \bold x_{t+1} = \bold x_t - \alpha_t (\bold H(\bold x_t) + \mu_t\bold I)^{-1}\nabla f(\bold x_t) \quad \text{with the step size $\alpha_t$}
  $$

- **Newton's method for nonlinear least square problems**

  > **Non-linearity**
  >
  > 함수 $f(\bold x)$ 의 어떤 parameter $x_i$에 대한 편미분 $\frac{\partial f}{\partial x_i}$ 이 $x_1$ 의 함수이면 함수 $f(\bold x)$ 는 비선형이다.

  Nonlinear least square 문제는 비선형 함수의 minimizer $\bold x^*$ 를 찾는 것이다. 아래와 같은 non-linear least squares 문제를 생각해 볼 수 있다.
  $$
  \begin{align}
  & \min_{\bold x} \frac{1}{2}\Vert \bold r(\bold x) \Vert^2 = \min_{\bold x} \frac{1}{2}\sum_{i=1}^{m}r_i(\bold x)^2 \\
  & \text{where $\bold x \in \mathcal{R^n}$} \quad
  \text{and}\quad
  \bold r(\bold x) = 
  \begin{bmatrix}
  r_1(\bold x) \\
  r_2(\bold x) \\
  \vdots \\
  r_m(\bold x)
  \end{bmatrix}
  \in \mathcal{R^m}
  \end{align}
  $$
  잔차제곱합의 벡터 표현식의 1차 미분과 2차 미분을 구하자. 테일러 급수로 근사 할려면 필요하다. 먼저 1차 미분을 하자.
  $$
  \nabla \frac{1}{2}\bold r(\bold x)^\intercal \bold r(\bold x) = \bold r(\bold x)\nabla \bold r(\bold x)
  $$
  $\nabla \bold r (\bold x)$ 는 **Jacobian Matrix** 를 사용해서 표현 할 수 있다.

  > **Jacobian Matrix**
  >
  > 함수의 출력변수와 입력변수가 모두 벡터(다차원) 데이터인 경우에는 입력 변수 각각과 출력변수 각각의 조합에 대해 모두 미분이 존재한다. 도함수는 행렬 형태가 된다. 이를 **야코비안 행렬** 이라고 한다.
  > $$
  > \bold {J} \ \bold f(\bold x) = \bold J = \big( \frac{\partial \bold f}{\partial \bold x} \big) = 
  > \begin{bmatrix}
  > \nabla_{\bold x}f_1^\intercal \\ \vdots \\ \nabla_{\bold x}f_M^\intercal
  > \end{bmatrix} =
  > \begin{bmatrix}
  > \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_N} \\
  > \vdots & \ddots & \vdots \\
  > \frac{\partial f_M}{\partial x_1} & \cdots & \frac{\partial f_M}{\partial x_N} \\
  > \end{bmatrix}
  > $$

  $$
  \begin{align}
  \nabla \frac{1}{2}\bold r(\bold x)^\intercal \bold r(\bold x) 
  & = \bold r(\bold x)^\intercal\nabla \bold r(\bold x) \\
  & = \bold r (\bold x)^\intercal
  \begin{bmatrix}
  \nabla r_1(\bold x) \\
  \vdots \\
  \nabla r_n(\bold x)
  \end{bmatrix} \\
  & = \bold r (\bold x)^\intercal
  \begin{bmatrix}
  \frac{\partial r_1}{\partial x_1} & \cdots & \frac{\partial r_1}{\partial x_n} \\
  \vdots & \ddots & \vdots \\
  \frac{\partial r_m}{\partial x_1} & \cdots & \frac{\partial r_m}{\partial x_n}
  \end{bmatrix} \\
  & = \bold J(\bold x)^\intercal \bold r(\bold x)
  \end{align}
  $$

  2차 미분을 하자.
  $$
  \begin{align}
  \nabla^2 \bigg \{\frac{1}{2} \bold r(\bold x)^\intercal \bold r(\bold x) \bigg \} 
  & =  \nabla \big \{ \bold r(\bold x)^\intercal\bold \nabla \bold r(\bold x) \} \\
  & = \nabla \bold r(\bold x)^\intercal\nabla \bold r(\bold x) + \bold r(\bold x)^\intercal\nabla^2\bold r(\bold x) \\
  & \approx \nabla\bold r(\bold x)^\intercal\nabla \bold r(\bold x) \\
  & = \bold J(\bold x)^\intercal \bold J(\bold x) \\
  & = 
  \begin{bmatrix}
  \frac{\partial r_1}{\partial x_1} & \cdots & \frac{\partial r_m}{\partial x_1} \\
  \vdots & \ddots & \vdots \\
  \frac{\partial r_1}{\partial x_n} & \cdots & \frac{\partial r_m}{\partial x_n}
  \end{bmatrix}
  \begin{bmatrix}
  \frac{\partial r_1}{\partial x_1} & \cdots & \frac{\partial r_1}{\partial x_n} \\
  \vdots & \ddots & \vdots \\
  \frac{\partial r_m}{\partial x_1} & \cdots & \frac{\partial r_m}{\partial x_n}
  \end{bmatrix} \\ 
  & = 
  \begin{bmatrix}
  \frac{\partial r_1}{\partial x_1}\frac{\partial r_1}{\partial x_1} + \cdots + \frac{\partial r_m}{\partial x_1}\frac{\partial r_m}{\partial x_1} & \cdots 
  & \frac{\partial r_1}{\partial x_1}\frac{\partial r_1}{\partial x_n} + \cdots + \frac{\partial r_m}{\partial x_1}\frac{\partial r_m}{\partial x_n} \\
  \vdots & \ddots & \vdots \\
  \frac{\partial r_1}{\partial x_n}\frac{\partial r_1}{\partial x_1} + \cdots + \frac{\partial r_m}{\partial x_n}\frac{\partial r_m}{\partial x_1} & \cdots 
  & \frac{\partial r_1}{\partial x_n}\frac{\partial r_1}{\partial x_n} + \cdots + \frac{\partial r_m}{\partial x_n}\frac{\partial r_m}{\partial x_n}
  \end{bmatrix} \\ 
  & =
  \begin{bmatrix}
  \frac{\partial^2 \bold r}{\partial x_1^2} & \cdots & \frac{\partial^2 \bold r}{\partial x_1\partial x_m} \\
  \vdots & \ddots & \vdots \\
  \frac{\partial^2 \bold r}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 \bold r}{\partial x_m^2}
  \end{bmatrix}
  \end{align}
  $$
  $\bold r(\bold x)^\intercal\nabla^2\bold r (\bold x)$ 는 무시 할 만큼 작다고 가정하자. Hessian은 Jacobian 곱하기 Jacobian이 되었다. Newton's method를 사용한 다음 이동할 위치를 계산해보자.
  $$
  \bold x_{t+1} = \bold x_t - (\bold J(\bold x)^\intercal \bold J(\bold x))^{-1}\bold J(\bold x)^\intercal \bold r(\bold x)
  $$
  이것이 **Gauss-Newton method**이다.

- **Lavenberg-Marquardt algorithm**

  Hessian 행렬이 positive definite 하지 않을 수 있어서 항상 convex하게 만들기 위해서 위에서 우리는 Levenberg-Marquardt modification을 적용했다. Gauss-Newton method에 LM modification을 적용한 것을 **Lavenberg-Marquardt algorithm** 이라고 한다.
  $$
  \bold x_{t+1} = \bold x_t - (\bold J(\bold x)^\intercal \bold J(\bold x) + \mu_t \bold I))^{-1}\bold J(\bold x)^\intercal \bold r(\bold x)
  $$
  더 나아가 convex한 parabola의 폭이 얇거나 넓을때 그 상황에 맞게 스텝사이즈를 조절 해줄수 있다. 폭이 좁으면 기울기가 급속히 변하니(Hessian이 큼) 스텝사이즈를 줄여주고 폭이 넓으면(Hessian이 작음) 스텝사이즈를 크게 해 줄수 있다.
  $$
  \bold x_{t+1} = \bold x_t - (\bold J(\bold x)^\intercal \bold J(\bold x) + \mu_t \cdot diag(\bold J(\bold x)^\intercal \bold J(\bold x)))^{-1}\bold J(\bold x)^\intercal \bold r(\bold x)
  $$
  $\mu$ 가 작으면 Gauss-Newton method 처럼 작동 하지만 크면 $diag(\bold J(\bold x)^\intercal \bold J(\bold x))$ 항은 스텝사이즈 처럼 작동한다.



# Continuous Optimization II: constrained

### Linear Programming

목적함수도 linear 형태이고 contraint들도 linear 형태이면 linear programming 문제라고 한다. 어느 하나라도 linear 하지 않다면 non-linear programming 문제가 된다.

### Nonlinear Programming: equality contraints

#### Methods of Lagrange Multiplier

이전에 배운 gradient descent나 Newton's method를 constraint 때문에 적용 하기 힘들다. 라그랑주 수학자가 contraint와 objective 함수를 하나의 식으로 묶어서 local maxima 혹은 local minima를 푸는 방식을 고안해냈다.  다음과 같은 최적화 문제를 생각해보자.
$$
\min \quad f(x) \quad \text{s.t.} \quad h(x) = 0
$$
라그랑지안 함수를 정의하자.
$$
\mathcal{L}(\bold x, \mu) = f(\bold x) + \mu h(\bold x)
$$
$\bold x$ 와 $\mu$ 에 대해서 미분 해보자. $\bold x^*$ 와 $\mu^*$가 $\mathcal{L}(x)$  의 minimizer라고 생각하자. 
$$
\begin{align}
& \nabla_{\bold x}\mathcal{L}(\bold x^*, \mu^*) = \bold 0 \longrightarrow -\nabla_{\bold x}f(\bold x^*) = \mu^*\nabla_{\bold x}h(\bold x^*) \tag1  \\
& \nabla_{\mu} \mathcal{L}(\bold x^*, \mu^*) = 0 \longrightarrow h(\bold x^*) = 0 \tag2 \\
& \bold y^\intercal(\nabla_{\bold x\bold x}^2) \mathcal{L}(\bold x^*, \mu^*) \geq 0 \quad \forall \bold y \quad \text{s.t.}\quad \nabla_{\bold x}h(\bold x^*)^\intercal \bold y = 0 \tag 3
\end{align}
$$
비록 우리는 $f(x)$ 를 최소화 하는 거였지만 constraint를 합쳐서 라그랑지안 함수를 만들어서 최소화를 하였다. 어거지로 끼워 맞춰서 풀었지만 모든 조건을 만족하니 최소화 문제를 풀었다고 할수 있다. 라그랑지안 함수를 $\mu$ 에 대해 미분 하면 $h(\bold x) = 0$ 조건이 만족 되는 것을 볼수 있다. 하지만 우리는 미분한 식이 0일때의 $x^*$ 을 찾은 것만으로 min인지 max인지 판단하기 힘들다. 그래서 (3)에서 2차 미분을 해서 Hessian 행렬을 구하고 임의의 벡터 $\bold y$ 를 곱해주고 양수면 positive definive 음수이면 negative definite인지 판단 할 수 있다.

### Nonlinear Programming: inequality constraints

다음 비선형 문제를 생각해보자. 우리는 이것을 **primal problem**이라고 부른다.
$$
\begin{align}
\min_\bold{x} f(\bold x) \quad \text{s.t.} & \quad g_i(\bold x) \leq 0, \quad i = 1, 2, ..., m \\
& \quad \bold x \in \Omega
\end{align}
$$
라그랑지안 함수를 정의하자.
$$
\begin{align}
\text{Lagrangian function:} \quad &\mathcal{L(\bold x, \bold u)} := f(\bold x) + \bold u^\intercal g(\bold x) \\
\text{Dual function:} \quad & \mathcal{L^*(\bold u)} := \min_{\bold x} f(\bold x) + \bold u^\intercal g(\bold x) & \quad \text{s.t.} \quad \bold x \in \Omega \\
\text{Dual problem:} \quad & D: d^* = \max_\bold{u} \mathcal{L}^*(\bold u) & \quad \text{s.t.} \quad \bold u \geq 0
\end{align}
$$
라그랑지안 함수를 $\bold x$ 에 대해 미분하고 0으로 만들어주는 $\bold x^*$ 를 구한다. $\bold x^* $ 를 찾았으니 **Dual function**은 $\bold u$ 만의 함수이다. $\mathcal{L}^*$에는 $\bold x$에 대해 미분 하는 모든 것이 들어 있는 것이다. 이제 Dual function을 최적화 해주는 $\bold u^*$ 를 찾으면 된다. 이 문제는 **Dual problem** 이라고 한다.  여기서 왜 우리는 dual function을 maximize를 하는 걸까? 이유는 우리가 라그랑지안 함수를 $\bold x$ 에 대해 최적화한 dual function가 concave 하기 때문이다.

> **Concave functions**
>
> - A function $f(x )$ is a **concave** function if $f(\lambda x + (1-\lambda)y) \geq \lambda f(x) + (1-\lambda) f(y)$ for all $x$ and $y$ and for all $\lambda \in [0, 1]$
> - A function $f(x )$ is a **strictly concave** function if $f(\lambda x + (1-\lambda)y) \gt \lambda f(x) + (1-\lambda) f(y)$ for all $x$ and $y$, $x \neq y$, and for all $\lambda \in (0, 1)$

> **Themrem** : The dual function $\mathcal{L}^*( u)$ is a concave function.
>
> **Proof** : Let $u_1 \geq 0$ and $u_2 \geq 0$ be two values of the dual variables, and let 
> $$
> u = \lambda u_1 + (1-\lambda)u_2, \quad \text{where $\lambda \in [0, 1]$} \\
> \begin{align}
> \mathcal{L}^*(u)
> & = \min_{x\in P} f(x) + u^\intercal g(x) \\
> & = \min_{x \in P} f(x) + u_1^\intercal g(x) + (1- \lambda)[f(x) + u_2^\intercal g(x)] \\
> & \geq \lambda[\min_{x \in P} f(x) + u_1^\intercal g(x)] + (1- \lambda)[\min_{x\in P} (f(x) + u_2^\intercal g(x))] \\
> & = \lambda \mathcal{L}^*(u_1) + (1- \lambda)\mathcal{L}^* (u_2) \\
> & \therefore \mathcal{L^*(x)} \ \text{is a concave function}
> \end{align}
> $$

> **Example**
> $$
> \min f(\bold x) = 6 x_1 x_2 \quad \text{s.t.} \quad g(\bold x) = 3x_1 + 4x_2 = 18 \\
> $$
>
> $$
> \begin{align}
> \mathcal{L(\bold x, \bold u)} 
> & = f(\bold x) + u \ g(\bold x) \\
> & = 6x_1x_2 + u(3x_1 + 4x_2 - 18) \tag 1
> \end{align}
> $$
>
> $$
> \begin{align}
> & \frac{\partial \mathcal{L}}{\partial x_1} = 6x_2 + 3u = 0 \rightarrow x_2 = -\frac{1}{2}u \tag 2 \\
> & \frac{\partial \mathcal{L}}{\partial x_2} = 6x_2 + 4u = 0 \rightarrow x_1 = -\frac{2}{3}u \tag 3 \\
> & \frac{\partial \mathcal{L}}{\partial u} = 3x_1 + 4x_2 - 18 = 0 \tag 4
> \end{align}
> $$
>
> (2)와 (3)을 (1)에 대입해서 $u$에 대해 풀어보자.
> $$
> \mathcal{L^*(u)} = -2u^2 - 18 u \\
> \therefore \mathcal{L^*(u)} \text{ is concave}
> $$
> Minimizer를 찾아보자. (2)와 (3)을 (4) 대입해서 $u$에 대해 풀어보자.
> $$
> \begin{align}
> u^* & = -4.5 \\
> x_1^* & = 3 \\
> x_2^* & = \frac{9}{4}
> \end{align}
> $$







#### Karush-Kuhn-Tucker(KKT) Conditions

최적 포인트 $\bold x^*$가 있다면 다음과 같은 **KKT Conditions**이 만족해야 한다.  다음 예로 어떤 것들인지 알아보자. 
$$
\min_{\bold x\in \mathbb{R}^2} f(\bold x) \quad \text{s.t.}\quad g(\bold x) \leq 0 \\
\text{where }\quad f(\bold x) = (x_1 - 1.1)^2 + (x_2 + 1.1)^2 \quad \text{and }\quad g(\bold x) = x_1^2 + x_2^2 - 1
$$
$f(\bold x)$ 는 4사분면에 있는 minumum이 $x_1 = 1.1$ 와 $x_2 = -1.1$ 에 위치한 convex한 paraboloid이다. $g(\bold x)$ 는 origin에  minimum이 위치하고 $g(\bold x)$ 의 절편은 -1인 paraboloid이다. 조건을 만족하는 local minimum은 다음 일때 만족 된다.
$$
-\nabla_{\bold x}f(\bold x) = \lambda \nabla_{\bold x}g(\bold x) \quad \text{and} \quad \lambda \gt 0
$$
$\nabla f(\bold x)$ 와 $\nabla g(\bold x)$ 의 방향은 반대여야 함으로 전자에 $-$ 을 붙혀줬다. $f(\bold x)$ 는 convex 하기 때문에 gradient가 바깥쪽으로 향해서 있다. minimize 해야 함으로 -1을 곱해 $\nabla g(\bold x)$ 와 같은 방향을 만들어 줬다.



두가지의 상황에 대해 생각해보자.

| Case I                                                       | Case II                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Unconstrained local minimum occurs in the feasible region    | Unconstrained local minimum likes outside the feasible region |
| $g(\bold x^*)  < 0$                                          | $g(\bold x^*) = 0$ : $\bold x^*$ 는 $g(\bold x)$ 테두리(솔루션에 가장 가까운)에 존재한다. |
| $\nabla_{\bold x} f(\bold x^*) = \bold 0$                    | $-\nabla_{\bold x} f(x^*) = \lambda \nabla_{\bold x}g(\bold x^*) \quad \text{with} \quad \lambda \gt 0$ : 조건 등식인지 부등식인지에 따라 $\lambda​$ 가 플러스 인지 마이너스인지 바뀔 수 있다. |
| $\nabla_{\bold x\bold x}f(\bold x^*)$ is a positive semi-definite matrix | $\bold y^\intercal\nabla_{\bold x\bold x}\mathcal{L}(\bold x^*)\bold y \geq 0$ for all $\bold y​$ orthogonal to $\nabla_\bold{x} g(\bold x^*)​$ |

조건이 있는 최적화 문제가 있을 때 $\bold x^*$ 라는 정답을 찾았다면 위와 같은 KKT 조건을 만족해야 한다.

위의 케이스들을 일반화 하자. 

아래와 같은 문제가 주어졌을때,
$$
\min_{\bold x\in \mathbb{R}^n}f(\bold x) \quad \text{s.t.} \quad g(\bold x) \leq 0
$$
라그랑지안 함수를 정의하자.
$$
\mathcal{L}(\bold x, \lambda) = f(\bold x) + \lambda g(\bold x)
$$
$\bold x^*$ is a local minimum $\Longleftrightarrow$ $\exists! \ \lambda^*$

1. $\nabla_{\bold x}\mathcal{L}(\lambda^*, \bold u^*) = \bold 0$
2. $\lambda^* \geq 0$ : Case II에서는 $\lambda$ 가 0이 양수여야 되고 Case I에서는 $\lambda$ 가 0이여야 한다.
3. $\lambda g(\bold x^*) = 0$ : maximize 해야 되는데 그럼 $\bold x^*$ 일 때 조건에 맞게 최대값 0 이어야 한다.
4. $g(\bold x^*) \leq 0$
5. positive definite constraints on $\nabla_{\bold x \bold x}\mathcal{L}(\bold x^*, \lambda^*)$



등식과 부등식등을 조건으로 가진 문제를 통해 더 일반화 해보자.
$$
\min_{\bold x \in \mathbb{R}^n} f(\bold x) \quad \text{s.t.} \quad
\begin{cases}
h_i(\bold x) = 0 \quad \text{for} \quad i = 1,..., l \\
g_j(\bold x) \leq 0 \quad \text{for }\quad j = 1,..., m
\end{cases}
$$
라그랑지안 함수를 정의하자.
$$
\mathcal{L}(\bold x, \bold \mu, \bold \lambda) = f(\bold x) + \bold \mu^\intercal \bold h(\bold x) + \lambda^\intercal \bold g (\bold x)
$$
$\bold x^*$ is a local minimum $\Longleftrightarrow$ $\exists! \ \lambda^*, \mu^*$ such that

1. $\nabla_{\bold x} \mathcal{L}(\bold x^*, \bold \mu^*, \bold \lambda^*) = \bold 0$
2. $\lambda_j^* \geq 0$ for $j=1,...,m$
3. $\lambda_j^*g_j(\bold x^*) = 0$ for $j = 1, ..., m$
4. $g_j(\bold x^*) \leq 0 $ for $j=1,..., m$
5. $\bold h(\bold x^*) = \bold 0$
6. Positive definite constraints on $\nabla_{\bold x\bold x}\mathcal{L}(\bold x^*, \bold \lambda ^*)$



Constrained optimization 문제에 솔루션 포인트 $\bold x^*$ 가 있다면 KKT 조건들을 만족시킨다. 만약 주어진 문제가 Non-linear programming에서도 convex programming 이라면 KKT 조건이 필요 조건 일 수도 있고 충분 조건 일 수도 있다. 즉, KKT 조건들이 만족 되면 솔루션이 반듯이 존재한다.

 Convex 하지 않은 문제는 relaxation을 통해 convex하게 만들어서 풀수도 있고 relaxation도 힘들면 gradient descent를 통해 최적활 할 수도 있다.

### Nonlinear programming: others

