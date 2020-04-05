# Lagrange Multiplier

## Lagrange Multipliers

Given, 
$$
\min \quad f(\bold x) \quad \text{s.t.} \quad g(\bold x) = 0
$$
그레이던트가 서로 평행 할 때가 해이다.
$$
\nabla f(\bold x) = \lambda \nabla g(\bold x)\\
\nabla f(\bold x) - \alpha \nabla g(\bold x) = 0
$$
위의 식을 풀면 해가 나온다.

----

## Example 01

최소/최대를 잘 풀수 있는 방법이다. 예를들어
$$
\max x + 2y \quad \text{s.t.} \quad x^2 + y^2 = 1
$$
두개의 식이 접하는 부분을 찾아야 하는 문제이다.
$$
\begin{align}
\nabla (x^2+y^2) & = k \nabla(x + 2y) \\
(2x, 2y) & = k(1, 2) \\
\end{align}
$$
위에 식을 풀어보자
$$
2x = k, \quad 2y = 2k \\
2x = k, \quad y = k \\
\therefore 2x = y
$$
또 다른 문제를 풀어보자
$$
\max xy \quad \text{s.t.} \quad x^m + y^n = m+n
$$
풀이는 다음과 같다.
$$
\begin{align}
\nabla (x^m + y^n) & = \lambda \nabla xy \\
(mx^{m-1}, ny^{n-1}) & = \lambda (y, x) \\
\end{align}
$$
위의 식을 풀면
$$
\begin{align}
\frac{mx^{m-1}}{y} & = \frac{ny^{n-1}}{x} \\
mx^m & = ny^ n
\end{align}
$$
$mx^m = ny^n$ 일 때 최대가 된다. 



## Lagrange multipliers - Dual variables

밑에 문제를 풀어보자.
$$
\min_x x^2 \quad \text{s.t.} \quad x \geq b
$$
Move the constraint to objective function
$$
L(x, \alpha) = x^2 - \alpha(x-b) \quad \text{s.t.} \quad \alpha \geq 0
$$
Solve
$$
\min_x \max_\alpha L(x, \alpha) \quad \text{s.t.} \quad \alpha \geq 0 \\
\min_x \max_\alpha \bigg\{x^2 - \alpha(x-b)\bigg\}\quad \text{s.t.} \quad \alpha \geq 0 \\
\min_x x^2 \max_\alpha\alpha(x-b) \quad \text{s.t.} \quad \alpha \geq 0
$$


