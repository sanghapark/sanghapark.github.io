# Taylor Expansion & Gradient Descent/Ascent

테일러 급수는 어떤 복잡한 수식도 어느 한점의 정보만 주어진다면 그 점의 주변 함수 값을 유추 할 수 있다. 1차 도함수만 주면 유추의 에러는 크다. 많은 도함수를 줄수록 더 가깝게 실제값에 근사할 수 있다.

## Taylor Expansion
함수 $f(x)$ 를 $x=a$ 일 때 테일러 급수를 이용하여 $f(x=a)$ 를 근사 할 수 있다.
$$
\begin{align}
f(x=a) = f_{taylor}(x=a) 
& = f(a) + \frac{f^{'}(a)}{1!}(x-a) + \frac{f^{''}(a)}{2!}(x-a)^2 + \frac{f^{'''}(a)}{3!}(x-a)^3 + \cdots  \\
& = \sum_{n=0}^\infty\frac{f^{(n)}(a)}{n!}(x-a)^n 
\end{align}
$$


## Gradient Descent/Ascent

Gradient Descent/Ascent는 아래와 같이 동작한다.
1. 미분 가능한 $f(x)$와 초기 모수값 $x_i$가 주어진다.
2. 모수 $x_i$를 조금 움직여서 더 낮은 혹은 높은 $f(x)$값으로 이동한다. 이동에는 방향과 속력이 필요하다. 방향은 $\nabla f(x)​$ 의 nagative/positive 방향으로 잡는다. 속력은 알아서 선택한다.
$$
f(x) = f(a) + \frac{f^{'}(a)}{1!}(x-a) + \mathcal{O}\big(\big| |x-a| \big|^2\big)
$$
테일러 급수 1차 도함수 이후에 있는 텀들을 다 더해도 그것의 growth는 Big-O notation 텀에 국한되어진다. 초기값을 $a=x_1$로 정하고 다음 위치는 $x=x_1+h\textbf{u}$이다.  $\textbf{u}$는 편미분의 unit direction vector이다. 위의 식에 대입 해보자.
$$
f(x_1 + h\textbf{u}) = f(x_1) + hf^{'}(x_1)\textbf{u} + h^2\mathcal{O}(1) \\
f(x_1+h\textbf{u}) - f(x_1) \approx hf^{'}(x_1)\textbf{u}
$$
$h​$가 엄청 작은 수이면 제곱수 이기 때문에 $\mathcal{O}​$ 항은 무시 할 수 있다. 

우리가 하려고 하는 건 어떤 방향으로 나아가는지 정하는 것이다. 함수를 증가시키는 방향도 있고 감소 시키는 방향도 있다. 그리고 우린 증가/감소를 최대화해서 이동을 하고 싶어한다. 아래 수식은 이동을 가장 작게하는 unit vector $\textbf{u}$를 구하는 것이다.  위에 수식을 사용하여 밑에 식으로 전개 할 수 있다.
$$
\begin{align}
\textbf{u}^* 
& = \underset{\textbf{u}}{\arg\min}\,\, \{f(x_1 + h\textbf{u}) - f(x_1)\} \\
& = \underset{\textbf{u}}{\arg\min}\,\, hf^{'}(x_1)\textbf{u} \\
& = -\frac{f^{'}(x_1)}{\mid f^{'}{x_1}\mid}
\end{align}
$$
아래와 같이 이동하면 된다.
$$
x_{t+1} \longleftarrow x_t + h\textbf{u}^* = x_t - h\frac{f^{'}(x_1)}{f^{'}(x_1)}
$$


## 왜 gradient는 가장 가파른 방향인가?
gradient의 각 구성 요소는 함수가 각 축으러 얼마나 빨리 변하는지를 알려준다. 어떤 방향 $\vec{v}$이 주어지면 함수는 그 방향으로 얼마나 빨리 변하는지는 아래와 같다.
$$
\nabla f(a) \cdot \vec{v} = |\nabla f(a)||\vec{v}|\cos(\theta)
$$
$cos(\theta)$ 가 $1$일때 가장 빨리 변하는 것을 알수 있다. 즉 $\theta=0$일때 $\nabla f(a)$와 $\vec{v}$가  평행 할때 가장 빠르게 변하는 것을 할 수 있다.
























