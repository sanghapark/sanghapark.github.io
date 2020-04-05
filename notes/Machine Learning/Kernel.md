# Kernel

Kernel is a function that returns the result of a dot product performed in another space.

Kernel은 두개의 벡터 $\bold u, \bold v \in \mathbb{R}^n$가 기존 $n$ 차원 보다 더 높은 고차원 $m$ 으로 펌핑되어진 두개의 벡터의 inner product이다. $\mathbb{R}^n$ 차원의 벡터를 $\mathbb{R}^m$ 차원 공간으로 매핑 해주는 함수 $\phi: \mathbb{R}^n \rightarrow \mathbb{R}^m$ 있다고 하자. 펌핑된 두개의 벡터 $\phi(\bold a \in \mathbb{R}^n)$ 와 $\phi(\bold b \in \mathbb{R}^n)$ 의  점곱(dot product)은 다음과 같다. 펌핑된 차원의 두개의 벡터 점곱을 Kernel 함수 $k$ 라고 하자.
$$
k(\bold a, \bold b) = \phi(\bold a)^\intercal \cdot \phi(\bold b)
$$
커널 함수가 유용한 점은 고차원으로 펌핑된 두개의 벡터의 점곱을 구할때 두개의 벡터 각각을 고차원으로 펌핑하고 점곱을 하지 않고 바로 고차원으로 펌핑된 두개 벡터의 점곱의 형태를 아니 고차원으로 변형할 필요 없이 점곱을 구할 수 있다는 것이다.

예를 들어보자.

2차원의 벡터 $\bold a = (a_1, a_2)​$ 와 $\bold b = (b_1, b_2)​$ 가 있다고 하자. 차원 뻥튀기 함수를 아래와 같이 정의하자.
$$
\phi(\bold x) = \phi(x_1, x_2) = (1, x_1^2, x_2^2, \sqrt2x_1, \sqrt2x_2, \sqrt2 x_1 x_2)
$$
점곱은 다음과 같다.
$$
\begin{align}
\phi(\bold a)^\intercal \cdot \phi(\bold b) 
& = (1, a_1^2, a_2^2, \sqrt2 a_1, \sqrt2 a_2, \sqrt2 a_1 a_2)\cdot (1, b_1^2, b_2^2, \sqrt2 b_1, \sqrt2 b_2, \sqrt2 b_1 b_2) \\
& = 1^2 + a_1^2b_1^2 + a_2^2b_2^2 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 \\
& = (1 + a_1 b_1 + a_2 b_2)^2 \\
& = (1 + \bold a^\intercal \bold b)^2 \\
& = k(\bold a, \bold b)
\end{align}
$$
커널 함수 $k(\bold a, \bold b)$ 는 각 데이터를 고차원으로 변환 하지 않고 바로 고차원으로 변환된 두개의 벡터 점곱을  대신 해줄 수 있다.



## Kernel Trick Functions

1. Linear kernel
   $$
   k(\bold a, \bold b) = \bold a \cdot \bold b
   $$
   

2. Polynomial kernel

   $d$가 높으면 높을 수록 트레이닝 데이터에서의 퍼포먼스는 높아지겠지만 오버피팅이 발생 할 수 있다. The kernel returned the result of a dot product performed in $\mathbb{R}^d$

3. $$
   k(\bold a, \bold b) = (\bold a \cdot \bold b  + c)^d
   $$

3. Radial Basis Function (Gaussian kernel)

   The RBF kernel returns the result of a dot product performed in $\mathbb{R}^\infty$

$$
k(\bold a, \bold b) = \exp(-\gamma \Vert \bold a - \bold b\Vert^2 )
$$





