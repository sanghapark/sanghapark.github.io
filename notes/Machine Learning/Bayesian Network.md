# Bayesian Network

## Objectives

1. 베이지안 네트워크의 이해
   - Understand the syntax and the semantics of Bayesian networks
   - Understand how to factorize Bayesian networks
   - Able to calculate a probability with given conditions
2. Inference of Bayesian networks
   - Able to calculate parameters of Bayesian networks
   - Able to list the exact inference of Bayesian networks



## **Prerequisites: Probability**

### **Marginalization**

두개의 확률 변수 $a​$ 와 $b​$ 가 있다. 각각에 대해 알고 싶다면 $a​$ 와 $b​$ 에 대한 결합 확률(joint probability)를 알면 **marginalization** 을 통해서 개별확률에 대해서 알 수 있다. 
$$
P(a) = \sum_b P(a, b) = \sum_b P(a \vert b)\ P(b)
$$
예를들어,
$$
\sum_b P(a, \ b) = P(a, \ b=\text{true}) + P(a, \ b=\text{false}) = P(a)
$$
그럼 반대로 $P(a)$ 와 $P(b)$ 에 대해 알고 있다면 $P(a, b)$에 대해서 알 수 있을까? 쉬운 작업은 아니다. 만약 결합확률은 모르고 조건부 확률을 안다면 다음을 사용 할 수 있다. 대신에 개별 특정 확률 변수에 대해서 알아야 한다.
$$
P(a) = \sum_b P(a \vert b) \ P(b)
$$
$P(a, \ b, \ c, \ d)$ 를 안다면
$$
P(b) = \sum_a \sum_c \sum_d P(a, b, c, d)
$$
결합 확률과 개별 확률을 알면 조건부 확률에 대해서도 계산 할 수 있다.
$$
P(c \ \vert \ b) = \sum_a \sum_d P(a, c, d \ \vert \ b)= \frac{\sum_a \sum_d P(a, b, c, d)}{P(b)} = \frac{P(b, c)}{P(b)}
$$

### **Chain Rule of Factorization**

어떤 결합 확률은 확률곱의 시리즈로 factorization 할 수 있다. 
$$
\begin{align}
& P(a, b, c, ..., z) = P(a \ \vert \ b, c, d,..., z)P(b, c, ..., z) \\
& P(a, b, c, ..., z) = P(a \ \vert \ b, c, d, ..., z)P(b \ \vert \ c, d, ..., z) P(c \ \vert \ d, ..., z)...P(z)
\end{align}
$$

### **Independence**

두 확률 변수 $A$ 와 $B$ 가 있을때 $B$ 를 알아도 $A$ 에는 어떤 영향도 주지 않는다. 이때 우리는 $A$ 와 $B$ 는 독립적이라고 한다.
$$
\begin{align}
P(A \vert B) & = P(A) \\
P(A, B) & = P(A)P(B)
\end{align}
$$
일반화 하면 모든 확률 변수가 독릭적이라면 다음과 같다.
$$
P(C_1, C_2, C_3, ..., C_n) = \prod_{i=1}^n P(C_i)
$$

#### Marginal Independence

조건부가 어떤 영향도 주지 않으면 **marginal independent** 하다.
$$
\begin{align}
& P(A=\text{true} \vert B=\text{true}) = P(A) \quad \text{or} \\
& P(A=\text{true}, B=\text{true}) = P(A=\text{true})P(B=\text{true})
\end{align}
$$

#### Conditional Independence

<img src="/Users/kakao/Documents/notes/Machine Learning/assets/image-20190216142848472.png" style="zoom:50%" />

Yellow 가 주어졌을때 Red 와 Blue 가 **conditional independent** 하면 다음식을 만족한다. Purple은 Red와 Blue가 겹쳐진 부분이다. Yellow가 주어진 상태에서 Red와 Blue 결합 확률은 다음과 같다.
$$
P(\text R, \text B \vert \text Y ) = P(\text R \vert \text B, \text Y)P( \text B \vert \text Y) = P(\text R \vert \text Y)P(\text B \vert \text Y)
$$

$$
P(\text R \vert \text B, \text Y) = P(\text R \vert \text Y)
$$

## **Interpretation of Bayesian Network**

베이지안 네트워크는 결합확률을 표현 할 수 있는 방법이다. 그래프는 acyclic 하고 directed 하다. 각 노드는 random variable을 의미 한다. 노드사이에 있는 링크는 노드간의 상관관계를 의미한다.

## **Bayes Ball Algorithm**

|                        Common Parent                         |                          Cascading                           |                         V-Structure                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20190216162902361](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190216162902361.png) | ![image-20190216162841376](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190216162841376.png) | ![image-20190216162916827](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190216162916827.png) |
|        $\text {J} \perp \text {M} \ \vert \ \text{A}$        |           $ \text B \perp \text M \vert \text A $            |         $\sim (\text B \perp \text E \vert \text A)$         |
| $P(\text J, \text M \vert \text A) = P(\text J \vert \text A) P(\text M \vert \text A)$ | $P(\text M \vert \text B, \text A) = P(\text M \vert \text A)$ | $P(\text B, \text E, \text A) = P(\text B) P(\text E) P(\text A \vert \text B, \text E)$ |
| 알람이 울렸으면 John이 나한테 전화하건 안하건 Mary는 나한테 어떤 영향도 없이 전화 할 것이다. 알람이 울린지 안울린지 모르는 상태인데 John이 나한테 전화 했으면 Mary도 알람이 울렸을수도 있으니 매리가 나한테 전화 할 수 있다. | 알람이 울렸으면 매리가 나한테 전화 할 것이다. 근데 알람의 상태를 모르는데 도둑이 들었으면 매리가 나한테 전화하는 것에 영향을 끼침 | 알람에 대해 모름. 도둑 드는거에 지진일어나는 거랑 아무 상관 없음. 하지만 알람이 울림. 그러면 도둑이 들었을수도 있고 지진이 일어 났을 수도 있음. 즉 서로 도둑이 들었으면 지진이 안일어 났을수도 있고 도둑이 안 들었으면 지진이 났을수 있음. |

베이지안 네트워크를 분석 해서 확률을 계산 할 때 conditional independence를 알고 있는게 유용하다. Full joint probability를 fatorization을 할 때 사용된다.

베이즈 볼 알고리즘은 한 노드 a 에서 다른 노드 b로 이동 할 때 알려져 있는 노드에 막혀서 못 가면 a와 b는 independent하고 이동 할 수 있다면 dependent하다고 할 수 있다. 이것은 Common parent와 Cascading 케이스에서 유효하다. V-Structure에서는 반대이다. 공통 자식 노드가 알려져 있으면 부모 노드간 이동이 가능 하다고 보고 dependent 하고 자식 노드에 대해 모르면 부모 노드간 이동이 불가능해서 independent 하다.

### **Markov Blanket** 

| 그림                                                         | 설명                                                         |
| ------------------------------------------------------------ | :----------------------------------------------------------- |
| ![image-20190216171257940](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190216171257940.png) | 노드 A에 대해 다른 노드들의 영향을 막아주는 주변 노드들의 집합을 Markov Blanket이라고 한다. 부모노드들과 자식 노드들은 직접적인 영향이 있는 노드들이다. 하지만 자식 노드의 부모노드들까지 포함 해야 한다. 이유는 V-Structure에서 발생하는 자식이 알려지면 자식의 부모를 통해 영향력을 끼칠수 있는 노드들이 생기기 때문이다. |

### **D-Separation**

$X$ is d-seperated (directly separated) from $Z$ given $Y$ if we cannot send a ball from any node in $X$ to any node in $Z$ using the Bayes ball algorithm.
$$
\text X \perp \text Z \ \vert \ Y
$$

## Factorization of Bayesian Network

<img src="/Users/kakao/Documents/notes/Machine Learning/assets/image-20190216173738432.png" style="zoom:30%" />

Full joint probability를 표현 해보자.
$$
\begin{align}
& P(X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8) \\ 
& = P(X_1)P(X_2)P(X_3\vert X_1)P(X_4 \vert X_2) P(X_6 \vert X_3, X_4)P(X_5 \vert X_2) P(X_7 \vert X_6) P(X_8 \vert X_5, X_6)
\end{align}
$$
Conditional independence를 찾고 marginalization을 통해서 우리는 위 식의 parameter 갯수를 줄여 나갈 수 있다.

## Inference Questions on Bayesian Networks

Probability inference에 대해 알아보자.

<img src="/Users/kakao/Documents/notes/Machine Learning/assets/bn01.png" style="zoom:10%" />

#### Inference Question: Conditional Probability

확률 변수는 관측된 evidence 확률 변수와 숨어있는 hidden  확률 변수로 나누어진다. Hidden 확률변수들은 관심있는 변수와 관심 없는 변수로 나늬어 질수 있다.

| Evidence Variable $X_v$ | Hidden Variable $X_h$             |
| ----------------------- | --------------------------------- |
|                         | $Y$: 관심 있는 hidden 확률 변수들 |
|                         | $Z$: 관심 없는 hidden 확률 변수들 |

다음과 같은 식으료 표현 할 수 있다.
$$
\begin{align}
P(Y \vert x_v) 
& = \sum_Z P(Y, Z= z \vert x_v) \\
& = \sum_Z \frac{P(Y, Z, x_v)}{P(x_v)} \\
& = \sum_Z \frac{P(Y, Z, x_v)}{\sum_{y, z}P(Y=y, Z=z, x_v)}
\end{align}
$$
Given a set of evidence, what is the conditional probability of interested hidden variables?
$$
P( \text A \vert \text B = \text{True}, \text M = \text{True}) = \text{?}
$$
일부 결합 확률에 대해서 알면 조건부 확률을 구할 수 있다.

#### Inference Question: Most Probable Assignment

$$
\underset{a}{\arg \max} \ P(\text A \vert \text B = \text{true}, \text M = \text{true}) = \text{?}
$$

MAP를 사용해서 찾을수 있다. 다음과 같이 조건부 확률 문제를 나눌 수 있다.

- Prediction: $P(\text A \vert \text B, \text E)$
- Diagnosis: $P(\text B, \text E \vert \text A)$

## Variable Elimiation

### Marginalization and Elimination

전체 혹은 부분 결합 확률식을 구하고 팩토리제이션을 한다. 부분 결합확률을 구하기 위해서는 우선 전체 결합 확률 식을  hidden variable들에 대해 summation을 통한 marginalization을 하면 된다. 전체 결합 확률의 식은 베이지안 네트워크를 통해 factorization을 할 수 있다.
$$
\begin{align}
P(a=\text{true}, b = \text{true}, m = \text{true})
& = \sum_{J} \sum_{E} P(a, b, E, J, m) \\
&= \sum_{J} \sum_{E} P(J \vert a) P(M \vert a) P(a \vert b, E) P(E)P(b)
\end{align}
$$
이제 주어지는 conditional probability table 에 있는 확률을 사용해서 부분 결합확률을 수치화 하면 된다. 여기서 우리는 좀 더 계산을 간단하게 할 수 있을까? Big O notation 통해 계산복잡도를 통해 알아보자. Summation 한번 할 때 곱하기를 네번을 한다. Summation에 독립적인 항은 앞으로 뺄 수 있다.
$$
\begin{align}
P(a=\text{true}, b = \text{true}, m = \text{true})
& = \sum_{J} \sum_{E} P(a, b, E, J, m) \\
& = \sum_{J} \sum_{E} P(J \vert a) P(M \vert a) P(a \vert b, E) P(E)P(b) \\
& = P(M \vert a)P(b) \sum_{J} P(J \vert a) \sum_{E} P( a \vert b, E)P(E)
\end{align}
$$
계산 복잡도가 줄어든 것을 볼수 있다. 계산 복잡도를 줄이는 체계적인 방법에 대해 알아보자

### Variable Elimination

$P(e \vert j, m) = \alpha P(e, j, m)$ 식에서 우리는 $\alpha$ 는 normalizing constant $\frac{1}{P(j, m)}$ 인 것을 알 수 있다. 이것을 고려 하여 다음 부분 결합 확률을 구해보자
$$
\begin{align}
P(e, j, m, B, A)
& = \alpha P(e) \sum_B P(b) \sum_A P(a \vert b, e) p(j\vert a)P(m \vert a)
\end{align}
$$
여기서 우리는 확률 분포 함수를 하나의 함수로 생각할 수 있다.  주어지는 probability table을 통해서 항을 곱하고 maliginalization을 진행 하면서 아래와 같이 변수들을 주여 나갈수 있다.
$$
\begin{align}
P(e, j, m, B, A)
& = \alpha P(e) \sum_B P(b) \sum_A P(a \vert b, e) p(j\vert a)P(m \vert a) \\
& = \alpha f_E(e) \sum_B f_B(b) \sum_A(a, b, e)f_J(a)f_M(a) \\
& = \alpha f_E(e) \sum_B f_b(b) \sum_Af_A(a, b, e)f_{JM}(a) \\
& = \alpha f_E(e) \sum_B f_B(b) f_{\bar{A}JM}(b, e) \\
& = \alpha f_E(e) \sum_B f_{B\bar{A}JM}(be) \\
& = \alpha f_E(e) f_{\bar B \bar A JM} (e) \\
& = \alpha f_{E \bar B \bar A JM}(e)
\end{align}
$$


## Potential Function and Clique Graph

Probability inference를 좀더 효율적으로 해보자. 큰 베이지안 네트워크에서 특정 observation이 있고 거기서 만들어지는 belief가 다른 노드들에 전파를 하는 알고리즘에 대해 알아보자. 무턱대고 위에서 배운 방식으로 summation을 통해 하는 것보다 좀 더 효율적으로 확률값을 계산 할 수 있다. Clique는 그래프의 부분 집합으로 그 부분 집합에 속하는 모든 노드들 간에링크가 존재하는 경우를 의미한다. 클리크에 속하는 노드들은 서로 간에 완전히 연결 되어있다. 전에 배운 것 처럼 무턱데로 summation을 통해서 구하는 것보다는 서브그룹으로 나누어서 하면 효율적일 수 있다.

### Potential Fuctions

확률에 관련된 함수이다. 하지만 normalized된 확률분포 함수는 아니다.

- Potential function on nodes: $\psi(a, b), \ \psi(b, c), \ \psi(c, d)$
- Potential function on links: $\phi(b), \  \phi(c)$

| Bayesian Network                                             | Clique and Seperator                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20190218001243108](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190218001243108.png) | ![image-20190218001311277](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190218001311277.png) |

$$
P(A, B, C, D) = P(A\vert B) P(B\vert C) P(C \vert D) P(D) \tag1
$$

위의 예제를 사용해서 potential function을 사용해보자. 아래와 같이 표현 할 수 있다고 potential function들에 가정해보자.


$$
p(A, B, C, D) = \frac{\prod_N \psi(N)}{\prod_L \psi(L)} = \frac{\psi(a, b) \psi(b, c) \psi(c, d)}{\phi(b) \phi(c)} \tag2
$$
위의 (1) 식과 (2) 식을 동일하게 만들어 줄수 있다. 같게 만들어 줄려면 $\psi$ 와 $\phi$ 를 아래와 같이 정의 해줄 수 있다.
$$
\begin{align}
& \psi(a, b) = P(A \vert B) \\
& \psi(b, c) = P(B \vert C) \\
& \psi(c, d) = P(C \vert D)P(D) \\
& \phi(b) = 1 \\
& \phi(c) = 1
\end{align}
$$
Potential function을 다른 방식으로 정의 할 수는 없을까? 다음과 같이 정의 해줄수 있다.
$$
p(A, B, C, D) = \frac{\prod_N \psi(N)}{\prod_L \psi(L)} = \frac{\psi^*(a, b) \psi^*(b, c) \psi^*(c, d)}{\phi^*(b) \phi^*(c)}
$$
위의 식을 만족하기 위한 $\psi^*$ 와 $\phi^*$ 는 다음과 같이 정의 할 수 있다.
$$
\begin{align}
& \psi^*(a, b) = P(A, B) \\
& \psi^*(b, c) = P(B, C) \\
& \psi^*(c, d) = P(C, D) \\
& \phi^*(b) = P(B) \\
& \phi^*(c) = P(C)
\end{align}
$$
Potential function에 또한 marginalization을 적용 가능하다. $w$ 는 모든 확률변수들 $v$ 의 부분집합이다.
$$
\psi(w) = \sum_{v-w}\psi(v)
$$


### Absorption in Clique Graph

Clique 그래프에서의 absorption이라는 operation을 알아보자. 아래 예제는 트리구조의 clique graph에서만 적용가능하다. 다음과 같이 가정하자.
$$
\begin{align}
P(B) & = \sum_A \psi(A, B), \quad \psi(A, B) = P(A, B)\\
P(B) & = \sum_C \psi(B, C). \quad \psi(B, C) = P(B, C) \\
P(B) & = \phi(B)
\end{align}
$$
$\psi$ 와 $\phi$ 를 어떻게 찾을 수 있을까? 위 식들의 같은 성질을 활용하여 inference을 할 것이다.  다음과 같은 데이터 관측, $P(A, B) \rightarrow P(A=1, B)$, 이 있다면 하나의 $\psi$ 변화가 다른 $\psi$ 들에게도 영향을 끼칠 것이다.이것을 **Belief Propagation** 이라고 한다. Belief propagation 은 어떻게 진행 될까? Absorption (update) Rule에 대해 알아보자. 업데이트된 $\psi^*(A, B)$, $\psi(B, C)$, $\phi(B)$ 에 대해 생각해보자. Seperator $B$가 업데이트 되는 방법은 처음 클리크 $A$, $B$ 에 대해 관측을 해서 알고 있다고 가정을 하기 때문에 $A$에 대해 marginalization을 해주면 된다.
$$
\phi^*(B) = \sum_A \psi^*(A, B)
$$
Clique $B$, $C$ 은 다음과 같이 업데이트 될 수 있다.
$$
\psi^*(B, C) = \psi(B, C)\frac{\phi^*(B)}{\phi(B)}
$$
왜 위와 같은 식이 성립하는지 보자.
$$
\begin{align}
\sum_C \psi^*(B, C)
& = \sum_C \psi(B, C) \frac{\phi^*(B)}{\phi(B)} \\
& = \frac{\phi^*(B)}{\phi(B)}\sum_C \psi(B, C) \\
& = \frac{\phi^*(B)}{\phi(B)} \phi(B) \\
& = \sum_A \psi^*(A, B)
\end{align}
$$
Local consistency가 성립하는 것을 볼 수 있다. Seperator $C$ 와 clique $CD$에 대해서 같은 방식으로 업데이트 할 수 있다.

### Simple Example of Belief Propagation

| Bayesian Network                                             | Clique Graph                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20190219020602131](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190219020602131.png) | ![image-20190219020621552](/Users/kakao/Documents/notes/Machine Learning/assets/image-20190219020621552.png) |

Potential function들을 다음과 같이 정의하자.
$$
\begin{align}
\psi(a, b) & = P(a \vert b) \\
\psi(b, c) & = P(b \vert c)P(c) \\
\phi(b) & = 1
\end{align}
$$
위의 식을 참고하여 아래 두가지 예제를 풀어보자.

**Example 1**: $P(b) = $ ?

Bayesian network에서 $P(b)$ 같은 형태 보다는 $P(b \vert  c)$ 같은 조건부 확률 형태의 테이블이 주어진다. $P(b)$ 에 대해서 찾아보자. 아래와 같이 시작하자.
$$
\phi^*(b) = \sum_a \psi(a, b) = \sum_a P(a \vert b) = 1
$$
Belief를 클리크 $BC$ 쪽으로 전파해보자. 
$$
\psi^*(b, c) = \psi(b, c)\frac{\phi^*(b)}{\phi(b)} = P(b \vert c)P(c)\frac{1}{1} = P(b \vert c)P(c) = P(b, c)
$$
클리크 $BC$ 는 결합 확률로 표현 할수 있게 되었다. 하지만 클리크 $AB$ 는 아직 조건부 확률로 표현되고 있으니 local structure가 맞게 되어 있지 않다. Belief propagation을 통해 local consistency를 만들어주자.
$$
\phi^{**}(a, b) = \sum_c \psi(b, c) = \sum_c P(b, c) = p(b)
$$
클리스 $AB$ 를 업데이트 하자.
$$
\psi^*(a, b) = \psi(a, b) \frac{\phi^{**}(b)}{\phi^*(b)}  =  P(a \vert b)\frac{P(b)}{1} = P(a, b)
$$
시작은 conditional probability table에서 시작했다. Belief propagation을 몇번 하고 나니까 클리크 $AB$ 와 $BC$ 가 결합확률 형태의 로컬화되어졌다. 다시 반대로 belief propagation을 한번더 진행 해보자.
$$
\phi^{***}(b) = \sum_a \psi^*(a, b) = \sum_a P(a, b) = P(b)
$$
$\phi^{**}(b)$ 와 $\phi^{***}(b)$ 가 같은 것을 볼 수 있다. Belief가 전체 를 순환 하니 local consistency 지켜지는 것을 확인 할 수 있다.



다음은 관측(evidence) 이 있는 상황에서 우리가 알지는 못하지만 관심 있는 hidden variable에 대해 알아보자.

**Example 2**: $P(b \vert a = 1, b=1) = $ ?
$$
\begin{align}
& \phi^*(b) = \sum_a \psi(a, b)\delta(a = 1) = P(a = 1 \vert b) \\
& \psi^*(b, c) = \psi(b, c)\frac{\phi^*(b)}{\phi(b)} = P(b\vert c=1)P(c=1)P(a=1 \vert b)  \\
& \phi^{**}(b) = \sum_c \psi^*(b, c)\delta(c=1) =  P(b\vert c=1)P(c=1)P(a=1 \vert b) \\
& \psi^*(a, b) = \psi(a, b)\frac{\phi^{**}(b)}{\phi^*(b)} = P(a = 1 \vert b)\frac{P(b \vert c=1)P(c=1)P(a=1 \vert b)}{P(a = 1 \vert b)} = P(b \vert c=1)P(c=1)P(a=1 \vert b) \\
& \phi^{***}(b) = \sum_a \psi^*(a, b)\delta(a = 1) = P(b \vert c=1)P(c = 1) P(a = 1 \vert b)
\end{align}
$$
 마지막에 구해진 $P(b \vert c=1)P(c = 1) P(a = 1 \vert b)$ 는 조건부 확률 테이블을 사용해서 구할 수있다.













