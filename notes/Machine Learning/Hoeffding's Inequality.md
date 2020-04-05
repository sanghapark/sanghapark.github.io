

# Hoeffding's Inequality

어떤 확률 모형의 모수의 참된 값은 $\theta^* $라고 하자. 추정된 모수 $\hat{\theta}$와 참된 모수 $\theta^*$의 차이가 어떤 에러 $\varepsilon$보다 클 확률에는 상한성이 있다. 즉,
$$
P(|\hat{\theta} - \theta^*| \geq \varepsilon) \le 2e^{-2N\varepsilon^2}
$$

$\varepsilon$가 크거나 N이 커지면 우항은 더 작아진다. 추정된 모수와 진짜 모수 사이의 차이가 주어진 에러보다 더 클 확률은 작아진다. $\varepsilon = 0.1$이고 추정된 모수와 참된 모수의 차이가 $\varepsilon$ 보다 클 확률을 0.01%으로 제한한다면 필요되는 $N$을 구할 수 있을까? 위에 식을 역산 하여 풀수 있다.

