# K-Means Clustering

## Training Algorithm

- Input:  $N \times D$ matrix $X$

- $N$ samples

- $D$ features (dimensionality)

  

수렴 할 때까지 다음 두가지 스텝을 반복한다.

- 초기화: K개의 임의의 포인트들을 선택해서 클러스터 중심점으로 간주한다.

- 수렴 할 때까지 (수렴이 되었는지 안되었는지 판다하는 부분이 objective function):
  1. 각 포인트를 가장 가까운 클러스터 중심점과 같은 레이블로 만든다.

  2. 각 클러스터에 속하는 포인트들로 중심점을 다시 계산한다. (중심점은 각 클러스터에 해당하는 포인트들의 각 축의 평균)



## One problem with K-Means

- Highly sensitive to initialization

- Possible resolution: restart multiple times, use whichever result gives us the best final objective

  - Local minima problem

- Another possible resolution: **fuzzy** membership in each class

  - Just a small adjustment to the original k-means algorithm

  

### Soft K-Means

1. Inialize $m_1$,…,$m_k$ = random points in $X$

2. While not converged:

   - Step 1: Calculate cluster reponsiblities (fuzzy membership in each class):
     $$
     r_k^{(n)} = \frac{\exp[-\beta\cdot d(m_k, x^{(n)})]}{\sum_j^{K} \exp[-\beta\cdot d(m_j, x^{(n)})]}
     $$

   - Step 2: Recalculate means:
     $$
     m_k = \frac{\sum_n{r_k^{(n)}x^{(n)}}}{\sum_n r_k^{(n)}}
     $$

$d(m_k, x^{(n)})$ is a distance between a point $x^{(n)}$ and a mean $m_k$. $r$ is now always a fraction between 0 and 1. It can interpret "hard k-means" to be where $r$ is exactly 0 or 1. When $r_k^{(n)}$ is larger, it has more influence on the calculation of $m_k$



#### Responsibility

$$
r_k^{(n)} = \frac{\exp[-\beta\cdot d(m_k, x^{(n)})]}{\sum_j^{K} \exp[-\beta\cdot d(m_j, x^{(n)})]}
$$

#####Relationship to Gaussian

The numerator of resposibility looks a lot like a Gaussian PDF. We will make this connection directly later when we discuss **Gaussian Mixture Model**

- Gaussian PDF
  $$
  f(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp\bigg ({-\frac{1}{2\sigma^2}(x-\mu)^2}\bigg)
  $$

- Variance controls how fat or skinny the Gaussian PDF is. It tells how influent of each cluster is on each data point.

#### Calculating the mean

- It is called **weighted arithmetic mean**. If a data point is far away from cluster, it shouldn't influence that cluster's center as much.

  - Regular mean
    $$
    m_k = \frac{1}{N}\sum_n x^{(n)} = \frac{1\cdot x^{(1)} + 1\cdot x^{(2)}+\cdots + 1\cdot x^{(n)}}{1+1+\cdots + 1}
    $$

  - Weighted mean
    $$
    m_k = \frac{\sum_n r_k^{(n)}x^{(n)}}{\sum_n r_k^{(n)}}
        = \frac{r_k^{(1)} x^{(1)} + r_k^{(2)} x^{(2)} + \cdots + r_k^{(n)} x^{(n)}}{r_k^{(1)} + r_k^{(2)} + \cdots + r_k^{(n)}}
    $$
    

#### What is the purpose of soft K-Means?

It reflects out **confidence** in the cluster assignment which hard k-means can't.



## Objective Function

- Coordinate descent
  $$
  J = \sum_n\sum_k r_k^{(n)} \| m_k - x^{(n)} \|^2
  $$
  Mathematics guarantee that $J$ will always descrease with each iteration (but not necessarily to global minimum)

## Disadvantages of K-Means Clustering

- You have to choose $K$
  - We can look at 2-D or 3-D data to help up choose. What about 100-D data?
- Local Minima
  - Not necessarily bad for deep learning. But in K-Means it is bad.
  - Restart multiple times to try different initial starting points for centroids for clutsters.
- Sensitive to initial configuration
- Can't solve donut problem.
  - Can't even solve elliptical problem
  - Can only look for spherical clusters
- Does't take into account density (?)
- 































