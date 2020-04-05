# Singular Value Decomposition

- We can write any $n \times d$ matrix $X$ (assume $n > d$) as $X = USV^T$, where 

  1. $U$: $n \times d$ and orthogonal in the columns, i.e. $U^TU = I$

  2. $S$: $d \times d$ non-negative diagonal matrix, i.e., $S_{ii} \geq 0$ and $S_{ij} = 0$ for $i \neq j$

  3. $V$: $d \times d$ and orthogonal, i.e., $V^TV = VV^T = I$ 

- From this we have the immediate equalities
  $$
  X^TX = (USV)^T(USV^T) = VS^2V^T, \quad XX^T = US^2U^T
  $$

- Assuming $S_{ii} \neq 0$ for all $i$ (i.e., "$X$ is full rank"), we also have that
  $$
  (X^TX)^{-1} = (VS^2V^T)^{-1} = VS^{-2}V^T
  $$
  Proof: Plug in and see that it satifies definition of inverse
  $$
  (X^TX)(X^TX)^{-1} = VS^2V^TVS^{-2}V^T = I
  $$
  