# CS 182 Lecture 4: Optimization

The loss surface:

Negative likelihood loss for logistic regression is guaranteed to be convex.

- local optima
  - surprisingly, this becomes less of an issue as the number of parameters increases!
  - for big networks, local optima exist, but tend to be not much worse than global optima
- plateaus
  - a large region where gradients are small
  - can't choose tiny learning rates to prevent oscillation
  - need learning rates to be large enough not to get stuck in a plateau (momentum)
- saddles points
  - gradient is zero, but local minimum along some dimensions, local maximum along some dimensions 
  - in fact, most critical points in neural net loss landscapes are saddle points
  - critical points $\grad_{\theta} L(\theta) = 0$
  - maximum, minimum, or saddle?
  - low dimension: second derivative is positive => local minimum
  - high dimension: second partial derivative, Hessian matrix



## Newton's method

Taylor expansion $f(x) \approx f(x_0)(x-x_0) + \frac{1}{2}f^{''}(x_0)(x-x_0)^2$

$\theta^{*} \leftarrow \theta - (\grad_{\theta}^2L(\theta_0))^{-1} \grad_{\theta}L(\theta_0)$ 

runtime: $\mathscr O(n^3)$



## Momentum

update rule:

$\theta_{k+1} = \theta_{k} - \alpha v_{k}$

$v_k = \grad_{\theta}L(\theta_k) + \mu v_{k-1}$ "blend in" previous direction



"Nesterov accelerated gradient":

If the momentum term points in the wrong direction or overshoots, the gradient can still "go back" and correct it in the same update step.

https://ruder.io/optimizing-gradient-descent/



## Algorithm: RMSProp

Estimate per-dimension magnitude (scale)

$s \leftarrow \beta s_{k-1} + (1-\beta)(\grad_{\theta}L(\theta_k))^2$  the squared length of each dimension, moving avg

$\theta_{k+1} = \theta_{k} - \alpha \dfrac{\grad_{\theta}L(\theta_k)}{\sqrt{s_k}}$                  each dimension is divided by its magnitude



## Algorithm: AdaGrad

$s_k \leftarrow s_{k-1} + (\grad_{\theta}L(\theta_k))^2$

$\theta_{k+1} = \theta_{k} - \alpha \dfrac{\grad_{\theta}L(\theta_k)}{\sqrt{s_k}}$

AdaGrad vs RMSProp: 

- AdaGrad has some appealing guarantees for **convex** problems. Learning rate effectively decreases over time, but this only work if we find the optimum quickly before the rate decays too much.
- RMSprop tends to be much better for deep learning (and most non-convex problems)



## Algorithm: Adam

$m_k = (1-\beta_1)\grad_{\theta}L(\theta_k) + \beta_1 m_{k-1}$        first moment estimate

$v_k = (1-\beta_2)(\grad_{\theta}L(\theta_k))^2 + \beta_2 v_{k-1}$      second moment estimate

$\hat{m_k} = \dfrac{m_k}{1-\beta_1^k}$

$\hat{v_k} = \dfrac{v_k}{1-\beta_2^k}$

$\theta_{k+1} = \theta_{k} - \alpha \dfrac{\hat{m_k}}{\sqrt{\hat{v_k}+\epsilon}}, \epsilon=10^{-8}, \alpha=0.001, \beta_1=0.9, \beta_2=0.999$



## Stochastic optimization

gradient descent is expensive, $L(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log p_{\theta}(y_i|x_i) \approx -E_{p_{\text{data}}(x,y)} [p_{\theta}(y_i|x_i)]$

SGD with minibatches:

1. Sample $\mathscr B \subset \mathscr D$
2. Estimate $g_k \leftarrow -\grad_{\theta} \frac{1}{B} \sum_{i=1}^{B} \log p_{\theta}(y_i | x_i) \approx \grad_{\theta}L(\theta)$
3. $\theta_{k+1} \leftarrow \theta_{k} - \alpha g_k$

In practice, sampling randomly is slow due to random memory access. Instead, shuffle the data set once in advance, then just construct batches out of consecutive groups of B data points.



SGD + learning rate decay/Adam









