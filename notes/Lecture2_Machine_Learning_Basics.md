# CS 182 Lecture 2: Machine Learning Basics

Different types of learning problems:

1. Supervised learning
   - Given $\mathscr{D} = \{(x_1, y_1),...,(x_n,y_n)\}$, learn $f_{\theta}(x) \approx y$
2. Unsupervised learning
   - Generative modeling: GANs, VAEs, pixel RNN
   - Self-supervised representation learning: BERT
3. Reinforcement learning
   - Choose $f_{\theta}(s_t) = a_t$ to maximize $\sum_{t=1}^{H} r(s_t,a_t)$
   - actually subsumes supervised learning (one time step)



The machine learning method

1. Define your model class
2. Define your loss function
3. Pick your optimizer
4. Run it on a big GPU



Why is it called a softmax?

$P(\text{red}|x) = \dfrac{e^{\theta_{\text{red}}^T x}}{e^{\theta_{\text{red}}^T x}+e^{\theta_{\text{blue}}^T x}}$ When scaling $\theta$ to $1000\theta$, it looks like a "MAX" function, assigning a high probability to the label with the largest dot product.



How is the dataset "generated"?

$\mathscr{D} = \{(x_1, y_1),...,(x_n,y_n)\}$

Key assumption: i.i.d. 

$p(\mathscr{D}) = \prod_i p(x_i,y_i) = \prod_i p(x_i) p(y_i | x_i)$, we are learning $p_{\theta}(y | x)$

Idea: choose $\theta$ such that $p(\mathscr D) = \prod_i p(x_i) p_{\theta}(y_i | x_i)$ is maximized. But the problem is you are multiplying together many numbers $\leq 1$.
$$
\log p(\mathscr D) = \sum_{i} \log p(x_i) + \sum_{i}\log p_{\theta}(y_i|x_i)\\
\theta ^* = \arg \min_{\theta} -\sum_{i} \log p_{\theta}(y_i | x_i)
$$
Loss functions

- negative log-likelihood (cross entropy): $-\sum_i \log p_{\theta}(y_i | x_i)$
- zero-one loss: $-\sum_i \delta( f_{\theta}(x_i)\neq y_i)$
- mean squared error: $\sum_i || f_{\theta}(x_i)-y_i  ||^2$
  - actually just negative log-likelihood of multi normal



Gradient descent

1. Compute $-\grad_{\theta} L(\theta)$
2. $\theta \leftarrow \theta - \alpha \grad_{\theta} L(\theta)$



Logistic regression

$f_{\theta}(x) = x^T \theta$

$\theta = \left[ \begin{matrix} \theta_{y_1}, \theta_{y_2}, \theta_{y_3}\end{matrix}\right]$

$p_{\theta}(y=i | x) = \text{softmax}(f_{\theta}(x)) [i] = \dfrac{\exp f_{\theta,i}(x)}{\sum_{j=1}^{m}\exp f_{\theta,j}(x)}$



Special case: binary classification

$P(y_1|x) = \dfrac{e^{\theta_{y_1}^T x}}{e^{\theta_{y_1}^T x}+e^{\theta_{y_2}^T x}} = \dfrac{1}{1 + e^{-\theta_{+}^T x}}, \theta_{+}=\theta_{y_1} - \theta_{y_2}$ (sigmoid)



Empirical risk and true risk

- Empirical risk $=\frac{1}{n}\sum_{i=1}^{n} L(x_i, y_i, \theta)$
- True risk $=E_{x \sim p(x), y \sim p(y|x)} [L(x,y,\theta)]$

Overfitting: empirical risk is low, but the true risk is high

- can happen if the dataset is too small
- can happen if the model is too powerful (too many parameters/capacity)

Underfitting: empirical risk is high, and the true risk is high

- can happen if the model is too weak (too few parameters/capacity)
- can happen if your optimizer is not configured well (e.g. wrong learning rate)



Summary

1. Define your model class
2. Define your loss function
3. Pick your optimizer
4. Run it on a big GPU

