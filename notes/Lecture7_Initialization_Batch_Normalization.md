# CS 182 Lecture 7: Initialization, Batch Normalization

## Standardization: $\mu=0, \sigma=1$

- Inputs

- Activations
  - **batch normalization**: $\mu$ and $\sigma$ are calculated over the batch
  - in practice, $\bar{a}_i^{(1)} = \dfrac{a_i^{(1)}-\mu^{(1)}}{\sigma^{(1)}}\gamma + \beta$, learnable scale and bias
  - often we can use a larger learning rate with batch norm
  - train much faster
  - requires less regularization (doesn't need dropout)
  - train time: compute $\mu$ and $\sigma$ and freeze them, test time: directly use them



## Weight Normalization

Basic initialization: 

- $W_{jk}^{(i)} \sim N(0,0.0001)$
  - Goal is to have well-behaved gradients & activations
  - Why is this bad? $\dfrac{L}{W^{(i)}} = \dfrac{z^{(i)}}{dW^{(i)}}\dfrac{dL}{dz^{(i)}}=\delta {a^{(i-1)}}^T$ $a$ has mean 0 and small std in later layers
  - Place is in the middle of a really big plateau
- Xavier initialization
  - $z_i = \sum_{j} W_{ij}a_j + b_i, b_i\approx 0$
  - $E[z_i^2] = \sum_{j} E[W_{ij}^2]E[a_j^2] = D_a \sigma_{W}^2 \sigma_a^2$
  - choose $\sigma_W^2 = 1 / D_a$
  - problem: $a_j = \text{ReLU}(z_j)$ moves the negative half
  - basic principle: get std of $W_{ij}$ to be about $1 / \sqrt{\frac{1}{2}D_a}$
- ReLUs & biases, $b_i=0.1$, otherwise half of our units will be "dead" if $b_i\approx 0$



## Advanced Initialization

$W = U \Lambda V$

using singular value decomposition

force $\Lambda$ to be identity matrix



## Gradient Clipping

- took a step that was too big in the wrong place

- divided by something small (in batch norm, softmax)

  

- per-element clipping $\bar g_i = \max ( \min (g_i, c_i), -c_i)$
- norm clipping: $\bar g_i = g \dfrac{\min(||g||, c)}{||g||}$

how to choose c? run a few epochs, and see the healthy gradients



## Ensembles & dropout

Problem: neutral networks have high var

Idea: there are many more ways to be wrong than to be right

Variance $=E_{\mathscr D \sim p(\mathscr D)}[||f_{\mathscr D}(x)-\bar f(x)||^2]$

### Ensemble in theory

- resampling with replacement
- average the possibilities/majority vote

### Ensemble in practice

there is a lot or randomness in NN

- Random initialization
- Random mini-batch shuffling
- Stochastic gradient descent

even faster ensembles

- task-agnostic features often the most expensive part (vs task-specific classification layers)
- share the task-agnostic features for all models in the ensemble

even fasterer ensembles

- "Snapshot Ensembles: Train 1, Get M For Free"
- save out parameter snapshots over the course of SGD, use each snapshot as a model 
- combining predictions
  - average probabilities 
  - vote
  - average the parameter vectors together

Can we make multiple models out of a single neural network?

**Dropout**

"new" network made out of the old one

$a_{ij}^{(i)} m_{ij}, m_{ij} \sim \text{Bernoulli}(0.5)$

How does it work? Forces the network to have a redundant representation.

At test time:

before: on average 1/2 of dimensions are forced to 0

now: none of them are, so $W a$ will be $\approx$ 2 x bigger

solution: $\bar{W} = \dfrac{1}{2}W$ divide all weights by 2



Hyperparameters

optimization (training)

- learning rate
- momentum
- initialization
- batch norm

generalization (validation)

- ensembling
- dropout
- architecture (# and size of layers)



How to pick? Random layout is better than grid layout. More cases for important hyperparameters with same cost.









