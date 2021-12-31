# CS 182 Lecture 3:  Error Analysis

[data] => computer program => 

- classification [object probability] 
- **regression** [continuous distribution]

$\log p_{\theta}(y|x) = -\frac{1}{2}||f_{\theta}(x)-y||^2 + \text{const}$  if $\sum_{\theta}(x) = I$

$L(\theta,x,y) = -\frac{1}{2}||f_{\theta}(x)-y||^2$



Question: how does the error change for different training sets?

- Overfitting
  - The training data is fitted well
  - The true function is fitted poorly
  - The learned function looks different each time
- Underfitting
  - The training data is fitted poorly
  - The true function is fitted poorly
  - The learned function looks similar, even if we pool together all the datasets!



What is the expected error, given a distribution over datasets?

$E_{\mathscr D \sim p(\mathscr D)}[|| f_{\mathscr D}(x) - f(x)||^2]$



## Bias-variance trade off

Let $\bar f(x) = E_{\mathscr D \sim p(\mathscr D)}[f_{\mathscr D}(x)]\\$
$$
\begin{align}
& E_{\mathscr D \sim p(\mathscr D)}[|| f_{\mathscr D}(x) - f(x) ||^2]\\
& = E_{\mathscr D \sim p(\mathscr D)}[|| f_{\mathscr D}(x) - \bar f(x) + \bar f(x) - f(x) ||^2]\\
& = E_{\mathscr D \sim p(\mathscr D)}[|| f_{\mathscr D}(x) - \bar f(x)||^2]  + ||\bar f(x) - f(x) ||^2\\
& = \text{Variance + Bias}^2
\end{align}
$$
high variance => overfitting

bias => underfitting



## How to regulate bias/variance?

- Get more data, addresses variance, has no effect on bias
- Change your model class
- Can we "smoothly" restrict the model class? Can we construct a "continuous knob" for complexity



## Regularization

Regularization: something we add to the loss function to reduce variance

### Bayesian interpretation

Regularization could be regarded as a prior on parameters.

High level intuition:

- When we have high variance, it's because the data doesn't give enough information to identify parameters
- If there is not enough information in the data, can we give more information through the loss function

Question: Given $\mathscr D$, what is the most likely $\theta$?

$p(\theta |\mathscr D) = \dfrac{p(\theta, \mathscr D)}{p(\mathscr D)}\propto p(\mathscr D | \theta) p(\theta)$

What kind of distribution assigns higher probabilities to small numbers?

Simple idea: $p(\theta) = N(0, \sigma^2)$

New loss function:
$$
\begin{aligned}
L(\theta)&= - (\sum_{i=1}^{N}\log p(y_i | x_i)) - \log p(\theta)\\
&= (\sum_{i=1}^{N}||f_{\theta}(x_i) - y_i||^2) + \lambda ||\theta||^2
\end{aligned}
$$
(weight decay)



Other examples of regularizers:

- $\lambda \sum_{i=1}^{D}|\theta_i|$
  - L1 regularization
  - creates a preference for zeroing out dimensions!
- $\lambda \sum_{i=1}^{D} \theta_i^2$
- Dropout
  - a special regularizer for NNs
- Gradient penalty
  - a special regularizer for GANs

Other perspectives:

1. Baysian perspective: the regularizer is prior knowledge about parameters
2. Numerical perspective: the regularizer makes undetermined problems determined
3. Optimization perspective: the regularizaer makes the loss landscape easier to serach, paradoxically, regularizers can sometimes reduce underfitting if ti was due to poor optimization! especially common with GANs



## The machine learning workflow

training set || validation set

$L(\theta, \mathscr D_{\text{train}})$ is the empirical risk.

$L(\theta, \mathscr D_{\text{val}})$ is an unbiased estimator for the true risk.

- Train $\theta$ with $L(\theta, \mathscr D_{\text{train}})$ 

  if $L(\theta, \mathscr D_{\text{train}})$ not low enough

  ​    you are underfitting

  ​    decrease regularization

  ​    improve your optimizer

- Look at $L(\theta, \mathscr D_{\text{val}})$ 

  if $L(\theta, \mathscr D_{\text{val}})$ >> $L(\theta, \mathscr D_{\text{train}})$

  ​        you are overfitting

  ​        increase regularization



Training set is used to select

- $\theta$ (via optimization)
- optimizer hyperparameters (learning rate $\alpha$)

Validation set is used to select

- model class
- regularizer hyperparameters
- which features to use



Learning curves

- Overfitting: validation loss starts going up after a while
  - when to stop? right before the validation loss goes up?
  - does early stop mitigate all overfitting?
    - even if you stop the optimizer right at that sweet spot, the validation error might never get as low as it could if you use regularizers.
- Underftting: a gap in the end is the bias



How good is our final classifier?

- $L(\theta, \mathscr D_{\text{val}})$ 
  - That's no good - we already used the validation set to pick hyperparameters, not unbiased anymore
- test set
  - used only to report final performance



Summary

- Where do errors come from?
  - Variance: too much capacity, not enough information in the data to find the right parameters
  - Bias: too little capacity, not enough representational power to represent the true function
  - Error = Variance + Bias^2
  - Overfitting = too much variance
  - Underfitting = too much bias
- How can we trade off bias and variance?
  - Select your model class carefully
  - Select your features carefully
  - Regularization: stuff we add to the loss to reduce variance
- How do we select hyperparameters?
  - Training/validation split
  - Training set is for optimization (learning)
  - Validation set is for selecting hyperparameters
  - Test set if for reporting final results and nothing else



 

