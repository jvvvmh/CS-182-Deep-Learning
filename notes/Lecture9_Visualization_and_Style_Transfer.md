# CS 182 Lecture 9: Visualization and Style Transfer

what do we visualize?

1. visualize the filter itself (e.g. first conv, detect edges)

2. visualize the stimuli that activate a "neuron": what image patch makes the output of the filter large?

3. using gradients for visualization

   - given a pixel $x_{ij}$ and a unit $a^{l}_{mnp}$

   - how much does changing $x_{ij}$ change $a^{l}_{mnp}$

   - TORCH.AUTOGRAD.BACKWARD(tensors=0,...,1...,0 在第mnp这个位置为1)

   - guided backpropagation

     - only keep positive gradients
     - put a ReLU on the backward pass

   - visualizing "classes"

     - maximize the thing before softmax, need $R(x)=\lambda ||x||^2$

     - R(x) "Understanding NN Through Deep Visualization"

       - update image with gradient

       - blur the image

       - zero out pixel with small value

       - repeat

         

## Deep Dream & Style Transfer

Deep Dream

- pick a layer
- run forward pass to compute activations at that layer

- set dx=x
- jitter regularizer
- image update (backprop and apply the gradient)



Another idea.

## Gram matrix

How do we quantify style?

style: relationships between features

feature covariance: $\text{Cov}_{km} = E[f_k f_m]$

Gram matrix: $G_{km} = \text{Cov}_{km}$

matrix size: #channels x #channels

$x \leftarrow \arg \min_x (L_{\text{style}}(x)+L_{\text{content}}(x))$

$G^l:$ source image Gram matrix at layer $l$ (style)

$A^{l}$: new image Gram matrix at layer $l$

$L_{\text{style}}(x) = \sum_{l} \sum_{km} (G_{km}^l - A_{km}^l(x))^2 w_l$



content: spatial positions of features

$L_{\text{content}}(x)=\sum_{ij}\sum_{k} (f_{ijk}^{l}(x_{\text{content}})-f_{ijk}^{l}(x))^2$  Pick a layer $l$ to represent the positions

