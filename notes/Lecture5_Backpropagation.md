# CS 182: Lecture 5 Backpropagation

High-dimensional chain rule

$x \xrightarrow f y\xrightarrow g z$

$\dfrac{d}{dx}f(g(x)) = \dfrac{dz}{dx} = \dfrac{dy}{dx} \dfrac{dz}{dy}$

$(\dfrac{dy}{dx})_{i,j} = \dfrac{d y_j}{d x_i}$



Linear layer: $z = Wa+b$

$\dfrac{dz}{da}\delta = W^T \delta$

$\dfrac{dL}{dW} = \delta a^T$

$\dfrac{dz}{db} \delta = I \delta = \delta$

