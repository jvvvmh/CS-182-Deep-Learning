# CS 182 Lecture 10: Recurrent Neural Networks

$f(x) = g(x, h(x))$ resembles the role of $W$ in the RNN

$a_t = q(a_{t-1}, x_t)$ 

intuition: want $\dfrac{dq}{d a_{t-1}} \approx 1$ (eigenvalue)

$a_t = a_{t-1} f_t + g_t$, $a_{t-1}$ is cell state, long term memory

$h_{t-1}$ RRN output 



Autoregressive models and structured prediction

idea: past outputs should impact future

problem: training/test discrepancy, say true sequences as inputs, but at  test time... distributional shift

an old trick: randomly replace true words with previous output. 

- Scheduled Sampling: at the end of training, mostly feed model's own predictions



RNN encoder and decoders

RNNs with many layers

Bidirectional models



