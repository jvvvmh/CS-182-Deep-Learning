# CS 182 Lecture 11: Sequence to Sequence

A basic neural language model

tokenize the sentence:

- simplest: one-hot vector
- more complex: word embeddings
- real models...



Q: how to complete a sentence?

A: <EOS>, <START>



A conditional language model

- a picture -> vector encoding as the initial hidden state $a_0$
- condition on another sequence. RNN encoder -> RNN decoder
  - if only one giant RNN: <START><START> as <EOS> token for input sequence
  - in reality, the input sentence is often read in reverse
  - typically two separate RNNs
  - trained end-to-end on paired data



Want to maximize the **product** of all probabilities. $M^T$ sequences for $M$ words

Decoding is a search problem.

Beam search: store the k (5-10) best sequences so far

- use soft-max for log-probs
- stop decoding? one of the highest hypotheses ends in <END>, save its score, do not expand
- sum of $\log(p)$ is a too negative number -> Score(...) e.g. take average



## Bottleneck Problem

All info about the source sequence is contained in the first activation.

Idea: "peek" at the source sentence while decoding.

a small **key** vector: a linear transformation is usually enough (encode)

**query** vector: find which **key** is the most relevant, and we will send that information to the decoder

Attention:

$k_t = k(h_t)$  encoder

$q_l = q(s_l)$   decoder

score $e_{t,l} = k_t q_l$ dot product

argmax is not differentiable, so we use softmax

$\alpha_{t,l} = \text{softmax}(e_{...,l})$

**send** $a_l = \sum_{t} \alpha _{t,l} h_l$ weighted encoder's hidden

how to use $a_l$?

1. output $\hat y_l = f(s_l, a_l)$
2. next RNN layer
3. next decoder step $\bar s_{l} = [s_{l-1}, a_{l-1}, x_{l}]$

A reasonable choice: $k_t = W_k h_t, q_l = W_q s_l$

$e_t$ = $h_t ^T W_e s_l$    only learn one matrix!!



## Attention Summary

Every encoder step $t$ produces a key $k_t$

Every decoder step $l$ produces a query $q_l$

Decoder gets "sent" encoder activation $h_t$, weighted by softmax of $k_t q_l$: $\sum_t \alpha_t h_t$

Why is attention good?

1. All decoder steps are connected to all encoder steps!
2. Length of Connection goes from O(T) to O(1). 
3. Short path of gradients. O(1) propagation length. Gradients are much better behaved
4. Becomes very important for very long sentences















