# CS 182 Lecture 12: Transformers

Attention can access **every** time step. Do we need recurrent connections?

Issue 1: now step $l=2$ can't access previous states of the decoder $s_1, s_0$

Fix 1: Self-Attention

$x_1$

$h_1$

$k_1, q_1, v_1$

$q_1 k_1, q_2 k_2, q_3 k_3$  softmax-> $\alpha_{1,1}, \alpha_{1,2}, \alpha_{1,3}$

weighted: $\sum_{i} \alpha_{1,i} v_i$



x -> self-attention layer -> a -> self-attention layer -> ... decode



## Positional encoding

Address lack of sequence information; otherwise, change the order, transformer gives the same answer

$h_t = f(x_t, t)$ 

- naive positional encoding: $x_t, t$, 
  - but absolute position is less important than relative position
  - i walk my dog every day VS every single day i walk my dog
  - "my dog" is right after "i walk" is the important part, not its absolute position.

- freqeuncy-based $p_t = [\sin(t / 10000^{2 * 1/d}), \cos(t / 10000^{2 * 1 /d}),..., \sin(t / 10000^{2 * \frac{d}{2}/d}), \cos(t / 10000^{2 * \frac{d}{2} /d})]$

  \ \ \ \ \ \ 

  \  \  \  \  \

  \     \     \    \

  \         \           \

  ------index in the sequence------------------------------

  "even-odd" indicator; "first-half / second-half" indicator

- learn a positional encoding $p_1, p_2, ...$ $d\times T$ values to learn

  - need to pick a max length

$\text{emb}(x_t) + p_t$ or concatenate them



## Multi-headed attention

(类比cnn, multi filters)

allows query multiple positions at each layer

$q_l k_t$ cannot focus two things e.g. get verb and noun in one time

multi K Q V independently give $a$s and stack $a$ (like CNN multi filters)

8 heads work for big models.



## Adding nonlinearities

so far, each successive layer is linear in the previous one in the exceptions of softmax.

k, q, v are linear to h

e = q k

$a_l = \sum_t \alpha_{l,t} v_t = W_v \sum_t \alpha_{l,t} h_t$  is linear in $h$ (nonlinear weights)

"if ... and ... then..." is hard to do.

so: self-attention layer -> $a_1$ -> nonlinear function $h_t^2 = \sigma(W^{2} a_1^{l} + b^{l})$



## Masked decoding

how to prevent attention lookups into the future? (language decode)

at **test time when decoding**, the inputs at step 2&3 will be based on the output at step 1...

... which requires knowing the input at step 2&3

Easy solution: $e_{l,t} = q_l k_t$   if $l < t$, then set to $- \inf$, cannot see the future

in practice: replace $\exp(e_{l,t})$ with 0 if $l < t$ inside the softmax



## Layer Normalization

batch norm is hard to use with sequence models

- sequences are of different lengths
- sequences can be very long, so sometimes have small batches

Layer normalization: across different dimensions of $a_t$

Add & Norm: a = LayerNorm(h + a) 

![image-20220201215638172](C:\Users\jvmh\AppData\Roaming\Typora\typora-user-images\image-20220201215638172.png)



Transformer $O(n^2)$ vs $O(n)$

benefits:

(1) much better long-range connections

(2) much easier to parallelize

(3) can make ti deep (more layers) than RNNs (3->6)

