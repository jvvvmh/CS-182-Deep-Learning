# CS 182 Lecture 13: NLP

how do we learn embeddings?

idea: the more interchangeable words are, the more similar

formally: predict the neighbors of a word from its embedding value

 $\arg\max \sum_{c,o}{p(o|c)}$  --- context word: o; center word c.

$p(o|c) = \dfrac{\exp (u_o^T v_c)}{\sum_{w} \exp(u_w^T v_c)}$



another idea, making word2vec tractable: p(o is the right word|c) = $\sigma (u_o^T v_c)$

this is not enough! only true lable.

p(o is the wrong word|c) = $\sigma (-u_o^T v_c)$ randomly chosen "negatives"



## Contextual representations

word embeddings: the vector does not change if the word is used in diff ways

idea: train a language model, run it on a sentence, use its hidden state

### ELMo

- train separately forward language model & backward LM
- predict the next (or previous) word

ELMO = [h_fwd, h_bwd]

This provides a context specific and semantically meaningful representation of each token.







