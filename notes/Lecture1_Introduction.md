# CS 182 Lecture 1: Introduction

What is machine learning? 

- A function is a set of **rules** for transforming inputs into outputs. 
- Sometimes we can define the rules by hand - this is called programming.
- Key idea: if the rules that describe how inputs map to outputs are complex and full of special cases & exceptions, it is easier to provide data or examples than to implement those rules

"Shallow" Learning: $f_{\theta}(x) = y$ 

- fixed function for extracting features from $x$
- $\phi(x)^T \theta \leq 0$

From shallow learning to deep learning

- Learn the parameters in $\phi$
- Multiple layers of representation. higher level representations are:
  - more abstract
  - more invariant to nuisances
  - easier for predicting label



The underlying themes:

- Acquire representations by using high-capacity models and lots of data, without requiring manual engineering of features or representations
  - Model capacity: how many different functions a particular model class can represent
- Learning vs inductive bias
  - Built-in knowledge or biases in a model designed to help it learned
- Algorithms that scale
  - Get better and better as we add more data, representational capacity, and compute




