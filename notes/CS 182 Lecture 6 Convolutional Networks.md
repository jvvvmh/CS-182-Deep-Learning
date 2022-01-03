# CS 182 Lecture 6: Convolutional Networks

## Convolutional layer in equations

$a^{(1)} \rightarrow z^{(2)}, W^{(2)}$

$a^{(1)}: H_{\text{in}} \times W_{\text{out}} \times C_{\text{in}}$

$z^{(2)}: H_{\text{out}} \times W_{\text{out}} \times C_{\text{out}}$

$W^{(2)}: H_{\text{F}} \times W_{\text{F}} \times C_{\text{out}} \times C_{\text{in}}$

$z^{(2)}[i,j,k] = \sum_{l=0}^{H_{\text{F}}-1} \sum_{m=0}^{W_{\text{F}}-1} \sum_{n=0}^{C_{\text{in}}-1} W^{(2)}[l,m,k,n] a^{(1)}[i+l-(H_{\text{F}}-1)/2, j+m-(W_{\text{F}}-1)/2, n]$

$z^{(2)}[i,j] = \sum_{l=0}^{H_{\text{F}}-1} \sum_{m=0}^{W_{\text{F}}-1}  W^{(2)}[l,m] a^{(1)}[i+l-(H_{\text{F}}-1)/2, j+m-(W_{\text{F}}-1)/2]$

$a^{(2)}[i,j,k] = \sigma( z^{(2)}[i,j,k] )$



https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

Pytorch: [Batch size, Channel, Height, Width]



## Padding and edges

Quiz: input is 32x32x3, filter is 5x5x6, what is the output in this case?

radius is $(H_{\text{F}} - 1)/2=2$ on each side

$H_{\text{out}} = H_{\text{in}} - (H_{\text{F}} - 1)/2 \times 2 = 32 - (5-1) = 28$

Option 2: zero pad

**Detail: remember to subtract the image mean first!**



## Strided convolutions

standard conv net structure at each layer:

1. Apply conv, $H \times W \times C_{\text{in}} \rightarrow H \times W \times C_{\text{out}}$
2. Apply activation function  $H \times W \times C_{\text{out}} \rightarrow H \times W \times C_{\text{out}}$
3. Apply pooling (width $N$) $H \times W \times C_{\text{out}} \rightarrow H/N \times W/N \times C_{\text{out}}$

expensive computationally, $C_{\text{out}} \times C_{\text{in}}$ matrix multiply at each position in $H\times W$ image.

Idea: What if skip over some positions?

Amount of skipping is called the **stride**.

Some people think that strided convolutions are just as good as conv + pooling



$(W - F  + 2P) / S +1$ 

(width - filter + 2 padding) / stride + 1



## AlexNet 2012 - a model of art

- first neural network to attain state-of-the-art results on the ImageNet large-scale visual recognition challenge (ILSVRC) 
- 1.5 million images 1000 categories
- 3 x 224 x 224 $\rightarrow$ Conv1: 11x11x96 Stride 4 without zero padding
- quiz: how many parameters in Conv1?
  - Weights: 11x11x3x96
  - Biases: 96
  - Total: 34,944
- Pool1: 3x3x96, Stride 2
- Norm1: local normalization layer
- Conv2: 5x5x256, Stride 1, **with** zero padding
- ....
  - filter does down, depth goes up
  - the first layer has a big image, losing a few pixels off the sides was not a big deal. but by the time we have 13x13, would potentially miss out useful information



## VGG - model of engineering, 19 layers

- motif repeated multiple times as a way to increase the **depth** of your model
- "homogeneous" stacks of multiple convolutions interspersed with resolution reduction



## ResNet, 152 layers

- As the number of layers increases, accuracy continues to improve
- **Main idea:** $x \rightarrow \text{weight layer} \rightarrow \text{relu} \rightarrow \text{weight layer} \xrightarrow{+x} =H(x) \rightarrow \text{relu}$
  - $H(x) = F(x) + x$
- Why good?
  - Why are deep neural networks hard to train? If we multiply many many derivatives together, we get zero or infinity. We only get a reasonable answer if the numbers are all close to 1!
  - Want $J_i \approx I$
  - ReLU: $(\dfrac{df}{dz})_i = \text{Ind}(z_i \geq 0)$
  - $\dfrac{dH}{dx} = \dfrac{dF}{dx} + I$ 



