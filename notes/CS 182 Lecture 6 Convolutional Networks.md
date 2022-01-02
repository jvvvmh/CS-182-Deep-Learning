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



AlexNet 2012: first neural network to attain state-of-the-art results on the ImageNet large-scale visual recognition challenge (ILSVRC)

