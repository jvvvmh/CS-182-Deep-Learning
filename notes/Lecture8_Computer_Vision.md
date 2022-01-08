# CS 182 Lecture 8: Computer Vision

Problems

- object classification
- object localization
- object detection
  - multi objects
- semantic segmentation a.k.a. scene understanding
  - detect shape



## Object Localization

$(x_i, y_i), y_i=(l_i, x_i, y_i, w_i, h_i)$

Intersection over Union (**loU** = Inersection / Union Area)

correct if loU > 0.5

loss?

1. regression (predict label & location)
2. sliding windows: classify every patch in the image

**A practical approach: OverFeat**

- Pretrain on just classification
- Train regression head on top of classification features
  - provides a little "correction" to sliding window
- Pass over different regions at different scales
- "Average" together the boxes to get a single answer
- too expensive
  - improvement: in the last conv, use 1x1 conv (for each position)



### Object Detection

"You Only Look Once"

- bounding boxes + confidence
- classification map probability



CNNs + Region proposals



## Semantic segmentation

want to label each pixel with its label

stupid method: slide 1, never downsampling

**up-sampling/transpose conv**, stride = 1/2

**un-pooling** 

- may loose spacial details
- U-net: upsample + original layer that has this resolution











