# Losses Used in Segmentation Task

Image Segmentation can be defined as classification task on pixel level. An image consists of various pixels, and these 
pixels grouped together define different elements in image. A method of classifying these pixels into elements is called
semantic segmentation. 

The choice of loss/objective function is extremely important. In the paper, they summarized 15 segmentation based loss 
functions that have been proven to provide state of the art results in different domains. 

Table of loss functions: 

| Type      | Loss Function |
| ----------- | ----------- |
| **Distribution-based Loss**      |  Binary Cross-Entropy       |
| .   | Weighted Cross-Entropy        |
| .   | Balanced Cross-Entropy        |
| .   | Focal Loss        |
| .   | Distance map derived loss penalty term        |
| **Region-based Loss**  |  Dice Loss   |
| .  |  Sensitivity-Specificity Loss   |
| .  |  Tversky Loss   |
| .  |  Focal Tversky Loss   |
| .  |  Log-Cosh Dice Loss   |
| .  |  Log-Cosh Dice Loss   |
| **Boundary-based Loss**  |  Hausdorff Distance loss   |
| .  |  Shape aware loss   |
| **Compounded Loss**  |  Combo Loss   |
| .  |  Exponential Logarithmic Loss   |

Optimizer is used to optimize and learn the Objective. To learn an objective accurately and faster, we need to ensure 
that the mathematical representation of objectives (aka loss function) are able to cover even the edge cases. 

In the paper, the author focused on Semantic Segmentation instead of Instance Segmentation, so the number of classes at
pixel level is restricted to 2.

## Binary Cross-Entropy

Cross-entropy is defined as a measure of the difference between two **probability distributions** for a given random 
variable or set of events. 

**Usage**: It is used for classification objective, and as segmentation is pixel level classification it works well.

Binary Cross-Entropy (BCE) is defined as: 

> ![](https://latex.codecogs.com/svg.image?L_{BCE}(y,&space;\hat{y})&space;=&space;-(ylog(\hat{y})&space;&plus;&space;(1&space;-&space;y)log(1&space;-&space;\hat{y})))

In this case, we just have 2 classes. If more classes, then the fomula become the sum of more terms, and the values 
inside log is result of **softmax** - which apply on tensor instead of **sigmoid** - which apply on a scalar.

Pytorch has the BCELoss in their built-in function.
Read more at: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss

**Notice: PyTorch use base e for the log function.** 

**Multi-class case:**

> ![](https://latex.codecogs.com/svg.image?L_{CE}&space;=&space;-&space;\frac{1}{N}&space;\sum_{c=1}^{N}\sum_{l=1}^{L}r_l^clog(p_l^c))

N: number of pixels need to classify in a minibatch

c : Notation for pixel

l: Notation for label, L is number of class we need to classify

$p^c$: Notation for **probability** vector of the predicted given by output of model. (Usually we use softmax after the 
output of model to get this) 

$r^c$: Notation for one hot encoded vector, where 1 stand for class it belong and others are 0.

The relation between $l^c$ and $p^c$ when use model to predict:

> ![](https://latex.codecogs.com/svg.image?l^c&space;=&space;arg&space;\underset{l}{max}&space;({p_l^c})&space;&space;)

Ok, we move to the next term.

## Weighted Binary Cross-Entropy (WCE)

It is the variance of binary cross entropy. It is widely used in case of skewed data (the number of instance in each 
class is imbalance):

> ![](https://latex.codecogs.com/svg.image?L_{W-BCE}(y,&space;\hat{y})&space;=&space;-(\beta&space;*&space;ylog(\hat{y})&space;&plus;&space;(1&space;-&space;y)log(1&space;-&space;\hat{y})))

**Multi-class case:**

The tendency to under-estimate can be mitigated by assigning higher weights to loss contributions from pixels with under
represented class labels (instance less then weight class hight)

class_weight computed in sklearn equivalent to term 1/w_c in the above equaltion:

> ![](https://latex.codecogs.com/svg.image?L_{WCE}&space;=&space;-&space;\frac{1}{N}&space;\sum_{c=1}^{N}&space;\frac{1}{w_c}\sum_{l=1}^{L}r_l^c&space;log(p_l^c))


One way to achieve the weight is taken from the one-hot $r^c$, example:

```python
# Minibatch has size 20, we have 5 classes and in Pytorch it present by a Tensor contain index of labels instead of 
# One Hot tensor
import sklearn.utils.class_weight as class_weight
import torch
import torch.nn as nn

y = torch.randint(0, 5, (20,))
class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y.numpy())
class_weight = torch.tensor(class_weight, dtype=torch.float)
# Then pass this weight as the param for the CrossEntropyLoss, example

loss_fn = nn.CrossEntropyLoss(reduction='mean')
# For each minibatch 
# Compute the class weight by the code above, then change the weight by apply 
loss_fn.weight = torch.tensor(class_weight, dtype=torch.float)
loss = loss_fn(putput, target)
loss.backward()
optimizer.step()
```

And there are many way to achive weight map like the one that introduced in the paper: https://arxiv.org/abs/1505.04597



Note: (I quite dont understand the note inside the paper)

## Balanced Cross-Entropy (BCE)

It is similar to Weighted Cross Entropy. The only difference is that we also add weight to negative examples.

> ![](https://latex.codecogs.com/svg.image?L_{BCE}(y,&space;\hat{y})&space;=&space;-(\beta&space;*&space;ylog(\hat{y})&space;&plus;&space;(1&space;-&space;\beta)*(1&space;-&space;y)log(1&space;-&space;\hat{y})))


## Focal Loss (Implemented)

**Binary Classification Case:**

This is also be seen as variation of Binary Cross-Entropy. It down-weights the contribution of easy examples and enables
the model to focus more on learning hard examples. 


Focal Loss proposes to down-weight easy examples and focus training on hard negatives using a modulating factor:

> ![](https://latex.codecogs.com/svg.image?FL(p_t)&space;=&space;-\alpha_t(1-p_t)^{\gamma}log(p_t))

Here gamma > 0 and when gamma = 1. Focal Loss works like Cross Entropy Loss function. Similarly, alpha in range [0, 1].
It can be set by inverse class frequency or treated as a hyper-parameter.

**Multi-class Classification Case:**

-------------------------------------------

## Dice Loss (Implemented)

Dice coefficient is widely used metric in computer vision to calculate the similarity between 2 image. Later in 2016, it
has also adapted as loss function known as Dice Loss

Visualize for Dice Coefficient in set theory:

> ![](https://latex.codecogs.com/svg.image?DSC&space;=&space;\frac{2\left|&space;A&space;\cap&space;B\right|}{\left|A&space;\right|&space;&plus;&space;\left|&space;B\right|})


**Binary classification**:

> ![](https://latex.codecogs.com/svg.image?DL(y,&space;\hat{p})&space;=&space;1&space;-&space;\frac{2y\hat{p}&space;&plus;&space;1}{y&space;&plus;&space;\hat{p}&space;&plus;&space;1})

Here 1 is added in numerator and denominator to ensure that the function is not undefined in edge case scenarios such as 
when ![](https://latex.codecogs.com/svg.image?y&space;=&space;\hat{p}&space;=&space;0). 

**Multi-class task:**

This loss is introduced in V-Net (2016), called **Soft Dice Loss**: used to tackle the class imbalance without the need 
for explicit weighting (which is used in **Weighted Cross Entropy**). One possible formulation is:


## Batch Soft Dice (This is a variance of Soft Dice) (Implemented but not sure)

## Tversky Loss (Implemented)

## Focal Tversky Loss (Implemented)

## Sensitivity-Specificity Loss (Implemented)

## Log-Cosh Dice Loss (Implemented)

## Hausdorff Distance Loss (Need time to read more papers)

References:

Github: https://github.com/HaipengXiong/weighted-hausdorff-loss, Paper: https://arxiv.org/pdf/1806.07564.pdf

## Blob loss

References:

Github: https://github.com/neuronflow/blob_loss, Paper: https://arxiv.org/abs/2205.08209

## Shape aware loss

-------------------------------------------

## Combo Loss (Implemented)

## Exponential Logarithmic Loss (Implemented)

### References:
* A survey of loss functions for semantic segmentation (Shruti Jadon - 2020).
* Segmentation of Head and Neck Organs at Risk Using CNN with Batch Dice Loss (2018).
* Discussion about the class weight compute: https://stackoverflow.com/questions/61414065/pytorch-weight-in-cross-entropy-loss
* Blob loss: instance imbalance aware loss functions
for semantic segmentation: https://arxiv.org/pdf/2205.08209.pdf

### TODO
- [ ] Crop small image chunks for testing with the loss function, I need to be sure with the `Hough loss`, so I need to do that 
- [ ] Next version, base on Kornia library (https://github.com/kornia/kornia), I implememted the stable version that can apply to higher dimensional Tensor,
that'll look like what the loss functions in Pytorch does.
- [ ] Read papers about the rest loss functions and try hard to implement it.
- [ ] Make a table to easy compare between them, when use these functions. 
- [ ] Take some of these functions into training process and test the current model and see how they improve the prediction performance.