#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Nov 17 12:48:36 2021

@author: Nacriema

Refs:

I build the collection of loss that used in Segmentation Task, beside the Standard Loss provided by Pytorch, I also
implemented some loss that can be used to enhanced the training process.

For me: Loss function is computed by comparing between probabilities, so in each Loss function if we pass logit as input
then we should convert them into probability. One-hot encoding also a form of probability.

For testing purpose, we should crete ideal probability for compare them. Then I give the loss function option use soft
max or not.

May be I need to convert each function inside the forward pass to the function that take the input and target as softmax
probability, inside the forward pass we just convert the logits into it


Should use each function, because most other functions like Exponential Logarithmic Loss use the result of the defined
function above for compute.

Difference between BCELoss and CrossEntropy Loss when consider with mutiple classification (n_classes >= 3):
    - When I'm reading about the fomular of CrossEntropy Loss for multiple class case, then I see the loss just "inclue" the t*ln(p) part, but not the (1 - t)ln(1 - p)
    for the "background" class. Then it can not "capture" the properties between each class with the background, just between each class together. 
    - Then I'm reading from this thread https://github.com/ultralytics/yolov5/issues/5401, the author give me the same idea. 


Reference papers: 
    * https://arxiv.org/pdf/2006.14822.pdf

"""
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
import torch.nn as nn
import torch.nn.functional as F


def get_loss(name):
    if name is None:
        name = 'bce_logit'
    return {
        'bce': BCELoss,
        'bce_logit': BCEWithLogitsLoss,
        'cross_entropy': CrossEntropyLoss,
        'soft_dice': SoftDiceLoss,
        'bach_soft_dice': BatchSoftDice,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'log_cosh_dice': LogCoshDiceLoss,
        'sensitivity_specificity': SensitivitySpecificityLoss, 
        'exponential_logarithmic': ExponentialLogarithmicLoss,
        'combo': ComboLoss,
    }[name]


def soft_dice_loss(output, target, epsilon=1e-6):
    numerator = 2. * torch.sum(output * target, dim=(-2, -1))
    denominator = torch.sum(output + target, dim=(-2, -1))
    return (numerator + epsilon) / (denominator + epsilon)
    # return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))

# DONE
class SoftDiceLoss(nn.Module):
    def __init__(self, reduction='none', use_softmax=True):
        """
        Args:
            use_softmax: Set it to False when use the function for testing purpose
        """
        super(SoftDiceLoss, self).__init__()
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, output, target, epsilon=1e-6):
        """
        References:
        JeremyJordan's Implementation
        https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py

        Paper related to this function:
        Formula for binary segmentation case - A survey of loss functions for semantic segmentation
        https://arxiv.org/pdf/2006.14822.pdf

        Formula for multiclass segmentation cases - Segmentation of Head and Neck Organs at Risk Using CNN with Batch
        Dice Loss
        https://arxiv.org/pdf/1812.02427.pdf

        Args:
            output: Tensor shape (N, N_Class, H, W), torch.float
            target: Tensor shape (N, H, W)
            epsilon: Use this term to avoid undefined edge case

        Returns:

        """
        num_classes = output.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == one_hot_target.shape
        if self.reduction == 'none':
            return 1.0 - soft_dice_loss(output, one_hot_target)
        elif self.reduction == 'mean':
            return 1.0 - torch.mean(soft_dice_loss(output, one_hot_target))
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


# NOT SURE
class BatchSoftDice(nn.Module):
    def __init__(self, use_square=False):
        """
        Args:
            use_square: If use square then the denominator will the sum of square
        """
        super(BatchSoftDice, self).__init__()
        self._use_square = use_square

    def forward(self, output, target, epsilon=1e-6):
        """
        This is the variance of SoftDiceLoss, it in introduced in:
        https://arxiv.org/pdf/1812.02427.pdf
        Args:
            output: Tensor shape (N, N_Class, H, W), torch.float
            target: Tensor shape (N, H, W)
            epsilon: Use this term to avoid undefined edge case
        Returns:
        """
        num_classes = output.shape[1]
        batch_size = output.shape[0]
        axes = (-2, -1)
        output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2))
        assert output.shape == one_hot_target.shape
        numerator = 2. * torch.sum(output * one_hot_target, dim=axes)
        if self._use_square:
            denominator = torch.sum(torch.square(output) + torch.square(one_hot_target), dim=axes)
        else:
            denominator = torch.sum(output + one_hot_target, dim=axes)
        return (1 - torch.mean((numerator + epsilon) / (denominator + epsilon))) * batch_size
        # return 1 - torch.sum(torch.mean(((numerator + epsilon) / (denominator + epsilon)), dim=1))


# DONE
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='none', eps=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, output, target):
        num_classes = output.shape[1]
        output_softmax = F.softmax(output, dim=1)
        output_log_softmax = F.log_softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        weight = torch.pow(1.0 - output_softmax, self.gamma)
        focal = -self.alpha * weight * output_log_softmax
        # This line is very useful, must learn einsum, bellow line equivalent to the commented line
        # loss_tmp = torch.sum(focal.to(torch.float) * one_hot_target.to(torch.float), dim=1)
        loss_tmp = torch.einsum('bc..., bc...->b...', one_hot_target, focal)
        if self.reduction == 'none':
            return loss_tmp
        elif self.reduction == 'mean':
            return torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            return torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


# DONE
class TverskyLoss(nn.Module):
    """
    Tversky Loss is the generalization of Dice Loss
    It in the group of Region-Base Loss
    """
    def __init__(self, beta=0.5, use_softmax=True):
        """
        Args:
            beta:
            use_softmax: Set to False is used for testing purpose, when training model, use default True instead
        """
        super(TverskyLoss, self).__init__()
        self.beta = beta
        self.use_softmax = use_softmax

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        if self.use_softmax:
            output = F.softmax(output, dim=1)  # predicted value
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == target.shape
        # Notice: TverskyIndex is numerator / denominator
        # See https://en.wikipedia.org/wiki/Tversky_index and we have the quick comparison between probability and set \
        # G is the Global Set, A_ = G - A, then
        # |A - B| = |A ^ B_| = |A ^ (G - B)| so |A - B| in set become (1 - target) * (output)
        # With ^ = *, G = 1
        numerator = torch.sum(output * target, dim=(-2, -1))
        denominator = numerator + self.beta * torch.sum((1 - target) * output, dim=(-2, -1)) + (1 - self.beta) * torch.sum(target * (1 - output), dim=(-2, -1))
        return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))


# DONE
class FocalTverskyLoss(nn.Module):
    """
    More information about this loss, see: https://arxiv.org/pdf/1810.07842.pdf
    This loss is similar to Tversky Loss, but with a small adjustment
    With input shape (batch, n_classes, h, w) then TI has shape [batch, n_classes]
    In their paper TI_c is the tensor w.r.t to n_classes index

    FTL = Sum_index_c(1 - TI_c)^gamma
    """
    def __init__(self, gamma=1, beta=0.5, use_softmax=True):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.use_softmax = use_softmax

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        if self.use_softmax:
            output = F.softmax(output, dim=1)  # predicted value
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == target.shape
        numerator = torch.sum(output * target, dim=(-2, -1))
        denominator = numerator + self.beta * torch.sum((1 - target) * output, dim=(-2, -1)) + (
                    1 - self.beta) * torch.sum(target * (1 - output), dim=(-2, -1))
        TI = torch.mean((numerator + epsilon) / (denominator + epsilon), dim=0)  # Shape [batch, num_classes], should reduce along batch dim
        return torch.sum(torch.pow(1.0 - TI, self.gamma))


# DONE
class LogCoshDiceLoss(nn.Module):
    """
    L_{lc-dce} = log(cosh(DiceLoss)
    """
    def __init__(self, use_softmax=True):
        super(LogCoshDiceLoss, self).__init__()
        self.use_softmax = use_softmax

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == one_hot_target.shape
        numerator = 2. * torch.sum(output * one_hot_target, dim=(-2, -1))  # Shape [batch, n_classes]
        denominator = torch.sum(output + one_hot_target, dim=(-2, -1))
        return torch.log(torch.cosh(1 - torch.mean((numerator + epsilon) / (denominator + epsilon))))


# Helper function for sensitivity-specificity loss
def sensitivity_specificity_loss(y_true, y_pred, w):
    """
    True positive example (True - Reality, Positive - Wolf):
    A sentence to describe it - we make the positive prediction and this is True in Reality .
    * Reality: A wolf threatened
    * Shepherd said: "Wolf"
    * Outcome: Shepherd is a hero
    Args:
        y_true: probability (one hot) shape [batch, n_classes, h, w]
        y_pred: probability (softmax(output) or sth like that) shape [batch, n_classes, h, w]
    Returns:
        Loss: A tensor
    """
    assert y_true.shape == y_pred.shape
    n_classes = y_true.shape[1]
    confusion_matrix = torch.zeros((n_classes, n_classes), dtype=torch.float)
    y_true = torch.argmax(y_true, dim=1)  # Reduce to [batch, h, w]
    y_pred = torch.argmax(y_pred, dim=1)
    # Use trick to compute the confusion matrix
    # Reference: https://github.com/monniert/docExtractor/
    for y_true_item, y_pred_item in zip(y_true, y_pred):
        y_true_item = y_true_item.flatten()  # Reduce to 1-D tensor
        y_pred_item = y_pred_item.flatten()
        confusion_matrix += torch.bincount(n_classes * y_true_item + y_pred_item, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    # From confusion matrix, we compute tp, fp, fn, tn
    # Get the answer from this discussion:
    # https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier
    sum_along_classified = torch.sum(confusion_matrix, dim=1)  # sum(c1_1, cn_1) return 1D tensor
    sum_along_actual = torch.sum(confusion_matrix, dim=0)  # sum(c1_1 -> c1_n)
    tp = torch.diagonal(confusion_matrix, offset=0)
    fp = sum_along_classified - tp
    fn = sum_along_actual - tp
    tn = torch.ones(n_classes, dtype=torch.float) * torch.sum(confusion_matrix) - tp - fp - fn
    smooth = torch.ones(n_classes, dtype=torch.float)  # Use to avoid numeric division error
    assert tp.shape == fp.shape == fn.shape == tn.shape
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    # Relation between tp, fp, fn, tn annotation vs set annotation here, so the actual loss become, compare this
    # loss vs the Soft Dice Loss, see https://arxiv.org/pdf/1803.11078.pdf
    return 1.0 - torch.mean(w * sensitivity + (1 - w) * specificity)


# XXX Bugs
class SensitivitySpecificityLoss(nn.Module):
    def __init__(self, weight=0.5):
        """
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        Args:
            weight: use for the combination of sensitivity and specificity
        """
        super(SensitivitySpecificityLoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        num_classes = output.shape[1]
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        output = F.softmax(output, dim=1)
        return sensitivity_specificity_loss(target, output, self.weight)


# TODO: NOT IMPLEMENTED
class CompoundedLoss(nn.Module):
    def __init__(self):
        super(CompoundedLoss, self).__init__()
        pass

    def forward(self, output, target):
        pass


# DONE
class ComboLoss(nn.Module):
    """
    It is defined as a weighted sum of Dice loss and a modified cross entropy. It attempts to leverage the 
    flexibility of Dice loss of class imbalance and at same time use cross-entropy for curve smoothing. 
    
    This loss will look like "batch bce-loss" when we consider all pixels flattened are predicted as correct or not

    Paper: https://arxiv.org/pdf/1805.02798.pdf. See the original paper at formula (3)
    Author's implementation in Keras : https://github.com/asgsaeid/ComboLoss/blob/master/combo_loss.py

    This loss is perfect loss when the training loss come to -0.5 (with the default config)
    """
    def __init__(self, use_softmax=True, ce_w=0.5, ce_d_w=0.5, eps=1e-12):
        super(ComboLoss, self).__init__()
        self.use_softmax = use_softmax
        self.ce_w = ce_w
        self.ce_d_w = ce_d_w
        self.eps = 1e-12
        self.smooth = 1

    def forward(self, output, target):
        num_classes = output.shape[1]
        
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        
        # At this time, the output and one_hot_target have the same shape        
        y_true_f = torch.flatten(one_hot_target)
        y_pred_f = torch.flatten(output)
        intersection = torch.sum(y_true_f * y_pred_f)
        d = (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

        # From this thread: https://discuss.pytorch.org/t/bceloss-how-log-compute-log-0/11390. Use this trick to advoid nan when log(0) and log(1)
        out = - (self.ce_w * y_true_f * torch.log(y_pred_f + self.eps) + (1 - self.ce_w) * (1.0 - y_true_f) * torch.log(1.0 - y_pred_f + self.eps))
        weighted_ce = torch.mean(out, axis=-1)

        # Due to this is the hibird loss, then the loss can become negative: https://discuss.pytorch.org/t/negative-value-in-my-loss-function/101776
        combo = (self.ce_d_w * weighted_ce) - ((1 - self.ce_d_w) * d)
        return combo


# DONE
class ExponentialLogarithmicLoss(nn.Module):
    """
    This loss is focuses on less accurately predicted structures using the combination of Dice Loss ans Cross Entropy
    Loss
    
    Original paper: https://arxiv.org/pdf/1809.00076.pdf
    
    See the paper at 2.2 w_l = ((Sum k f_k) / f_l) ** 0.5 is the label weight
    
    Note: 
        - Input for CrossEntropyLoss is the logits - Raw output from the model
    """
    
    def __init__(self, w_dice=0.5, w_cross=0.5, gamma=0.3, use_softmax=True, class_weights=None):
        super(ExponentialLogarithmicLoss, self).__init__()
        self.w_dice = w_dice
        self.gamma = gamma
        self.w_cross = w_cross
        self.use_softmax = use_softmax
        self.class_weights = class_weights

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        assert len(self.class_weights) == num_classes, "Class weight must be not None and must be a Tensor of size C - Number of classes"
        
        # Generate the class weights array. Shape (batch_size, height, width), at pixel n, the nuber is the weight of the true class
        weight_map = self.class_weights[target]
        
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        
        l_dice = torch.mean(torch.pow(-torch.log(soft_dice_loss(output, target)), self.gamma))   # mean w.r.t to label
        l_cross = torch.mean(torch.mul(weight_map, torch.pow(F.cross_entropy(output, target, reduction='none'), self.gamma)))
        return self.w_dice * l_dice + self.w_cross * l_cross


# This is use for testing purpose
if __name__ == '__main__':
    # loss = BCEWithLogitsLoss(reduction="none")
    # loss = BCELoss()
    # loss = CrossEntropyLoss()
    # loss = FocalLoss(alpha=1.0, reduction='mean', gamma=1)
    # loss = SoftDiceLoss(reduction='mean', use_softmax=False)
    # loss = SensitivitySpecificityLoss(weight=0.5)
    # loss = LogCoshDiceLoss(use_softmax=True)
    # loss = BatchSoftDice(use_square=False)
    # loss = TverskyLoss()
    # loss = FocalTverskyLoss(use_softmax=True)
    # loss = ExponentialLogarithmicLoss(use_softmax=True, class_weights=torch.tensor([0.2, 0.4, 0.1, 0.1, 0.1, 0.1]))
    loss = ComboLoss(use_softmax=True, ce_d_w=0.5, ce_w=0.5)

    ###### Binary classification test ######
    # output = torch.randn((1, 2, 1, 1), requires_grad=True)
    # target = torch.empty((1, 1, 1), dtype=torch.float).random_(2)
    # output_ = F.one_hot(target.to(torch.int64), num_classes=2).permute((0, 3, 1, 2))
    
    ###### Multiple classes classification test ######
    batch_size = 2
    n_classes = 6
    height = 3 
    width = 5
    
    output = torch.randn((batch_size, n_classes, height, width), requires_grad=True)  # Shape: n_samples, n_classes, h, w 
    target = torch.empty((batch_size, height, width), dtype=torch.long).random_(n_classes)   # Shape: n_samples, h, w, each cell represent the class index
    output_ = F.one_hot(target.to(torch.int64), num_classes=n_classes).permute((0, 3, 1, 2)).to(torch.float)  # Mimic the "ideal" model output after going through sigmoid function
    output_.requires_grad = True
    
    print(f'Output shape: {output.shape}')
    print(f'Output_ shape: {output_.shape}')
    print(f'Target shape: {target.shape}')

    # TEST: Test loss between the logit output of the model and the groud truth label then we need to enable the use_softmax=True flag to True when init the loss function to test this
    loss_1 = loss(output, target) 
    print(f'Loss 1 value: {loss_1}')
    loss_1.backward()
    print(output.grad)  
    
    # TEST: Test loss function when the input and target are the same (model prediction will be output like the one-hot encoded vector, so set the use_softmax=False when init the loss function to test this)
    # loss_2 = loss(output_, target)   
    # print(f'Loss 2 value: {loss_2}')
    # loss_2.backward()
    # print(output.grad)
