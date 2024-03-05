from __future__ import print_function, division
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import numpy as np

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

def dice_score(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0
    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return ((2. * intersection) / (i_flat.sum() + t_flat.sum()))

def bce_loss_w(input, target, w=0.8):
    # bce_loss = nn.BCELoss(size_average=True)
    weight=torch.zeros_like(target)
    weight=torch.fill_(weight,(1-w))
    weight[target>0]=w
    loss=nn.BCELoss(weight=weight,size_average=True)(input,target.float())
    # loss=nn.BCEWithLogitsLoss(weight=weight,size_average=True)(input,target.float())
    return loss


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    # prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def cross_entropy_loss(prediction, target, class_num = 100):
    # print(target)
    # class_num = 100
    # prediction = torch.eye(100)[0:3, :]*10
    # prediction[0][1] = 10

    # target = torch.zeros([3,2])
    # target[0][0] = 0
    # target[0][1] = 1

    # target[1][0] = 1
    # target[1][1] = 3

    # target[2][0] = 1
    # target[2][1] = 3
    # target = torch.from_numpy(target)

    # print(prediction.shape, target.shape)
    # print(target[:,0], target[:,1])
    # prediction_softmax = F.softmax(prediction, dim=1)
    # print(prediction_softmax)
    prediction_log_softmax = F.log_softmax(prediction, dim=1)
    target_onehot_0 = F.one_hot(target[:,0].unsqueeze(0).to(torch.int64), num_classes=class_num)
    target_onehot_1 = F.one_hot(target[:,1].unsqueeze(0).to(torch.int64), num_classes=class_num)

    
    cross_entropy = - (target_onehot_0 + target_onehot_1)/2 * prediction_log_softmax
    cross_entropy_loss = cross_entropy.sum() / prediction.shape[0]

    # print(cross_entropy)
    # print(cross_entropy_loss)

    # pause

    return cross_entropy_loss


def MSE_loss(prediction, target):
    # prediction_radius = torch.clone(prediction[:,:,:,0:1])
    # prediction_pos = torch.clone(prediction[:,:,:,1:4])

    # target_radius = target[:,:,:,0:1]
    # target_pos = target[:,:,:,1:4]

    # i_flat_pos = prediction_pos.view(-1)
    # t_flat_pos = target_pos.view(-1)

    # i_flat_radius = prediction_radius.view(-1)
    # t_flat_radius = target_radius.view(-1)


    # pos_error = (i_flat_pos - t_flat_pos)**2
    # pos_loss = pos_error.sum() / i_flat.shape[0]

    # radius_error = (i_flat_radius - t_flat_radius)**2
    # radius_loss = radius_error.sum() / i_flat.shape[0]

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    error = (i_flat - t_flat)**2

    loss = error.sum() / i_flat.shape[0]

    return loss

def Log_loss(prediction, target):
    # prediction_radius = torch.clone(prediction[:,:,:,0:1])
    # prediction_pos = torch.clone(prediction[:,:,:,1:4])

    # target_radius = target[:,:,:,0:1]
    # target_pos = target[:,:,:,1:4]

    # i_flat_pos = prediction_pos.view(-1)
    # t_flat_pos = target_pos.view(-1)

    # i_flat_radius = prediction_radius.view(-1)
    # t_flat_radius = target_radius.view(-1)


    # pos_error = (i_flat_pos - t_flat_pos)**2
    # pos_loss = pos_error.sum() / i_flat.shape[0]

    # radius_error = (i_flat_radius - t_flat_radius)**2
    # radius_loss = radius_error.sum() / i_flat.shape[0]

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    # print(i_flat.shape, t_flat.shape)

    error = abs(torch.log(i_flat/t_flat))
    loss = error.sum() / i_flat.shape[0]


    return loss

def L1_loss(prediction, target):

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    error = abs(i_flat - t_flat)
    loss = error.sum() / i_flat.shape[0] *100


    return loss


def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds