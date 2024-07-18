import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random


def rle2mask(rle: str, w, h, transpose=True):
    mask = np.zeros((h*w))
    if not pd.isna(rle):
        rle = rle.split(' ')
        pos = [int(x) for x in rle[0::2]]
        length = [int(x) for x in rle[1::2]]

        for p, l in zip(pos, length):
            mask[p-1:p-1+l] = 1
    
    mask = mask.reshape(h, w)
    if transpose:
        return mask.T
    else: 
        return mask
    
    
def mask2rle(mask):
    h, w = mask.shape
    mask_flatten = mask.reshape(-1)
    mask_flatten = np.append(mask_flatten, 0)
    mask_flatten = np.insert(mask_flatten, 0, 0)
    # find start position and end position
    start_pos, *_ = np.where((mask_flatten[1:] - mask_flatten[0:-1]) == 1) # start pos
    start_pos += 1
    end_pos, *_ = np.where((mask_flatten[1:] - mask_flatten[0:-1]) == -1) # end pos
    end_pos += 1
    # calculate length
    length = end_pos - start_pos
    # convert to rle
    rle_list = []
    for p, l in zip(start_pos, length):
        rle_list.append(str(p))
        rle_list.append(str(l))
    rle = ' '.join(rle_list)
    return rle


def predict_mask(inputs, threshold=0.5, logit=True, mode='binary'):
    '''
    Convert ouputs which are present in probability into mask (0, 1).
    Input probability should be `numpy` format.
    '''
    b, c, h, w = inputs.shape
    pred = np.zeros((b, c, h, w))
    
    inputs = torch.tensor(inputs)
    if logit and mode.lower == 'binary':
        inputs = torch.sigmoid(inputs)
    if logit and mode.lower == 'multiclass':
        inputs = F.softmax(inputs)
    
    inputs = inputs.numpy()
    for i in range(b):
        prob = np.squeeze(inputs[i])  # shape: (classes, height, width)
        mask = (prob > threshold).astype('uint8')  # uint8: 0 ~ 225
        pred[i, :, :, :] = mask
    return pred


# Dice coefficient
def dice_coef(y_true, y_pred, smooth = 1e-9):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    return dice


def compute_dice(y_true, y_pred, smooth=1e-9):
    '''
    Both y_true and y_pred are `numpy`
    '''
    b, c, h, w = y_pred.shape
    dice = 0.0
    
    for i in range(b):
        for j in range(c):
            dice += dice_coef(y_true[i, j, :, :], y_pred[i, j, :, :], smooth=smooth)
    dice /= (b * c)
    return dice

def set_seed(seed=None):
    seed = 2022
    torch.manual_seed(seed) # cpu
    np.random.seed(seed) #numpy
    random.seed(seed) #random and transforms

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) #gpu
        torch.backends.cudnn.deterministic=True # cudnn