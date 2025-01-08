import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc

from medpy.metric import binary
import scipy.stats as stats
def iou_score(output, target):
    smooth = 1e-5
    iou_value = 0
    dice_value = 0

    _, N, _, _ = output.shape

    output_list = torch.split(output, 1, dim=1)
    target_list = torch.split(target, 1, dim=1)

    for i in range(N):

        if torch.is_tensor(output_list[i]):
            _output = torch.sigmoid(output_list[i]).data.cpu().numpy()
        if torch.is_tensor(target_list[i]):
            _target = target_list[i].data.cpu().numpy()
        output_ = _output > 0.5
        target_ = _target > 0.5

        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * iou) / (iou + 1)
        iou_value += iou
        dice_value += dice

    return iou_value/N, dice_value/N





def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)



def all_score(output, target):
    smooth = 1e-5
    accuracy_tab = []
    precision_tab = []
    recall_tab = []
    p_value_tab = []
    iou_tab = []
    f1_tab = []


    _,N,_,_ = output.shape
    output_list = torch.split(output, 1, dim=1)
    target_list = torch.split(target, 1, dim=1)

    for i in range(N):
        if torch.is_tensor(output_list[i]):
            output = torch.sigmoid(output_list[i]).data.cpu().numpy()
        if torch.is_tensor(target_list[i]):
            target = target_list[i].data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5

        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        f1 = (2* iou) / (iou+1)

        tp = np.sum(np.logical_and(output_, target_))

        # 计算True Negative（TN）
        tn = np.sum(np.logical_and(np.logical_not(output_), np.logical_not(target_)))

        # 计算False Positive（FP）
        fp = np.sum(np.logical_and(output_, np.logical_not(target_)))

        # 计算False Negative（FN）
        fn = np.sum(np.logical_and(np.logical_not(output_), target_))

        # iou = tp / float(tp + fn + fp)
        # 计算Accuracy（准确率）
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)

        # 计算precision（精确率）
        precision = tp / (tp + fp + 1e-7)

        # 计算recall（召回率）
        recall = tp / (tp + fn + 1e-7)


        _, p_value = stats.ttest_ind(output_.reshape(-1), target_.reshape(-1))


        # 计算F1-score
        # f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        # # 计算Specificity（特异度）
        # specificity = tn / (tn + fp + 1e-7)
        # dice = 0
        accuracy_tab.append(accuracy)
        precision_tab.append(precision)
        recall_tab.append(recall)
        p_value_tab.append(p_value)
        iou_tab.append(iou)
        f1_tab.append(f1)


    return iou_tab, f1_tab, accuracy_tab, precision_tab,recall_tab,p_value_tab

