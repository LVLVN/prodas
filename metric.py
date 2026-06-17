import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

class SegmentationMetric(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        print(f'Pixel accuracy: diagonal sum={np.diag(self.confusion_matrix).sum()}, matrix sum={self.confusion_matrix.sum()}')
        return acc

    def mean_iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / (union + np.finfo(np.float32).eps)
        print(f'mIoU: intersection={intersection}, union={union}, class IoU={iou}')
        return np.nanmean(iou)

    def class_iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / (union + np.finfo(np.float32).eps)
        return iou

    def dice_coefficient(self):
        intersection = np.diag(self.confusion_matrix)
        sum_pred = np.sum(self.confusion_matrix, axis=0)
        sum_target = np.sum(self.confusion_matrix, axis=1)
        dice = (2.0 * intersection) / (sum_pred + sum_target + np.finfo(np.float32).eps)
        print(f'Dice: intersection={intersection}, pred sum={sum_pred}, target sum={sum_target}, class Dice={dice}')
        return np.nanmean(dice)

    def class_dice(self):
        intersection = np.diag(self.confusion_matrix)
        sum_pred = np.sum(self.confusion_matrix, axis=0)
        sum_target = np.sum(self.confusion_matrix, axis=1)
        dice = (2.0 * intersection) / (sum_pred + sum_target + np.finfo(np.float32).eps)
        return dice

    def _generate_matrix(self, pred, target):
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        if pred.ndim == 4 and pred.shape[1] > 1:
            pred = np.argmax(pred, axis=1)
        elif pred.ndim == 4 and pred.shape[1] == 1:
            pred = (pred.squeeze(1) > 0.5).astype(np.int64)
            
        print(f'Processed shapes - pred: {pred.shape}, target: {target.shape}')
        
        pred = pred.astype(np.int64)
        target = target.astype(np.int64)
        
        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype(np.int64) + pred[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        
        print(f'Confusion matrix:\n{confusion_matrix}')
        print(f'Pred class distribution: {np.bincount(pred[mask].astype(np.int64), minlength=self.num_classes)}')
        print(f'Target class distribution: {np.bincount(target[mask].astype(np.int64), minlength=self.num_classes)}')
        
        return confusion_matrix

    def update(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        print(f'Original input shapes - pred: {pred.shape}, target: {target.shape}')
        
        confusion_matrix = self._generate_matrix(pred, target)
        self.confusion_matrix += confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

def evaluate_prediction(pred, target, num_classes=2):
    metric = SegmentationMetric(num_classes)
    metric.update(pred, target)
    
    return {
        'mIoU': metric.mean_iou(),
        'Dice': metric.dice_coefficient(),
        'Class_IoU': metric.class_iou(),
        'Class_Dice': metric.class_dice(),
        'Pixel_Acc': metric.pixel_accuracy()
    }

