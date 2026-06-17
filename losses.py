import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalSegLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def binary_cross_entropy(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)
    
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.0
        
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        pred = torch.sigmoid(pred)
        
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_factor = torch.where(target == 1, alpha, 1 - alpha)
        modulating_factor = (1.0 - pt) ** gamma
        
        focal = alpha_factor * modulating_factor * bce
        return focal.mean()
    
    def forward(self, pred, target):
        bce_loss = self.binary_cross_entropy(pred, target)
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss + 
                     self.focal_weight * focal_loss)
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item(),
            'focal_loss': focal_loss.item()
        } 