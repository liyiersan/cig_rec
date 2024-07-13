import torch
import torch.nn as nn
from losses.base import BaseLoss


class BCELoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(BCELoss, self).__init__(loss_term_weight)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, feats, labels):
        if logits.dim() == 2 and logits.shape[-1] == 1:
            one_hot_labels = labels.unsqueeze(1).float()
        else:
            # 把标签转换成one-hot编码
            one_hot_labels = torch.zeros_like(logits)
            one_hot_labels.scatter_(1, labels.unsqueeze(1), 1) # shape: [batch_size, num_classes]
        # 计算损失
        loss = self.criterion(logits, one_hot_labels)

        return loss * self.loss_term_weight