import torch.nn as nn
from losses.base import BaseLoss


class CELoss(BaseLoss):
    def __init__(self, label_smooth=True, eps=0.1, loss_term_weight=1.0,):
        """
        交叉熵损失
        params:
            label_smooth: 是否使用标签平滑
            eps: 平滑参数
            loss_term_weight: 损失权重
        """
        super(CELoss, self).__init__(loss_term_weight)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=eps) if label_smooth else nn.CrossEntropyLoss()

    def forward(self, logits, feats, labels):
        loss = self.criterion(logits, labels)
        return loss * self.loss_term_weight