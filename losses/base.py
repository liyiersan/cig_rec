import losses
import torch.nn as nn
from utils.common import Odict, get_valid_args, is_dict, is_list

class BaseLoss(nn.Module):
    """
        基础损失函数类, 所有的损失函数都应该继承这个类
    """

    def __init__(self, loss_term_weight=1.0):
        """
        Args:
            loss_term_weight (float): 损失函数的权重, 用于多个损失函数的加权和
        """
        super(BaseLoss, self).__init__()
        self.loss_term_weight = loss_term_weight

    def forward(self, logits, features, labels):
        """
        Args:
            logits (torch.Tensor): 模型的输出, shape: [batch_size, num_classes]
            labels (torch.Tensor): 标签, shape: [batch_size]
            features (torch.Tensor, optional): 模型的输出特征, shape: [batch_size, num_channels]
        """
        return .0
    
class TrainingLoss(nn.Module):
    def __init__(self, loss_cfg):
        """
          根据配置文件构建训练损失函数
        """
        super(TrainingLoss, self).__init__()
        self.losses = nn.ModuleDict({loss_cfg['type']: self._build_loss_(loss_cfg)} if is_dict(loss_cfg) \
            else {cfg['type']: self._build_loss_(cfg) for cfg in loss_cfg})
        self.loss_list = list(self.losses.keys())

    def _build_loss_(self, loss_cfg):
        """Build the losses from loss_cfg.

        Args:
            loss_cfg: Config of loss.
        """
        Loss = getattr(losses, loss_cfg['type'])
        valid_loss_arg = get_valid_args(Loss, loss_cfg, ['type'])
        loss = Loss(**valid_loss_arg)
        return loss
    
    def forward(self, logits, feats, targets):
        """
        Args:
            logits (torch.Tensor): 模型的输出, shape: [batch_size, num_classes]
            feats (torch.Tensor): 模型的输出特征, shape: [batch_size, num_channels]
            targets (torch.Tensor): 标签, shape: [batch_size]
        """
        total_loss = 0
        loss_info = Odict()
        for loss_name, loss in self.losses.items():
            loss_value = loss(logits, feats, targets)
            total_loss += loss_value
            loss_info[loss_name] = loss_value
        return total_loss, loss_info