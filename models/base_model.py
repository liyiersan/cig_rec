import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class BaseModel(nn.Module):
    """
        Parameters:
            model_name: 模型的名称
            pretrained: 是否使用预训练模型
            feat_dim: 特征的维度
            num_classes: 分类的类别数
            norm_linear: 是否使用归一化的线性层
    """
    def __init__(self, model_name, pretrained, feat_dim, num_classes=2, norm_linear=False):
        super(BaseModel, self).__init__()
        BackboneModel = getattr(models, model_name)
        self.backbone = BackboneModel(pretrained=pretrained)
        # 有些是fc, 有些是classifier, 有些是head
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        self.linear = NormedLinear(feat_dim, num_classes) if norm_linear else nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 模型的输入, shape: [batch_size, c, h, w]
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1) # shape: [batch_size, feat_dim]
        return self.linear(x), x 