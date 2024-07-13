import torch.nn as nn
from models.base_model import BaseModel

class FaceEdge(BaseModel):
    def __init__(self, model_name, pretrained, feat_dim, num_classes=2, norm_linear=False):
        super(FaceEdge, self).__init__(model_name, pretrained, feat_dim, num_classes, norm_linear)
        self.in_conv = nn.Sequential(
            nn.Conv2d(6, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.in_conv(x)
        return super(FaceEdge, self).forward(x)

