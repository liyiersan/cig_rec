import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from models.base_model import BaseModel

class SeparableConv2d(nn.Module):
  def __init__(self, c_in, c_out, ks, stride=1, padding=0, dilation=1, bias=False):
    super(SeparableConv2d, self).__init__()
    self.c = nn.Conv2d(c_in, c_in, ks, stride, padding, dilation, groups=c_in, bias=bias)
    self.pointwise = nn.Conv2d(c_in, c_out, 1, 1, 0, 1, 1, bias=bias)

  def forward(self, x):
    x = self.c(x)
    x = self.pointwise(x)
    return x

class RegressionMap(nn.Module):
  def __init__(self, c_in):
    super(RegressionMap, self).__init__()
    self.c = SeparableConv2d(c_in, 1, 3, stride=1, padding=1, bias=False)
    self.s = nn.Sigmoid()

  def forward(self, x):
    mask = self.c(x)
    mask = self.s(mask)
    return mask


class DFFD(BaseModel):
    def __init__(self, model_name, pretrained, feat_dim, num_classes=2, norm_linear=False):
        super(DFFD, self).__init__(model_name, pretrained, feat_dim, num_classes, norm_linear)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2]) # remove the last two layers
        self.out_attention = RegressionMap(feat_dim)
        
    def forward(self, x):
        x = self.backbone(x)
        mask = self.out_attention(x)
        x = x * mask
        x = torch.mean(x, dim=(2,3)) # shape: [batch_size, feat_dim]
        return self.linear(x), x 

if __name__ == '__main__':
    model = DFFD('resnet18', False, 512)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y[0].shape, y[1].shape)