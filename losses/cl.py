import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from losses.base import BaseLoss


class BinaryContrastiveLoss(BaseLoss):
    def __init__(self, margin=10.0, loss_term_weight=1.0):
        super(BinaryContrastiveLoss, self).__init__(loss_term_weight)
        self.margin = margin

    def forward(self, logits, features, labels):
        # features: (N, D), labels: (N,)
        
        # normalize features to reduce numerical instability
        # feats = F.normalize(feats, dim=1)
        
        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(features, features, p=2)

        # Create positive and negative mask
        labels = labels.unsqueeze(1)
        positive_mask = ((labels == 1) & (labels == labels.t())).float() - torch.eye(labels.shape[0]).to(labels.device)
        positive_mask = torch.clamp(positive_mask, 0, 1)
        negative_mask = (labels != labels.t()).float()

        epsilon = 1e-6
        
        # Compute loss
        positive_loss = (dist_matrix * positive_mask).sum() / (positive_mask.sum() + epsilon)
        negative_loss = (F.relu(self.margin - dist_matrix) * negative_mask).sum() / (negative_mask.sum() + epsilon)
        # negative_loss = - (dist_matrix * negative_mask).sum() / negative_mask.sum()

        return positive_loss + negative_loss
    
class AllContrastiveLoss(BaseLoss):
    def __init__(self, margin=1.0, loss_term_weight=1.0):
        super(AllContrastiveLoss, self).__init__(loss_term_weight)
        self.margin = margin

    def forward(self, logits, features, labels):
        # features: (N, D), labels: (N,)
        
        # normalize features to reduce numerical instability
        # feats = F.normalize(feats, dim=1)
        
        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(features, features, p=2)

        # Create positive and negative mask
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float() - torch.eye(labels.shape[0]).to(labels.device)
        positive_mask = torch.clamp(positive_mask, 0, 1)
        negative_mask = (labels != labels.t()).float()

        epsilon = 1e-6
        
        # Compute loss
        positive_loss = (dist_matrix * positive_mask).sum() / (positive_mask.sum() + epsilon)
        negative_loss = (F.relu(self.margin - dist_matrix) * negative_mask).sum() / (negative_mask.sum() + epsilon)

        return positive_loss + negative_loss
    
if __name__ == "__main__":
    # Test CLLoss
    loss = AllContrastiveLoss()
    feats = torch.randn(1, 64)
    targets = torch.randint(0, 2, (1, ))
    print(loss(None, feats, targets))