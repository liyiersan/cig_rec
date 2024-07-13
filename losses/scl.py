import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from losses.base import BaseLoss

class SCLLoss(BaseLoss):
    def __init__(self, temperature=0.07, loss_term_weight=1.0):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning:
        https://arxiv.org/abs/2004.11362
        Copy-paste from: https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
        :param temperature: float, softmax temperature
        """
        super(SCLLoss, self).__init__(loss_term_weight)
        self.temperature = temperature

    def forward(self, logits, feats, targets):
        """
        :param feats: torch.Tensor, shape [batch_size, feat_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        # normalize features to reduce numerical instability
        feats = F.normalize(feats, dim=1)

        device = torch.device("cuda") if feats.is_cuda else torch.device("cpu")
        epsilon = 1e-6
        dot_product_tempered = torch.mm(feats, feats.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + epsilon
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True) + epsilon))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / (cardinality_per_samples + epsilon)
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss * self.loss_term_weight

if __name__ == "__main__":
    # Test SCLLoss
    loss = SCLLoss()
    feats = torch.randn(1, 64)
    targets = torch.randint(0, 2, (1, ))
    print(loss(None, feats, targets))