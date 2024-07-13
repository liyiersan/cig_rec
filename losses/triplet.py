import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from losses.base import BaseLoss


class TripletLoss(BaseLoss):
    """
        三元组损失的代码实现: 包含对难样本的挖掘.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for
            Person Re-Identification. arXiv:1703.07737.
    Imported from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/triplet_loss.py. (hard mining)
                 https://github.com/ShiqiYu/OpenGait/blob/master/opengait/modeling/losses/triplet.py (no hard mining)
    Args:
        margin (float, optional): Margin for triplet loss. Defaults to 0.3.
        loss_term_weight (float, optional): Weight of the loss. Defaults to 1.0.
        hard_mining (bool, optional): Whether to perform hard mining.
            Defaults to True.
    """

    def __init__(self,
                 margin: float = 0.3,
                 loss_term_weight: float = 1.0,
                 hard_mining=True):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.hard_mining = hard_mining

    def compute_dist(self, x: torch.Tensor) -> torch.Tensor:
        """
            计算欧式距离矩阵

        Args:
            x (torch.Tensor): 特征矩阵, shape: [batch_size, feat_dim]

        Returns:
            torch.Tensor: 欧式距离矩阵, shape: [batch_size, batch_size]
        """
        batch_size = x.size(0)
        # Compute Euclidean distance
        dist = torch.pow(x, 2).sum(
            dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist.addmm_(x, x.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def hard_mining_triplet_loss_forward(
            self, inputs: torch.Tensor,
            targets: torch.LongTensor) -> torch.Tensor:
        """
            带有难样本挖掘的三元组损失函数
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (batch_size).

        Returns:
            torch.Tensor: triplet loss with hard mining.
        """
        batch_size = inputs.size(0)
        dist = self.compute_dist(inputs)

        # For each anchor, find the furthest positive sample
        # and nearest negative sample in the embedding space
        mask = targets.expand(batch_size, batch_size).eq(
            targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):
            pos_mask = mask[i]
            neg_mask = ~mask[i]
            device = inputs.device
            if pos_mask.sum() > 1:  # 至少包含一个正样本（不包括自己）
                dist_ap.append(dist[i][pos_mask].max().unsqueeze(0))  # hardest positive
            else:
                dist_ap.append(torch.tensor(0.0).unsqueeze(0).to(device))  # 默认值

            if neg_mask.sum() > 0:  # 至少包含一个负样本
                dist_an.append(dist[i][neg_mask].min().unsqueeze(0))  # hardest negative
            else:
                dist_an.append(torch.tensor(2.0).unsqueeze(0).to(device))  # 默认值
        dist_ap = torch.cat(dist_ap) # [batch_size]
        dist_an = torch.cat(dist_an) # [batch_size]

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
    

    def no_hard_mining_triplet_loss_forward(
            self, inputs: torch.Tensor,
            targets: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (num_classes).

        Returns:
            torch.Tensor: triplet loss without hard mining.
        """
        batch_size = inputs.size(0)
        dist = self.compute_dist(inputs)
        # Create mask for positive and negative samples
        matches = (targets.unsqueeze(1) ==
                   targets.unsqueeze(0)).bool()  # [batch_size, batch_size]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        
        total_loss = 0
        for i in range(batch_size):
            dist_ap_i = dist[i][matches[i]].unsqueeze(1) # [n_pos_i, 1]
            dist_an_i = dist[i][diffenc[i]].unsqueeze(0) # [1, n_neg_i]
            diff_dist_i = dist_ap_i - dist_an_i # [n_pos_i, n_neg_i]
            loss_i = F.relu(diff_dist_i + self.margin) # [n_pos_i, n_neg_i]
            loss_avg_i = self.AvgNonZeroReducer(loss_i) # [1]
            total_loss += loss_avg_i

        return total_loss / batch_size
    
    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum()
        loss_num = (loss != 0).sum().float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg
        

    def forward(self, logits, feats: torch.Tensor,
                targets: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            feats (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (num_classes).

        Returns:
            torch.Tensor: triplet loss.
        """
        
        # normalize features to reduce numerical instability
        # feats = F.normalize(feats, dim=1)
        
        if self.hard_mining:
            loss = self.hard_mining_triplet_loss_forward(feats, targets)
        else:
            loss = self.no_hard_mining_triplet_loss_forward(feats, targets)
        return loss * self.loss_term_weight

if __name__ == "__main__":
    # Test Triplet Loss
    loss = TripletLoss(hard_mining=False)
    feats = torch.randn(1, 64)
    targets = torch.randint(0, 2, (1, ))
    print(loss(None, feats, targets))