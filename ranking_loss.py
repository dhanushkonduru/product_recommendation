"""
Ranking Loss Functions
======================
Proper ranking losses for recommendation (BPR, InfoNCE).

Replaces binary cross-entropy with ranking objectives that optimize
for relative ordering of items.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) Loss.
    
    Optimizes: P(item_i > item_j | user) for positive item_i and negative item_j.
    Loss = -log(sigmoid(score_positive - score_negative))
    """
    
    def __init__(self):
        super(BPRLoss, self).__init__()
    
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            positive_scores: [batch_size] - Scores for positive items
            negative_scores: [batch_size] - Scores for negative items
        
        Returns:
            Scalar loss
        """
        # BPR: -log(sigmoid(score_pos - score_neg))
        diff = positive_scores - negative_scores
        loss = -F.logsigmoid(diff).mean()
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive) Loss.
    
    Optimizes: Maximize similarity between positive pairs,
    minimize similarity between negative pairs.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature parameter for softmax (default: 1.0)
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            positive_scores: [batch_size] - Scores for positive items
            negative_scores: [batch_size, n_negatives] - Scores for negative items
        
        Returns:
            Scalar loss
        """
        batch_size = positive_scores.shape[0]
        
        # Concatenate positive and negatives
        # positive_scores: [batch_size, 1]
        # negative_scores: [batch_size, n_negatives]
        all_scores = torch.cat([
            positive_scores.unsqueeze(1),  # [batch_size, 1]
            negative_scores  # [batch_size, n_negatives]
        ], dim=1)  # [batch_size, 1 + n_negatives]
        
        # Apply temperature
        all_scores = all_scores / self.temperature
        
        # InfoNCE: -log(exp(score_pos) / sum(exp(all_scores)))
        # Positive is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=positive_scores.device)
        loss = F.cross_entropy(all_scores, labels)
        
        return loss


class SampledSoftmaxLoss(nn.Module):
    """
    Sampled Softmax Loss (for large item catalogs).
    
    Approximates full softmax by sampling negatives.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature parameter (default: 1.0)
        """
        super(SampledSoftmaxLoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            positive_scores: [batch_size] - Scores for positive items
            negative_scores: [batch_size, n_negatives] - Scores for negative items
        
        Returns:
            Scalar loss
        """
        # Concatenate positive and negatives
        all_scores = torch.cat([
            positive_scores.unsqueeze(1),
            negative_scores
        ], dim=1) / self.temperature
        
        # Softmax over all scores
        log_probs = F.log_softmax(all_scores, dim=1)
        
        # Negative log-likelihood of positive (at index 0)
        loss = -log_probs[:, 0].mean()
        
        return loss

