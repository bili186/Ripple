import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_wrapping, get_scoring, weighted_average

class ArcFaceR(nn.Module):
    """ArcFace + polarlike decoupling + pure magnitude regularization.

    - polarlike wrapping: dim 0 -> mag, dims 1: -> unit direction
    - logits = s * (cos(theta + m))  (identical to plain ArcFace)
      * magnitude does NOT enter logits; margin semantics preserved
    - loss_L anchors the batch mean magnitude to L (pure regularization)
    - val_norm=True normalizes away magnitude at inference time
    """
    def __init__(self, feat_dim, subj_num, s=64., m=0.5,
                 L=10.0, lambda_=1.0, val_norm=True):
        super().__init__()
        #
        self.f_wrapping = get_wrapping(mode='polarlike')
        self.f_scoring = get_scoring(b_theta=0.)

        self.feat_dim = feat_dim
        self.subj_num = subj_num
        self.s = s
        self.m = m
        self.L = L
        self.lambda_ = lambda_
        self.val_norm = val_norm

        # class weights live in the direction space (feat_dim - 1)
        self.w = nn.Parameter(torch.Tensor(subj_num, feat_dim - 1))
        nn.init.xavier_normal_(self.w)

    def f_fusing(self, feats, weights=None):
        post_norm = (not self.training) and self.val_norm
        return weighted_average(feats, weights, pre_norm=False, post_norm=post_norm)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=-1)

        x = self.f_wrapping(x)
        loss_L, m_bar = self.loss_L(x)
        x = self.f_fusing(x[:, None, :])

        cos_theta = self.f_scoring(F.normalize(x, dim=-1), self.w, all_pairs=True)
        with torch.no_grad():
            theta_m = torch.acos(cos_theta.clamp(-1 + 1e-5, 1 - 1e-5))
            theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)
        loss = loss + self.lambda_ * loss_L

        return loss

    def loss_L(self, x):
        mean_magnitude = torch.norm(x, p=2, dim=-1).mean()
        loss_L = F.smooth_l1_loss(
            mean_magnitude,
            self.L * torch.ones_like(mean_magnitude)
        )
        return loss_L, mean_magnitude
