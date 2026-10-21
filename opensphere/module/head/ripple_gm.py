import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.dist_helper import get_rank, get_world_size
from .common import get_wrapping, get_fusing, get_scoring

class RipPLE_GM(nn.Module):
    def __init__(
        self, wrap_mode='cartesian',
        b_theta = 0.3,
        alpha=0.0001,
        m=0., lw=1000.,
        L=10.0, lambda_=0.03,
        init_bias=-10.,
    ):
        super().__init__()
        # distributed
        self.rank = get_rank()
        self.world_size = get_world_size()

        # wrapping, fusing, scoring
        self.f_wrapping = get_wrapping(wrap_mode)
        self.f_fusing = get_fusing(pre_norm=False, post_norm=False)
        self.f_scoring = get_scoring(b_theta)

        # hyperparam, alpha is the lambda in paper
        self.alpha = alpha
        self.m = m
        self.L = L
        self.lambda_ = lambda_
        self.lw = lw

        # init bias
        self.bias = nn.Parameter(init_bias + 0. * torch.Tensor(1))


    def forward(self, x, y, x_bank, y_bank):
        # mask as label
        mask_p = y.view(-1, 1).eq(y_bank.view(1, -1))
        mask_n = mask_p.logical_not()
        pt = self.rank * x.size(0)
        mask_p[:, pt:pt+x.size(0)].fill_diagonal_(False)

        # forward
        # 先取第0维
        loss_L, _ = self.loss_L(x) 
        x = self.f_wrapping(x)
        # loss_L, _ = self.loss_L(x) 
        x_bank = self.f_wrapping(x_bank)
        x = self.f_fusing(x[:, None, :])
        x_bank = self.f_fusing(x_bank[:, None, :])
        logits = self.f_scoring(x, x_bank, all_pairs=True)

        logits_p = torch.masked_select(logits, mask_p)
        logits_p = (logits_p - self.m + self.bias)
        logits_n = torch.masked_select(logits, mask_n)
        logits_n = (logits_n + self.m + self.bias)

        # loss
        loss_p = F.binary_cross_entropy_with_logits(logits_p, torch.ones_like(logits_p))
        loss_n = F.binary_cross_entropy_with_logits(logits_n, torch.zeros_like(logits_n))
        bce_loss = self.alpha * loss_p + (1. - self.alpha) * loss_n
        loss = self.lw * bce_loss + self.lambda_ * loss_L

        return loss
    
    def loss_L(self, x):
        """
        :param x: raw feature [B, feat_dim]
        :return: loss_L, geo_mean_magnitude
        """
        log_mean_r = x[..., 0].float().mean()
        geo_mean_magnitude = torch.exp(log_mean_r)
        
        loss_L = F.smooth_l1_loss(
            geo_mean_magnitude,
            self.L * torch.ones_like(geo_mean_magnitude)
        )
        return loss_L, geo_mean_magnitude
    
    # def loss_L(self, x):
    #     """
    #     :param x: raw feature [B, feat_dim]
    #     :return: loss_L, geo_mean_magnitude
    #     """
    #     r = torch.norm(x, p=2, dim=-1)
    #     log_r = torch.log(r)
    #     log_mean_r = log_r.mean()
    #     geo_mean_magnitude = torch.exp(log_mean_r)
        
    #     loss_L = F.smooth_l1_loss(
    #         geo_mean_magnitude,
    #         self.L * torch.ones_like(geo_mean_magnitude)
    #     )
    #     return loss_L, geo_mean_magnitude