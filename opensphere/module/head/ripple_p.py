import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.dist_helper import get_rank, get_world_size
from .common import get_wrapping, get_scoring, get_fusing

class RipPLE_P(nn.Module):
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
        self.f_fusing = get_fusing(False, False)
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
        x = self.f_wrapping(x)
        # reg_loss, _ = self.norm_reg(x) # 这里的x是经过wrapped后的特征
        loss_L, _ = self.loss_L(x, y)
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
        # loss = self.lw * bce_loss + self.lambda_ * reg_loss
        loss = self.lw * bce_loss + self.lambda_ * loss_L

        return loss

    # def norm_reg(self, raw_feats):
    #     feat_norms = torch.norm(raw_feats, p=2, dim=1)
    #     mean_norm = feat_norms.mean()
    #     reg_loss = (mean_norm - self.L) ** 2
    #     return reg_loss, mean_norm
    
    def loss_L(self, x, y=None):
        """
        :param x: wrapped 特征 [B, feat_dim] (其 2-范数即判决分中的模长 r)
        :param y: 样本标签 [B] (用于构造 batch 内同身份对)
        :return: loss_L, mean_magnitude
        """
        n = torch.norm(x, p=2, dim=-1)                      # [B] 逐样本模长

        if y is not None:
            b = x.size(0)
            mask = y.view(-1, 1).eq(y.view(1, -1))
            mask = mask & (~torch.eye(b, dtype=torch.bool, device=x.device))
            if bool(mask.any()):
                u = x / (n[:, None] + 1e-12)                     # 单位方向
                # d^2 只当"目标"用, 必须 detach: 否则模型可以靠"让同身份特征互相
                # 靠拢(d^2->0)"来降低 log(d^2) —— 直接塌缩特征空间, 而不去调模长。
                d2 = (2.0 - 2.0 * (u @ u.t())).clamp(min=1e-4).detach()   # [B,B] 2(1-cos)
                # 批内归一化: 让似然隐含的 r 尺度 ≈ 锚 L, 否则锚与似然互相拉扯
                d2 = d2 / d2[mask].mean() * 2.0
                s = (self.L / n.clamp(min=1e-3)).pow(2)          # [B] "误差"
                s = s.clamp(min=1e-3)                            # 给 r 一个上界
                s2 = s[:, None] + s[None, :]                     # [B,B] 误差和
                mls = (d2 / (1e-6 + s2) + torch.log(s2))[mask].mean()
                return mls, n.mean()

        # ---- 兜底: batch 内无同身份对时退回"均值锚" ----
        mean_magnitude = n.mean()
        loss_L = F.smooth_l1_loss(
            mean_magnitude,
            self.L * torch.ones_like(mean_magnitude)
        )
        return loss_L, mean_magnitude