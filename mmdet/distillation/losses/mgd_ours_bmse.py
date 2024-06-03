import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from ..builder import DISTILL_LOSSES

from torch.distributions import MultivariateNormal as MVN

@DISTILL_LOSSES.register_module()
class FeatureLossOursBMSE(nn.Module):

    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65,
                 ):
        super(FeatureLossOursBMSE, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.name = name
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))


    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        # N,C,H,W = preds_S.shape
        #
        # feature_map_t = self.get_feature_map(preds_T)
        # feature_map_s = self.get_feature_map(preds_S)
        #
        # Mask_fg = torch.zeros_like(feature_map_t)
        # Mask_bg = torch.ones_like(feature_map_s)




        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_mgd
            
        return loss


    # def get_feature_map(self, preds):
    #     """ preds: Bs*C*W*H """
    #     N, C, H, W= preds.shape
    #
    #     value = torch.abs(preds)
    #     # Bs*W*H
    #     fea_map = value.mean(axis=1, keepdim=True)
    #
    #     return fea_map

    def bmc_loss(self, pred, target, noise_var):
        """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
          pred: A float tensor of size [batch, 1].
          target: A float tensor of size [batch, 1].
          noise_var: A float number or tensor.
        Returns:
          loss: A float tensor. Balanced MSE Loss.
        """
        logits = - (pred - target).pow(2) / (2 * noise_var)  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())  # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

        return loss

    def bmse_loss(self, inputs, targets, noise_sigma=8.):
        return self.bmc_loss(inputs, targets, noise_sigma ** 2)

    def weighted_focal_mse_loss(self, inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
        loss = (inputs - targets) ** 2
        loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
            (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
        if weights is not None:
            loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss

    def bmse_loss_md(self, inputs, targets, noise_sigma=8.):
        return self.bmc_loss_md(inputs, targets, noise_sigma ** 2)

    def bmc_loss_md(self, pred, target, noise_var):
        """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
          pred: A float tensor of size [batch, d].
          target: A float tensor of size [batch, d].
          noise_var: A float number or tensor.
        Returns:
          loss: A float tensor. Balanced MSE Loss.
        """
        I = torch.eye(pred.shape[-1]).cuda()
        logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())  # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        N, C, H, W = preds_T.shape
        # print('preds_S.shape', preds_S.shape)
        # print('preds_T.shape', preds_T.shape)

        device = preds_S.device
        mat_old = torch.rand((N,1,H,W)).to(device)
        mat_old = torch.where(mat_old>1-self.lambda_mgd, 0, 1).to(device)

        arr = np.tile(np.array([[0, 1], [1, 0]]), (math.ceil(H/2), math.ceil(W/2)))
        mat = torch.from_numpy(arr[:H, :W]).to(device)
        mat = mat.repeat(N, 1, 1)
        mat = torch.unsqueeze(mat, dim=1)


        # mat = torch.ones((N,1,H,W)).to(device)
        # for n in range(N):
        #     for h in range(H):
        #         for w in range(W):
        #             if h % 2 != w % 2:
        #                 mat[n][0][h][w] = 0



        masked_fea = torch.mul(preds_S, mat)
        from tools_det.visualize_feature import draw_feature_map
        # draw_feature_map(masked_fea, 'masked_fea_')

        masked_fea_old = torch.mul(preds_S, mat_old)
        # draw_feature_map(masked_fea_old, 'masked_fea_old_')

        new_fea = self.generation(masked_fea)
        # draw_feature_map(new_fea, 'new_feature_')

        new_fea_old = self.generation(masked_fea_old)
        # draw_feature_map(new_fea_old, 'new_feature_old_')

        # draw_feature_map(preds_T, 'preds_T_')

        # dis_loss = loss_mse(new_fea, preds_T)/N
        dis_loss = self.bmse_loss_md(preds_T, new_fea)/N

        return dis_loss