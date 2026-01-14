import torch
import torch.nn as nn
import torch.nn.functional as F


class reg_loss(nn.Module):
    def __init__(self):
        # super().__init__()
        super(reg_loss, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, pred, gt, pcd=[]):
        '''
        :param pred: shape=(B, )
        :param y: shape=(B, )
        :return: loss
        '''
        if pred.dim() > gt.dim():
            pred = pred.squeeze(1)
        loss = self.loss(pred, gt)
        return loss

class reg_loss_augcd(nn.Module):
    def __init__(self):
        # super().__init__()
        super(reg_loss_augcd, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, pred, gt, pcd):
        '''
        :param pred: shape=(B, N)
        :param y: shape=(B, N)
        :return: loss
        '''
        if pred.dim() > gt.dim():
            pred = pred.squeeze(1)
        loss_mse = self.loss(pred, gt)
        loss_cd = self.loss(1.732*pcd[:,3,:]*pred, 1.732*pcd[:,3,:]*gt)
        return loss_mse+loss_cd

class reg_loss_augcd2(nn.Module):
    def __init__(self):
        # super().__init__()
        super(reg_loss_augcd2, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, pred, gt, pcd):
        '''
        :param pred: shape=(B, N)
        :param y: shape=(B, N)
        :return: loss
        '''
        
        if pred.dim() > gt.dim():
            pred = pred.squeeze(1)
        expo = 2
        cd_aug_rate = pcd.shape[2]/torch.sum(torch.abs(pcd[:,3,:])**expo, dim=1)
        cd_aug_rate = cd_aug_rate.repeat((pcd.shape[2],1)).T
        loss_mse = self.loss(pred, gt)
        loss_cd = self.loss(cd_aug_rate*pred*pcd[:,3,:]**expo, cd_aug_rate*gt*pcd[:,3,:]**expo)
        return loss_mse+loss_cd


class LabelSmoothingCE(nn.Module):
    """
    带有标签平滑的交叉熵损失
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        :param pred: (B, num_class, N)  分类时N=1，分割时N等于点云数量
        :param gt: (B, N)
        :return: loss, acc
        """
        B, cls, N = pred.shape

        # (B, cls, N) -> (B*N, cls)
        pred = pred.permute(0, 2, 1).reshape(B*N, cls)
        gt = gt.reshape(B*N,)

        acc = torch.sum(torch.max(pred, dim=-1)[1] == gt) / (B * N)

        logprobs = F.log_softmax(pred, dim=-1)
        loss_pos = -logprobs.gather(dim=-1, index=gt.unsqueeze(1)).squeeze(1)
        loss_smoothing = -logprobs.mean(-1)
        loss = self.confidence * loss_pos + self.smoothing * loss_smoothing

        return loss.mean(), acc.item()



