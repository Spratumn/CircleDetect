import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    def __init__(self, cfg):
        super(CircleLoss, self).__init__()
        self.hm_loss = FocalLoss()
        self.wh_loss = nn.L1Loss()
        if cfg.USE_OFFSET:
            self.offset_loss = RegL1Loss() if cfg.OFFSET_LOSS == 'l1' else RegLoss()
        self.cfg = cfg

    def forward(self, output, label):
        cfg = self.cfg
        total_hm_loss = self.hm_loss(output['hm'], label['hm'])
        total_wh_loss = self.wh_loss(output['wh'] * label['dense_wh_mask'], label['dense_wh'] * label['dense_wh_mask'])
        if self.cfg.USE_OFFSET:
            total_offset_loss = self.offset_loss(output['offset'], label['offset_mask'], label['ind'], label['offset'])
            total_loss = cfg.HM_WEIGHT * total_hm_loss \
                         + cfg.WH_WEIGHT * total_wh_loss \
                         + cfg.OFF_WEIGHT * total_offset_loss
            loss_stats = {'total_loss': total_loss, 'hm_loss': total_hm_loss,
                          'wh_loss': total_wh_loss, 'offset_loss': total_offset_loss}
        else:
            total_loss = cfg.HM_WEIGHT * total_hm_loss + cfg.WH_WEIGHT * total_wh_loss
            loss_stats = {'total_loss': total_loss, 'hm_loss': total_hm_loss,
                          'wh_loss': total_wh_loss}

        print(loss_stats)
        return total_loss, loss_stats


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, beta=4, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.size_average = size_average

    def forward(self, inputs, target):
        """
        inputs: shape of (N,C,H,W)
        target: shape of (N,C,H,W)
        """
        inputs = inputs.sigmoid_()
        inputs = torch.clamp(inputs, min=1e-4, max=1 - 1e-4)

        pos_inds = target.eq(1)
        neg_inds = target.lt(1)

        neg_weights = torch.pow(1 - target[neg_inds], self.beta)

        loss = 0
        pos_pred = inputs[pos_inds]
        neg_pred = inputs[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, self.gamma)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, self.gamma) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = -1 * neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


def _reg_loss(regr, gt_regr, mask):
    """ L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


if __name__ == '__main__':
    target = torch.zeros(1, 2, 3, 3)
    inputs = torch.zeros(1, 2, 3, 3)
    target[0][0][0][1] = 1
    target[0][1][1][1] = 1
    inputs[0][0][0][1] = 1
    inputs[0][0][1][2] = 1
    inputs[0][1][1][0] = 1
    inputs[0][1][1][1] = 1
    print(inputs)
    print(target)
    loss = FocalLoss()
    loss_val = loss(inputs, target)
    print(loss_val)
