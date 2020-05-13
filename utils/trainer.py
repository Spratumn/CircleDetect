import torch
import time

from models.losses import CircleLoss
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class CircleTrainer:
    def __init__(self, cfg, model, optimizer):
        self.cfg = cfg
        self.optimizer = optimizer
        self.model = model
        self.loss_stats = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        self.loss = CircleLoss(cfg)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model = DataParallel(self.model, device_ids=gpus).to(device)
            self.loss = DataParallel(self.loss, device_ids=gpus).to(device)
        else:
            self.model = self.model.to(device)
            self.loss = self.loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, data_loader):
        if phase == 'train':
            self.model.train()
            self.loss.train()
        else:
            if len(self.cfg.GPU) > 1:
                self.model = self.model.module
                self.loss = self.loss.module
            self.model.eval()
            self.loss.eval()
            # release cuda cache
            torch.cuda.empty_cache()

        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        for iter_id, batch in enumerate(data_loader):
            batch_img, batch_label = batch
            batch_img = batch_img.to(device=self.cfg.DEVICE)
            for k in batch_label:
                batch_label[k] = batch_label[k].to(device=self.cfg.DEVICE)
            batch_output = self.model(batch_img)
            loss, loss_stats = self.loss(batch_output, batch_label)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch_img.size(0))
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        return ret

    def val(self, data_loader):
        return self.run_epoch('eval', data_loader)

    def train(self, data_loader):
        return self.run_epoch('train', data_loader)
