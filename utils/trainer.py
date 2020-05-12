import torch
import time

from models.losses import ModelWithLoss, CircleLoss
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class CircleTrainer:
    def __init__(self, cfg, model, optimizer):
        self.cfg = cfg
        self.optimizer = optimizer
        self.loss_stats = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        self.model_with_loss = ModelWithLoss(model, CircleLoss(cfg))

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.cfg.GPU) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            for k in batch:
                batch[k] = batch[k].to(device=self.cfg.DEVICE, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        return ret

    def val(self, data_loader):
        return self.run_epoch('eval', data_loader)

    def train(self, data_loader):
        return self.run_epoch('train', data_loader)
