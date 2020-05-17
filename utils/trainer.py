import torch
from tqdm import tqdm

from models.losses import CenterLoss
from models.data_parallel import DataParallel


class Trainer:
    def __init__(self, cfg, model, optimizer):
        self.cfg = cfg
        self.optimizer = optimizer
        self.model = model
        if cfg.USE_OFFSET:
            self.loss_stats = {'total_loss': [], 'hm_loss': [], 'wh_loss': [], 'offset_loss': []}
        else:
            self.loss_stats = {'total_loss': [], 'hm_loss': [], 'wh_loss': []}
        self.loss = CenterLoss(cfg)

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

    def run_epoch(self, epoch, phase, data_loader, log_file):
        epoch_total_loss = 0.0
        epoch_hm_loss = 0.0
        epoch_wh_loss = 0.0
        epoch_offset_loss = 0.0

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

        data_process = tqdm(data_loader)
        for batch_item in data_process:
            batch_img, batch_label = batch_item
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
            batch_loss = [loss_stats['total_loss'], loss_stats['hm_loss'], loss_stats['wh_loss']]

            epoch_total_loss += batch_loss[0]
            epoch_hm_loss += batch_loss[1]
            epoch_wh_loss += batch_loss[2]

            loss_str = "total_loss: {},hm_loss: {:.4f},wh_loss: {:.4f}".format(batch_loss[0],
                                                                               batch_loss[1],
                                                                               batch_loss[2])
            if 'offset_loss' in loss_stats:
                batch_loss.append(loss_stats['offset_loss'])
                epoch_offset_loss += batch_loss[3]
                loss_str += ",offset_loss: {:.4f}".format(batch_loss[3])

            data_process.set_description_str("epoch:{}".format(epoch))
            data_process.set_postfix_str(loss_str)

        log_str = "{},{:.4f},{:.4f},{:.4f}".format(epoch,
                                                   epoch_total_loss / len(data_loader),
                                                   epoch_hm_loss / len(data_loader),
                                                   epoch_wh_loss / len(data_loader)
                                                   )
        if self.cfg.USE_OFFSET:
            log_str += ",{:.4f}\n".format(epoch_offset_loss / len(data_loader))
        else:
            log_str += "\n"
        log_file.write(log_str)
        log_file.flush()

    def val(self, epoch, data_loader, eval_log):
        return self.run_epoch(epoch, 'eval', data_loader, eval_log)

    def train(self, epoch, data_loader, train_log):
        return self.run_epoch(epoch, 'train', data_loader, train_log)
