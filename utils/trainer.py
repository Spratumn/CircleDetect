import torch
from tqdm import tqdm

from models.losses import CenterLoss, TeacherLoss
from models.data_parallel import DataParallel
from utils.metrics import compute_metrics
from inference import Detector


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
        # for teacherloss
        # self.loss = TeacherLoss(cfg)

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

    def run_epoch(self, epoch, data_loader, log_file):
        epoch_total_loss = 0.0
        epoch_hm_loss = 0.0
        epoch_wh_loss = 0.0
        epoch_offset_loss = 0.0

        self.model.train()
        self.loss.train()

        data_process = tqdm(data_loader)
        for batch_item in data_process:
            batch_img, batch_label = batch_item
            batch_img = batch_img.to(device=self.cfg.DEVICE)
            for k in batch_label:
                batch_label[k] = batch_label[k].to(device=self.cfg.DEVICE)
            batch_output, feature = self.model(batch_img)
            # for teacherloss
            # batch_output = self.model(batch_img)

            loss, loss_stats = self.loss(batch_output, batch_label)
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss = [loss_stats['total_loss'], loss_stats['hm_loss'], loss_stats['wh_loss']]

            epoch_total_loss += batch_loss[0]
            epoch_hm_loss += batch_loss[1]
            epoch_wh_loss += batch_loss[2]

            loss_str = "total_loss: {},hm_loss: {:.6f},wh_loss: {:.6f}".format(batch_loss[0],
                                                                               batch_loss[1],
                                                                               batch_loss[2])
            if 'offset_loss' in loss_stats:
                batch_loss.append(loss_stats['offset_loss'])
                epoch_offset_loss += batch_loss[3]
                loss_str += ",offset_loss: {:.6f}".format(batch_loss[3])

            data_process.set_description_str("epoch:{}".format(epoch))
            data_process.set_postfix_str(loss_str)

        log_str = "{},{:.6f},{:.6f},{:.6f}".format(epoch,
                                                   epoch_total_loss / len(data_loader),
                                                   epoch_hm_loss / len(data_loader),
                                                   epoch_wh_loss / len(data_loader)
                                                   )
        if self.cfg.USE_OFFSET:
            log_str += ",{:.6f}\n".format(epoch_offset_loss / len(data_loader))
        else:
            log_str += "\n"
        log_file.write(log_str)
        log_file.flush()

    def train(self, epoch, data_loader, train_log):
        return self.run_epoch(epoch, data_loader, train_log)

    def val(self, epoch, model_path, val_loader, val_log, cfg):
        detecter = Detector(model_path, cfg)
        mean_precision = 0
        mean_recall = 0
        sample_count = val_loader.num_samples
        for i in range(sample_count):
            image_path, gt_bboxes = val_loader.getitem(i)
            results = detecter.run(image_path)
            pre_bboxes = results[1]
            if len(gt_bboxes) > 0:
                precision, recall = compute_metrics(pre_bboxes, gt_bboxes)
                mean_precision += precision
                mean_recall += recall

        log_str = "{},{:.6f},{:.6f}\n".format(epoch,
                                              mean_precision / sample_count,
                                              mean_recall / sample_count)
        val_log.write(log_str)
        val_log.flush()


class TeacherTrainer:
    def __init__(self, cfg, teacher, model, optimizer):
        self.cfg = cfg
        self.optimizer = optimizer
        self.teacher = teacher
        self.model = model
        if cfg.USE_OFFSET:
            self.loss_stats = {'total_loss': [], 'hm_loss': [], 'wh_loss': [], 'offset_loss': []}
        else:
            self.loss_stats = {'total_loss': [], 'hm_loss': [], 'wh_loss': []}
        self.loss = TeacherLoss(cfg)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.teacher = DataParallel(self.teacher, device_ids=gpus).to(device)
            self.model = DataParallel(self.model, device_ids=gpus).to(device)
            self.loss = DataParallel(self.loss, device_ids=gpus).to(device)
        else:
            self.teacher = self.teacher.to(device)
            self.model = self.model.to(device)
            self.loss = self.loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, epoch, data_loader, log_file):
        epoch_total_loss = 0.0
        epoch_hm_loss = 0.0
        epoch_wh_loss = 0.0
        epoch_offset_loss = 0.0

        self.teacher.eval()
        self.model.train()
        self.loss.train()

        data_process = tqdm(data_loader)
        for batch_item in data_process:
            batch_img, batch_label = batch_item
            batch_img = batch_img.to(device=self.cfg.DEVICE)
            for k in batch_label:
                batch_label[k] = batch_label[k].to(device=self.cfg.DEVICE)
            teacher_output = self.teacher(batch_img)
            batch_output = self.model(batch_img)
            batch_feature = batch_output['hm']
            teacher_feature = teacher_output['hm']

            loss, loss_stats = self.loss(batch_feature, teacher_feature, batch_output, batch_label)
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss = [loss_stats['total_loss'], loss_stats['hm_loss'], loss_stats['wh_loss']]

            epoch_total_loss += batch_loss[0]
            epoch_hm_loss += batch_loss[1]
            epoch_wh_loss += batch_loss[2]

            loss_str = "total_loss: {},hm_loss: {:.6f},wh_loss: {:.6f}".format(batch_loss[0],
                                                                               batch_loss[1],
                                                                               batch_loss[2])
            if 'offset_loss' in loss_stats:
                batch_loss.append(loss_stats['offset_loss'])
                epoch_offset_loss += batch_loss[3]
                loss_str += ",offset_loss: {:.6f}".format(batch_loss[3])

            data_process.set_description_str("epoch:{}".format(epoch))
            data_process.set_postfix_str(loss_str)

        log_str = "{},{:.6f},{:.6f},{:.6f}".format(epoch,
                                                   epoch_total_loss / len(data_loader),
                                                   epoch_hm_loss / len(data_loader),
                                                   epoch_wh_loss / len(data_loader)
                                                   )
        if self.cfg.USE_OFFSET:
            log_str += ",{:.6f}\n".format(epoch_offset_loss / len(data_loader))
        else:
            log_str += "\n"
        log_file.write(log_str)
        log_file.flush()

    def train(self, epoch, data_loader, train_log):
        return self.run_epoch(epoch, data_loader, train_log)

    def val(self, epoch, model_path, val_loader, val_log, cfg):
        detecter = Detector(model_path, cfg)
        mean_precision = 0
        mean_recall = 0
        sample_count = val_loader.num_samples
        for i in range(sample_count):
            image_path, gt_bboxes = val_loader.getitem(i)
            results = detecter.run(image_path)
            pre_bboxes = results[1]
            if len(gt_bboxes) > 0:
                precision, recall = compute_metrics(pre_bboxes, gt_bboxes)
                mean_precision += precision
                mean_recall += recall

        log_str = "{},{:.6f},{:.6f}\n".format(epoch,
                                              mean_precision / sample_count,
                                              mean_recall / sample_count)
        val_log.write(log_str)
        val_log.flush()
