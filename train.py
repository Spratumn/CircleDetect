import os
import torch
from torch.utils.data import DataLoader
from config import Config
from models.model import create_model, save_model, load_model
from utils.trainer import Trainer
from utils.dataset import MyDataset


def main():
    cfg = Config()
    torch.manual_seed(cfg.SEED)
    device = torch.device('cuda' if cfg.GPU[0] >= 0 else 'cpu')

    start_epoch = 11
    if start_epoch == 1:
        train_log = open(os.path.join(cfg.LOG_DIR, "train_log.csv"), 'w')
        train_log_title = "epoch,total_loss,hm_loss,wh_loss"
        if cfg.USE_OFFSET:
            train_log_title += ",offset_loss\n"
        else:
            train_log_title += "\n"
        train_log.write(train_log_title)
        train_log.flush()
    else:
        train_log = open(os.path.join(cfg.LOG_DIR, "train_log.csv"), 'a')

    print('Creating model...')
    model = create_model(cfg)
    if start_epoch != 1:
        model = load_model(model, 'log/weights/model_epoch_{}.pth'.format(start_epoch-1))
    optimizer = torch.optim.Adam(model.parameters(), cfg.LR)

    trainer = Trainer(cfg, model, optimizer)
    trainer.set_device(cfg.GPU, device)
    print('Setting up data...')
    train_loader = DataLoader(MyDataset(cfg, 'train'),
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True)
    print('Starting training...')
    epoch = start_epoch
    for epoch in range(start_epoch, start_epoch + cfg.NUM_EPOCHS):
        trainer.train(epoch, train_loader, train_log)
        save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_epoch_{}.pth'.format(epoch)), epoch, model, optimizer)
    save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_last.pth'), epoch, model, optimizer)


def test_dataloader():
    cfg = Config()
    data_loader = DataLoader(MyDataset(cfg, 'train'),
                             batch_size=1,
                             shuffle=True,
                             num_workers=cfg.NUM_WORKERS,
                             pin_memory=True,
                             drop_last=True)
    img, label = data_loader.dataset.__getitem__(1)
    print(img.shape)
    print(label.keys())  # dict_keys(['input', 'hm', 'reg_mask', 'ind', 'dense_wh', 'dense_wh_mask', 'reg'])


if __name__ == '__main__':
    main()



