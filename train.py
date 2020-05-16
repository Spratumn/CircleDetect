import os
import torch
from torch.utils.data import DataLoader
from config import Config
from models.model import create_model, save_model, load_model
from utils.trainer import CircleTrainer
from utils.dataset import CircleDataset


def main():
    cfg = Config()
    torch.manual_seed(cfg.SEED)
    device = torch.device('cuda' if cfg.GPU[0] >= 0 else 'cpu')
    print('Creating model...')
    model = create_model(cfg)
    model = load_model(model, 'log/weights/model_last.pth')
    optimizer = torch.optim.Adam(model.parameters(), cfg.LR)
    start_epoch = 0
    trainer = CircleTrainer(cfg, model, optimizer)
    trainer.set_device(cfg.GPU, device)
    print('Setting up data...')
    train_loader = DataLoader(CircleDataset(cfg, 'train'),
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True)
    print('Starting training...')
    for epoch in range(start_epoch + 1, cfg.NUM_EPOCHS + 1):
        trainer.train(train_loader)
    save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_last.pth'), epoch, model, optimizer)


def test_dataloader():
    cfg = Config()
    data_loader = DataLoader(CircleDataset(cfg, 'train'),
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



