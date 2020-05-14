import os
import torch
from torch.utils.data import DataLoader
from config import Config
from models.model import create_model, save_model, load_model
from utils.logger import Logger
from utils.trainer import CircleTrainer
from utils.circledataset import CircleDataset


def main():
    cfg = Config()
    torch.manual_seed(cfg.SEED)
    # torch.backends.cudnn.benchmark = not cfg.NOT_CUDA_BENCHMARK

    # logger = Logger(cfg)

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    device = torch.device('cuda' if cfg.GPU[0] >= 0 else 'cpu')
    print('Creating model...')
    model = create_model(cfg)
    # model = load_model(model, 'log/weights/model_last.pth')
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
    # eval_loader = DataLoader(CircleDataset(cfg, 'eval'),
    #                          batch_size=cfg.BATCH_SIZE/2,
    #                          shuffle=True,
    #                          num_workers=cfg.NUM_WORKERS,
    #                          pin_memory=True)
    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, cfg.NUM_EPOCHS + 1):
        log_dict_train = trainer.train(train_loader)
        # logger.write('epoch: {} |'.format(epoch))
        # for k, v in log_dict_train.items():
        #     logger.scalar_summary('train_{}'.format(k), v, epoch)
        #     logger.write('{} {:8f} | '.format(k, v))
        # if cfg.VAL_INTERVALS > 0 and epoch % cfg.VAL_INTERVALS == 0:
        #     save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_{}.pth'.format(epoch)),
        #                epoch, model, optimizer)
        #     with torch.no_grad():
        #         log_dict_val, preds = trainer.val(eval_loader)
        #     # for k, v in log_dict_val.items():
        #     #     logger.scalar_summary('val_{}'.format(k), v, epoch)
        #     #     logger.write('{} {:8f} | '.format(k, v))
        #     if log_dict_val[cfg.METRIC] < best:
        #         best = log_dict_val[cfg.METRIC]
        #         save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_best.pth'),
        #                    epoch, model)
        # else:
        #     save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_last.pth'),
        #                epoch, model, optimizer)
        # logger.write('\n')
        # if epoch in cfg.LR_STEP:
        #     save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_{}.pth'.format(epoch)),
        #                epoch, model, optimizer)
        #     lr = cfg.LR * (0.1 ** (cfg.LR_STEP.index(epoch) + 1))
        #     print('Drop LR to', lr)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
    # logger.close()
    save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_last.pth'), epoch, model, optimizer)


def test_dataloader():
    cfg = Config()
    data_loader = DataLoader(CircleDataset(cfg, 'train'),
                             batch_size=cfg.BATCH_SIZE,
                             shuffle=True,
                             num_workers=cfg.NUM_WORKERS,
                             pin_memory=True,
                             drop_last=True)
    img, label = data_loader.dataset.__getitem__(1)
    print(img.shape)
    print(label.keys())  # dict_keys(['input', 'hm', 'reg_mask', 'ind', 'dense_wh', 'dense_wh_mask', 'reg'])
    for key in label.keys():
        print(label[key].shape)
        # (1, 384, 512)
        # (2, 96, 128)
        # (128,)
        # (128,)
        # (2, 96, 128)
        # (2, 96, 128)
        # (128, 2)
    # import matplotlib.pyplot as plt
    # plt.imshow(label['hm'][0])
    # plt.show()


if __name__ == '__main__':
    main()



