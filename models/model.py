import torch

from models.resnet import resnet18, resnet34, resnet50
from models.circlenet import CircleNet


def create_model(cfg):
    backbone = resnet18()
    model = CircleNet(backbone, num_class=cfg.NUM_CLASS, head_conv=cfg.HEAD_CONV)
    return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


if __name__ == '__main__':
    from torchsummary import summary
    backbone = resnet18()
    model = CircleNet(backbone, num_class=1, head_conv=64)
    summary(model, (3, 512, 512), device='cpu')
