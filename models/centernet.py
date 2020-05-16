import torch.nn as nn
from torchsummary import summary

from models.resnet import res_18, res_34, res_50


def make_head_layer(head_conv, out_planes):
    if head_conv > 0:
        head_layer = nn.Sequential(
            nn.Conv2d(256, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, out_planes,
                      kernel_size=1, stride=1, padding=0))
    else:
        head_layer = nn.Conv2d(
            in_channels=256,
            out_channels=out_planes,
            kernel_size=1,
            stride=1,
            padding=0
        )
    return head_layer


class CenterNet(nn.Module):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        backbone_factory = {'res_18': res_18(),
                            'res_34': res_34(),
                            'res_50': res_50()}
        self.backbone = backbone_factory[cfg.ARCH]
        self.deconv_layer = self._make_deconv_layer()
        self.hm = make_head_layer(cfg.HEAD_CONV, cfg.NUM_CLASS)
        self.wh = make_head_layer(cfg.HEAD_CONV, 2)
        if cfg.USE_OFFSET:
            self.offset = make_head_layer(cfg.HEAD_CONV, 2)
        self.cfg = cfg

    def _make_deconv_layer(self):
        inplanes = self.backbone.inplanes
        layers = []
        for i in range(3):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            inplanes = 256
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layer(x)
        hm = self.hm(x)
        wh = self.wh(x)
        if self.cfg.USE_OFFSET:
            offset = self.offset(x)
            rst = {'hm': hm, 'wh': wh, 'offset': offset}
            return rst
        rst = {'hm': hm, 'wh': wh}
        return rst


