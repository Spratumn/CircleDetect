import torch.nn as nn
from torchsummary import summary

from models.resnet import res_18, res_34
from models.littlenet import LitNet


def make_head_layer(head_conv, out_planes, arch):
    if arch == 'litnet':
        mid_planes = 128
    else:
        mid_planes = 256
    if head_conv > 0:
        head_layer = nn.Sequential(
            nn.Conv2d(mid_planes, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, out_planes,
                      kernel_size=1, stride=1, padding=0))
    else:
        head_layer = nn.Conv2d(
            in_channels=mid_planes,
            out_channels=out_planes,
            kernel_size=1,
            stride=1,
            padding=0
        )
    return head_layer


class CenterNet(nn.Module):
    def __init__(self, cfg, arch):
        super(CenterNet, self).__init__()
        backbone_factory = {'res_18': res_18(),
                            'res_34': res_34(),
                            'litnet': LitNet()}

        self.backbone = backbone_factory[arch]

        self.cfg = cfg
        self.arch = arch
        self.deconv_layer = self._make_deconv_layer()
        self.hm = make_head_layer(cfg.HEAD_CONV, cfg.NUM_CLASS, arch)
        self.wh = make_head_layer(cfg.HEAD_CONV, 2, arch)
        if cfg.USE_OFFSET:
            self.offset = make_head_layer(cfg.HEAD_CONV, 2, arch)

    def _make_deconv_layer(self):
        if self.arch == 'litnet':
            inplanes = 128
            mid_planes = 128
        else:
            inplanes = self.backbone.inplanes
            mid_planes = 256
        layers = []
        for i in range(3):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=mid_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False))
            layers.append(nn.BatchNorm2d(mid_planes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = mid_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        feature = self.deconv_layer(x)
        hm = self.hm(feature)
        wh = self.wh(feature)
        if self.cfg.USE_OFFSET:
            offset = self.offset(feature)
            rst = {'hm': hm, 'wh': wh, 'offset': offset}
            return rst
        rst = {'hm': hm, 'wh': wh}
        return rst


