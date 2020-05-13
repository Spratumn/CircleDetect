import torch.nn as nn
from torchsummary import summary

from models.resnet import resnet18, resnet34, resnet50


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


class CircleNet(nn.Module):
    def __init__(self, backbone, num_class=2, head_conv=64):
        super(CircleNet, self).__init__()
        self.num_class = num_class
        self.backbone = backbone
        self.deconv_layer = self._make_deconv_layer()
        self.hm = make_head_layer(head_conv, self.num_class)
        self.wh = make_head_layer(head_conv, 2)
        self.reg = make_head_layer(head_conv, 2)

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
        reg = self.reg(x)
        rst = {'hm': hm, 'wh': wh, 'reg': reg}
        return rst


if __name__ == '__main__':
    re18 = resnet18()
    cn = CircleNet(re18)
    summary(cn, (3, 224, 224), device='cpu')
