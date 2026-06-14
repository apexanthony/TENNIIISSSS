import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNetV3(nn.Module):
    """TrackNet V3: low-resolution depthwise heatmap model for RK3588 throughput."""

    def __init__(self, base_channels=24, out_channels=1):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = DSConvBlock(9, c1)
        self.enc1 = DSConvBlock(c1, c1)

        self.down1 = DSConvBlock(c1, c2, stride=2)
        self.enc2 = DSConvBlock(c2, c2)

        self.down2 = DSConvBlock(c2, c3, stride=2)
        self.bottleneck1 = DSConvBlock(c3, c3)
        self.bottleneck2 = DSConvBlock(c3, c3)

        self.dec2 = DSConvBlock(c3 + c2, c2)
        self.dec1 = DSConvBlock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_weights()

    def forward(self, x, testing=False):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        x = self.bottleneck2(self.bottleneck1(self.down2(e2)))

        x = F.interpolate(x, size=e2.shape[-2:], mode="nearest")
        x = self.dec2(torch.cat([x, e2], dim=1))

        x = F.interpolate(x, size=e1.shape[-2:], mode="nearest")
        x = self.dec1(torch.cat([x, e1], dim=1))
        x = self.head(x)

        if testing:
            x = torch.sigmoid(x)
        return x

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    model = BallTrackerNetV3()
    inp = torch.rand(1, 9, 180, 320)
    out = model(inp)
    print("out = {}".format(out.shape))
