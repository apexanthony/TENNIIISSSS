import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNetLite(nn.Module):
    """TrackNet V2 for RK3588: half-width encoder-decoder with 1-channel heatmap output."""

    def __init__(self, base_channels=32, out_channels=1):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.conv1 = ConvBlock(9, c1)
        self.conv2 = ConvBlock(c1, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(c1, c2)
        self.conv4 = ConvBlock(c2, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(c2, c3)
        self.conv6 = ConvBlock(c3, c3)
        self.conv7 = ConvBlock(c3, c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = ConvBlock(c3, c4)
        self.conv9 = ConvBlock(c4, c4)
        self.conv10 = ConvBlock(c4, c4)

        self.ups1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv11 = ConvBlock(c4, c3)
        self.conv12 = ConvBlock(c3, c3)
        self.conv13 = ConvBlock(c3, c3)

        self.ups2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv14 = ConvBlock(c3, c2)
        self.conv15 = ConvBlock(c2, c2)

        self.ups3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv16 = ConvBlock(c2, c1)
        self.conv17 = ConvBlock(c1, c1)
        self.conv18 = nn.Conv2d(c1, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_weights()

    def forward(self, x, testing=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)

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
    device = "cpu"
    model = BallTrackerNetLite().to(device)
    inp = torch.rand(1, 9, 360, 640)
    out = model(inp)
    print("out = {}".format(out.shape))
