import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each followed by BN and ReLU."""

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
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CoordinateAttention(nn.Module):
    """Standard bottleneck Coordinate Attention with export-friendly means.

    For F in [B,C,H,W], the module preserves H and W separately, shares a
    channel-reduction transform, then applies directional sigmoid gates.
    """

    def __init__(self, channels, reduction=32, min_channels=8):
        super().__init__()
        hidden = max(int(min_channels), int(channels) // int(reduction))
        self.shared = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)
        self.attn_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.attn_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        height = x.shape[2]
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        shared = self.act(self.bn(self.shared(torch.cat([x_h, x_w], dim=2))))
        feature_h, feature_w = torch.split(shared, [height, x.shape[3]], dim=2)
        feature_w = feature_w.transpose(2, 3)
        gate_h = torch.sigmoid(self.attn_h(feature_h))
        gate_w = torch.sigmoid(self.attn_w(feature_w))
        return x * gate_h * gate_w


class BallTrackerNetV38(nn.Module):
    """V3.7 lightweight U-Net backbone with optional bottleneck CA.

    The network always returns logits. Sigmoid belongs to probability
    decoding/inference, while BCEWithLogits consumes the logits directly.
    """

    def __init__(
        self,
        base_channels=24,
        out_channels=1,
        use_ca=False,
        ca_reduction=32,
        ca_min_channels=8,
        initialization_seed=None,
    ):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        self.use_ca = bool(use_ca)

        self.stem = DSConvBlock(9, c1)
        self.enc1 = DSConvBlock(c1, c1)
        self.down1 = DSConvBlock(c1, c2, stride=2)
        self.enc2 = DSConvBlock(c2, c2)
        self.down2 = DSConvBlock(c2, c3, stride=2)
        self.bottleneck1 = DSConvBlock(c3, c3)
        self.bottleneck2 = DSConvBlock(c3, c3)
        self.dec2 = DSConvBlock(c3 + c2, c2)
        self.dec1 = DSConvBlock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1, bias=True)
        # Register CA after every shared layer so a fixed seed gives identical
        # backbone/decoder initialization with and without the optional block.
        self.ca = (
            CoordinateAttention(c3, reduction=ca_reduction, min_channels=ca_min_channels)
            if self.use_ca
            else nn.Identity()
        )
        if initialization_seed is None:
            self._init_weights()
        else:
            # Layer constructors consume RNG state. Reset it locally before
            # explicit initialization so all shared tensors are bit-identical
            # across ablation variants built with the same paper seed.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(initialization_seed))
                self._init_weights()

    def forward(self, x, testing=False):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        x = self.bottleneck2(self.bottleneck1(self.down2(e2)))
        x = self.ca(x)

        x = F.interpolate(x, size=e2.shape[-2:], mode="nearest")
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="nearest")
        x = self.dec1(torch.cat([x, e1], dim=1))
        logits = self.head(x)
        return torch.sigmoid(logits) if testing else logits

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
    for enabled in (False, True):
        model = BallTrackerNetV38(use_ca=enabled)
        sample = torch.rand(1, 9, 360, 640)
        output = model(sample)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"use_ca={enabled}, parameters={parameters}, output={tuple(output.shape)}")
