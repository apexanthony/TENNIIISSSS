import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v3 import DSConvBlock


class BallTrackerNetV34MultiTask(nn.Module):
    """V3.4: lightweight multi-task TrackNet with contextual auxiliary heads."""

    def __init__(self, base_channels=24, status_classes=3):
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

        self.ball_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.player_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.court_head = nn.Conv2d(c1, 14, kernel_size=1)
        self.status_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, c2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(c2, status_classes),
        )

        self._init_weights()

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        bottleneck = self.bottleneck2(self.bottleneck1(self.down2(e2)))

        x = F.interpolate(bottleneck, size=e2.shape[-2:], mode="nearest")
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="nearest")
        features = self.dec1(torch.cat([x, e1], dim=1))

        return {
            "ball": self.ball_head(features),
            "player": self.player_head(features),
            "court": self.court_head(features),
            "status": self.status_head(bottleneck),
        }

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    model = BallTrackerNetV34MultiTask()
    out = model(torch.rand(2, 9, 270, 480))
    print({key: value.shape for key, value in out.items()})
