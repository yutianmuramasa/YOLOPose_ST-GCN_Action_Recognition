import torch
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()

        self.data_bn = nn.BatchNorm1d(3 * 17)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):
        # x: [N, 3, T, 17]
        N, C, T, V = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()   # [N, T, 3, 17]
        x = x.view(N * T, C * V)
        x = self.data_bn(x)
        x = x.view(N, T, C, V).permute(0, 2, 1, 3)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool(x).view(N, -1)
        return self.fc(x)
