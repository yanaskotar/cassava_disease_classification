from efficientnet_pytorch import EfficientNet
from torch import nn

class EfficientNetModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = EfficientNet.from_name('efficientnet-b0')
        self.net._fc = nn.Linear(1280, n_classes)

    def forward(self, x):
        return self.net(x)