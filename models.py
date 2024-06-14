from torch import device, Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    Linear,
    ReLU,
    MaxPool2d,
    BatchNorm2d,
    Dropout,
)
from torch.utils.data import DataLoader

from torchviz import make_dot

from kan import KAN


def make_model_image(
    model: Module,
    loader: DataLoader,
    device: device,
    filename: str,
    format: str = "png",
):
    for x, y in loader:
        x: Tensor
        y: Tensor
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        y_pred: Tensor
        make_dot(y_pred.mean(), params=dict(model.named_parameters())).render(
            filename=filename, format=format, cleanup=True
        )
        break


class ConvNet(Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=81, out_channels=160, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(3, 3), stride=2),
            BatchNorm2d(160),
            ReLU(),
            Conv2d(in_channels=160, out_channels=240, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            BatchNorm2d(240),
            ReLU(),
            Conv2d(in_channels=240, out_channels=320, kernel_size=(5, 5), stride=1),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            BatchNorm2d(320),
        )
        self.fc = Sequential(
            Linear(in_features=320 * 113 * 155, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=10),
            ReLU(),
            Linear(in_features=10, out_features=10),
            Dropout(0.35),
            ReLU(),
            Linear(in_features=10, out_features=2),
            ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv(input)
        input = input.view(-1, 320 * 113 * 155)
        input = self.fc(input)
        return input


class ConvNet2(Module):
    def __init__(self) -> None:
        super(ConvNet2, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=81, out_channels=160, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(3, 3), stride=2),
            BatchNorm2d(160),
            ReLU(),
            Conv2d(in_channels=160, out_channels=240, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            BatchNorm2d(240),
            ReLU(),
            Conv2d(in_channels=240, out_channels=320, kernel_size=(5, 5), stride=1),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            BatchNorm2d(320),
        )
        self.fc = Sequential(
            Linear(in_features=320 * 113 * 155, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=10),
            ReLU(),
            Linear(in_features=10, out_features=10),
            ReLU(),
            Linear(in_features=10, out_features=2),
            ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv(input)
        input = input.view(-1, 320 * 113 * 155)
        input = self.fc(input)
        return input


class ConvNet3(Module):
    def __init__(self) -> None:
        super(ConvNet3, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=81, out_channels=160, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(3, 3), stride=2),
            ReLU(),
            Conv2d(in_channels=160, out_channels=240, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            ReLU(),
            Conv2d(in_channels=240, out_channels=320, kernel_size=(5, 5), stride=1),
            MaxPool2d(kernel_size=(5, 5), stride=1),
        )
        self.fc = Sequential(
            Linear(in_features=320 * 113 * 155, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=10),
            ReLU(),
            Linear(in_features=10, out_features=10),
            ReLU(),
            Linear(in_features=10, out_features=2),
            ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv(input)
        input = input.view(-1, 320 * 113 * 155)
        input = self.fc(input)
        return input


class KanNet(Module):
    def __init__(self) -> None:
        super(KanNet, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=81, out_channels=160, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(3, 3), stride=2),
            BatchNorm2d(160),
            ReLU(),
            Conv2d(in_channels=160, out_channels=240, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            BatchNorm2d(240),
            ReLU(),
            Conv2d(in_channels=240, out_channels=320, kernel_size=(5, 5), stride=1),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            BatchNorm2d(320),
            ReLU(),
            Conv2d(in_channels=320, out_channels=400, kernel_size=(10, 10), stride=3),
            MaxPool2d(kernel_size=(10, 10), stride=3),
            BatchNorm2d(400),
        )

        self.kan = KAN(
            [400 * 9 * 14, 10, 2],
            grid=6,
            device="cuda:0",
            symbolic_enabled=False,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv(input)
        input = input.view(-1, input.shape[1] * input.shape[2] * input.shape[3])
        input = self.kan(input)
        return input


class KanNet2(Module):
    def __init__(self) -> None:
        super(KanNet2, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=81, out_channels=160, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(3, 3), stride=2),
            ReLU(),
            Conv2d(in_channels=160, out_channels=240, kernel_size=(5, 5), stride=2),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            ReLU(),
            Conv2d(in_channels=240, out_channels=320, kernel_size=(5, 5), stride=1),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            ReLU(),
            Conv2d(in_channels=320, out_channels=400, kernel_size=(10, 10), stride=3),
            MaxPool2d(kernel_size=(10, 10), stride=3),
        )

        self.kan = KAN(
            [400 * 9 * 14, 10, 2],
            grid=6,
            device="cuda:0",
            symbolic_enabled=False,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv(input)
        input = input.view(-1, input.shape[1] * input.shape[2] * input.shape[3])
        input = self.kan(input)
        return input


class KanNet3(Module):
    def __init__(self) -> None:
        super(KanNet3, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=81, out_channels=160, kernel_size=(5, 5), stride=1),
            MaxPool2d(kernel_size=(3, 3), stride=2),
            ReLU(),
            Conv2d(in_channels=160, out_channels=240, kernel_size=(5, 5), stride=1),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            ReLU(),
            Conv2d(in_channels=240, out_channels=320, kernel_size=(5, 5), stride=1),
            MaxPool2d(kernel_size=(5, 5), stride=1),
            ReLU(),
            Conv2d(in_channels=320, out_channels=400, kernel_size=(5, 5), stride=3),
            MaxPool2d(kernel_size=(5, 5), stride=3),
            ReLU(),
            Conv2d(in_channels=400, out_channels=480, kernel_size=(5, 5), stride=3),
            MaxPool2d(kernel_size=(5, 5), stride=3),
        )

        self.kan = KAN(
            [480 * 5 * 7, 20, 2],
            grid=10,
            device="cuda:0",
            symbolic_enabled=False,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv(input)
        input = input.view(-1, input.shape[1] * input.shape[2] * input.shape[3])
        input = self.kan(input)
        return input
