import torch
import torch.nn as nn
import torch.onnx
from visualdl import LogWriter


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_units),  # Add BatchNorm here
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_units),  # Add BatchNorm here
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),  # Add BatchNorm here
            nn.ReLU(),
            nn.Conv2d(
                hidden_units, out_channels=hidden_units, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_units),  # Add BatchNorm here
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


logdir = "./log"
writer = LogWriter(logdir)

# Initialize the model
model = TinyVGG(input_shape=3, hidden_units=10, output_shape=5)

# Create a dummy input
dummy_input = torch.randn(1, 3, 64, 64)

# Log the model
writer.add_graph(model, dummy_input)
writer.close()
