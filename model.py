import torch
import torch.nn as nn
from torchvision.models import resnet50
from options import parse_arguments

args = parse_arguments()

class TinyVgg(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * int(args.load_size / 8) * int(args.load_size / 8), output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.classifier(x)
        return x

class ImageNetModel(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        # self.weights=resnet50.DEFAULT
        self.model = resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, output_shape)

    def forward(self, x):
        return self.model(x)

def create_model(input_shape, hidden_units, output_shape):
    if args.model == "ImageNet":
        model = ImageNetModel(output_shape)
    else:
        model = TinyVgg(input_shape, hidden_units, output_shape)
    return model
