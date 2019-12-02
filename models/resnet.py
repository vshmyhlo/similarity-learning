import torch.nn as nn
import torchvision.models


class ResNet(nn.Module):
    def __init__(self, type):
        super().__init__()

        if type == 18:
            self.model = torchvision.models.resnet18(pretrained=True)
        elif type == 34:
            self.model = torchvision.models.resnet34(pretrained=True)
        else:
            fail

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(512, 256)

    def forward(self, input):
        input = self.model.conv1(input)
        input = self.model.bn1(input)
        input = self.model.relu(input)
        input = self.model.maxpool(input)

        input = self.model.layer1(input)
        input = self.model.layer2(input)
        input = self.model.layer3(input)
        input = self.model.layer4(input)

        input = self.model.avgpool(input)
        input = input.view(input.size(0), input.size(1))
        # input = self.model.fc(input)

        input = self.output(input)

        return input
