import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class ArcFace(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False):
        super().__init__()

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        if label is None:
            return x

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


class ResNet(nn.Module):
    def __init__(self, type, num_classes):
        super().__init__()

        if type == 18:
            self.model = torchvision.models.resnet18(pretrained=True)
        elif type == 34:
            self.model = torchvision.models.resnet34(pretrained=True)
        else:
            fail

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(512, 256)
        self.logits = nn.Linear(256, num_classes)

    def forward(self, input, ids=None):
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
        logits = self.logits(input)

        return input, logits
