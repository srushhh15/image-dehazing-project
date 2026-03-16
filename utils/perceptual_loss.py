import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(pretrained=True).features[:16]
        self.vgg = vgg.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, pred, target):

        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        loss = self.criterion(pred_features, target_features)

        return loss