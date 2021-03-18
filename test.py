from torch import nn
import torch
import timm

class FlowerImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0")
    model = FlowerImgClassifier("cuda:0", 4, pretrained=True).to(DEVICE)