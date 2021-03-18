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

import torch.nn.functional as F
class Model(nn.Module):
  def __init__(self):
    super(Model,self).__init__()
    self.fc1 = nn.Linear(10,100)
    self.fc2 = nn.Linear(100,10)

  def forward(self,x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0")
    model = Model("tf_efficientnet_b0", 4).to(DEVICE)