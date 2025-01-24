import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FedResNet(nn.Module):
    def __init__(self, config):
        super(FedResNet, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        
        # Modify first conv layer for grayscale images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.bn1 = nn.BatchNorm2d(64)
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(self.model.fc.in_features)
        
        # Modified classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, config.NUM_CLASSES)
        )
        
        self.model.fc = nn.Identity()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.model(x)
        normalized_features = self.layer_norm(features)
        return self.classifier(normalized_features)
