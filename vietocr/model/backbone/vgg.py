import torch
from torch import nn
from torchvision import models
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter
import numpy as np
import math

class Step(nn.Module):
    def __init__(self, module, alpha=0.99, eps=1e-8):
        super(Step, self).__init__()
        self.x = None
        self.mod = module
        self.xxx = 0.5

    def forward(self, inp=None):
        try:
            self.xxx = self.xxx*0.9+0.1*(1-((self.x.grad*(inp-self.x))>0).sum().item()/np.prod(list(inp.size())))
        except: pass
        self.x = nn.Parameter(inp)
        out1 = self.mod(self.x)
        out2 = self.mod(inp)
        scale = math.sqrt(float(self.xxx))
        if random.random() < 0.0001:
            print(scale)
        return scale*out2+out1*(1-scale)


class Vgg(nn.Module):
    def __init__(self, name, ss, ks, hidden, pretrained=True, dropout=0.5):
        super(Vgg, self).__init__()

        if name == 'vgg11_bn':
            cnn = models.vgg11_bn(pretrained=pretrained)
        elif name == 'vgg19_bn':
            cnn = models.vgg19_bn(pretrained=pretrained)

        pool_idx = 0
        
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):        
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.Conv2d):        
                cnn.features[i] = Step(cnn.features[i])
 
        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = Step(nn.Conv2d(512, hidden, 1))

    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """

        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

#        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv

def vgg11_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg11_bn', ss, ks, hidden, pretrained, dropout)

def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg19_bn', ss, ks, hidden, pretrained, dropout)
   
