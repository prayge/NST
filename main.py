from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
from init import Config
from net import VGGNet
from utils import load_image, printf

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(cfg):
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])
    
    content = load_image(cfg.contentimage, transform, max_size=cfg.max_size)
    style = load_image(cfg.styleimage, transform, shape=[content.size(2), content.size(3)])
    
    target = content.clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([target], lr=cfg.lr, betas=[0.5, 0.999])
    vgg = VGGNet().to(device).eval()
    
    for step in range(cfg.steps):

        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)

        style_loss = 0
        content_loss = 0
        for targetR, contentR, styleR in zip(target_features, content_features, style_features):

            content_loss += torch.mean((targetR - contentR)**2)

            _, c, h, w = targetR.size()
            targetR = targetR.view(c, h * w)
            styleR = styleR.view(c, h * w)

            targetR = torch.mm(targetR, targetR.t())
            styleR = torch.mm(styleR, styleR.t())

            style_loss += torch.mean((targetR - styleR)**2) / (c * h * w) 

        loss = content_loss + cfg.style_weight * style_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % cfg.log_step == 0:
            printf('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}' 
                   .format(step+1, cfg.steps, content_loss.item(), style_loss.item()))
        
        if (step+1) % cfg.sample_step == 0:

            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-{}.png'.format(step+1))


if __name__ == "__main__":
    cfg = Config().parse()
    print(cfg)
    main(cfg)