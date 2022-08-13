from __future__ import absolute_import

import torchvision
import torch
import os.path as osp
import torch.nn as nn
from torch.nn import init

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=False):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        model_path = osp.join(osp.dirname(osp.abspath(__file__)), 'resnet50-19c8e357.pth')
        resnet.load_state_dict(torch.load(model_path), strict=False)

        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, feature_withbn=False):
        x = self.base(x)
        x = self.gap(x)
        return x

def resnet18(**kwargs):
    return ResNet(18, **kwargs)

def resnet34(**kwargs):
    return ResNet(34, **kwargs)

def resnet50(**kwargs):
    return ResNet(50, **kwargs)

def resnet101(**kwargs):
    return ResNet(101, **kwargs)

def resnet152(**kwargs):
    return ResNet(152, **kwargs)

class Model(nn.Module):

    def __init__(self, source_classes, target_images):
        super(Model, self).__init__()
        self.backbone = resnet50()
        self.classifier = nn.Linear(2048, source_classes+target_images,bias=False)
        self.feat_bn = nn.BatchNorm1d(2048)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

    def forward(self, x):
        if self.training == True:
            x = self.backbone(x)
            x = x.view(x.size(0),-1)
            x_af_bn = self.feat_bn(x)
            precs = self.classifier(x_af_bn)
            return precs, x, x_af_bn
        else:
            x = self.backbone(x)
            x = x.view(x.size(0),-1)
            x_af = self.feat_bn(x)
            return x,x_af

class DSBN2d(nn.Module):
    def __init__(self, planes):
        super(DSBN2d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out

class DSBN1d(nn.Module):
    def __init__(self, planes):
        super(DSBN1d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out

def convert_dsbn(model):
    for _, (child_name, child) in enumerate(model.named_children()):
        # assert(not next(model.parameters()).is_cuda)
        if isinstance(child, nn.BatchNorm2d):
            m = DSBN2d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm1d):
            m = DSBN1d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        else:
            convert_dsbn(child)

def convert_bn(model, use_target=True):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert(not next(model.parameters()).is_cuda)
        if isinstance(child, DSBN2d):
            m = nn.BatchNorm2d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN1d):
            m = nn.BatchNorm1d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        else:
            convert_bn(child, use_target=use_target)
