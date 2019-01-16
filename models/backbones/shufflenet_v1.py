import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N, C, H, W] -> [N, g, C/g, H, W] -> [N, c/g, g, H, W] -> [N, C, H, W]"""
        N, C, H, W = x.size()

        g = self.groups
        return x.view(N, g, C/g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes / 4
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        res = self.shortcut(x)
        out = self.relu(torch.cat((out, res), 1)) if self.stride == 2 else self.relu(out + res)
        return out

class ShuffleNet(nn.Module):
    def __init__(self, cfg):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.linear1 = nn.Linear(out_planes[2], 10)
        # self.linear2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU(True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant(m.weight, 1)
        #         nn.init.constant(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal(m.weight, std=0.01)
        #         nn.init.constant(m.bias, 0)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

    def name(self):
        return 'ShuffleNet'

def shuffleNetG1():
    cfg = {
        'out_planes': [144, 288, 576],
        'num_blocks': [4, 8, 4],
        'groups': 1
    }
    return ShuffleNet(cfg)

def shuffleNetG2():
    cfg = {
        'out_planes': [200, 400, 800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
    return ShuffleNet(cfg)

def shuffleNetG3():
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups': 3
    }
    return ShuffleNet(cfg)

def shuffleNetG4():
    cfg = {
        'out_planes': [272, 544, 1088],
        'num_blocks': [4, 8, 4],
        'groups': 4
    }
    return ShuffleNet(cfg)

def shuffleNetG8():
    cfg = {
        'out_planes': [384, 768, 1536],
        'num_blocks': [4, 8, 4],
        'groups': 8
    }
    return ShuffleNet(cfg)

def shufflenet(groups):
    if groups == 1:
        return shuffleNetG1()
    elif groups == 2:
        return shuffleNetG2()
    elif groups == 3:
        return shuffleNetG3()
    elif groups == 4:
        return shuffleNetG4()
    elif groups == 8:
        return shuffleNetG8()

def test():
    net = shuffleNetG3()
    print(net)
    x = Variable(torch.randn(1, 3, 32, 32))
    y = net(x)
    print(y.size())


if __name__ =='__main__':
   model =shuffleNetG2()
   from torchsummary import summary
   summary(model,(3,224,224),1,"cpu")