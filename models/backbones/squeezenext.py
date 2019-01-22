import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        reduction = 0.5
        if 2 == stride:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
            
        self.conv1 = nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=True)
        self.bn1   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv2 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=True)
        self.bn2   = nn.BatchNorm2d(int(in_channels * reduction * 0.5))
        self.conv3 = nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=True)
        self.bn3   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv4 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=True)
        self.bn4   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv5 = nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=True)
        self.bn5   = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if 2 == stride or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, stride, bias=True),
                            nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        #print(output.shape)
        output = F.relu(self.bn2(self.conv2(output)))
        #print(output.shape)
        output = F.relu(self.bn3(self.conv3(output)))
        #print(output.shape)
        output = F.relu(self.bn4(self.conv4(output)))
        #print(output.shape)
        output = F.relu(self.bn5(self.conv5(output)))
        #print(output.shape)
        output += F.relu(self.shortcut(input))
        output = F.relu(output)
        return output

class SqueezeNext(nn.Module):
    def __init__(self, width_x, blocks, num_classes):
        super(SqueezeNext, self).__init__()
        self.in_channels = 64
        
        #self.conv1  = nn.Conv2d(3, int(width_x * self.in_channels), 3, 1, 1, bias=True)     # For Cifar10
        self.conv1  = nn.Conv2d(3, int(width_x * self.in_channels), 3, 2, 1, bias=True)     # For Tiny-ImageNet
        self.bn1    = nn.BatchNorm2d(int(width_x * self.in_channels))
        self.stage1 = self._make_layer(blocks[0], width_x, 32, 1)
        self.stage2 = self._make_layer(blocks[1], width_x, 64, 2)
        self.stage3 = self._make_layer(blocks[2], width_x, 128, 2)
        self.stage4 = self._make_layer(blocks[3], width_x, 256, 2)
        self.conv2  = nn.Conv2d(int(width_x * self.in_channels), int(width_x * 128), 1, 1, bias=True)
        self.bn2    = nn.BatchNorm2d(int(width_x * 128))
        self.linear = nn.Linear(int(width_x * 128), num_classes)
        
    def _make_layer(self, num_block, width_x, out_channels, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers  = []
        for _stride in strides:
            layers.append(BasicBlock(int(width_x * self.in_channels), int(width_x * out_channels), _stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.max_pool2d(output,3, 2) # 55x55x64
        output = self.stage1(output)
        #print(output.shape)
        output = self.stage2(output)
        #print(output.shape)
        output = self.stage3(output)
        #print(output.shape)
        output = self.stage4(output)
        #print(output.shape)
        output = F.relu(self.bn2(self.conv2(output)))
        #print(output.shape)
        output = F.avg_pool2d(output, 7)
        #print("last:",output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        #print("fc:",output.shape)
        return output

def speed(model, name, inputX, inputY):
    import time
    t0 = time.time()
    input = torch.rand(1,3,inputX, inputY).cuda()
    input = Variable(input, volatile = True)
    t1 = time.time()

    out = model(input)
    t2 = time.time()
    
    print("=> output size = {}".format(out.size()))
    print('=> {} cost: {}'.format(name, t2 - t1))

def SqNxt_23_1x(num_classes):
    return SqueezeNext(1.0, [6, 6, 8, 1], num_classes)

def SqNxt_23_1x_v5(num_classes):
    return SqueezeNext(1.0, [2, 4, 14, 1], num_classes)

def SqNxt_23_2x(num_classes):
    return SqueezeNext(2.0, [6, 6, 8, 1], num_classes)

def SqNxt_23_2x_v5(num_classes):
    return SqueezeNext(2.0, [2, 4, 14, 1], num_classes)

if __name__ == '__main__':
    model =  SqNxt_23_1x_v5(1000)
    from torchsummary import summary
    summary(model.cuda(), (3, 224, 224))
    speed(model.cuda(), 'SqNxt_23_1x_v5', 224, 224) 
    
