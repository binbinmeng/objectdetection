from backbones import *
from torchsummary import summary

if __name__ == '__main__':
    model =ResNet(152)
    model =model.cuda()
    summary(model, (3,224,224))
