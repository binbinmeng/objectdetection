
from backbones import *
from torchsummary import summary

if __name__ == '__main__':
    #model =MobileNetV2()
    model = MobileNetV2(Bottleneck, [1, 2, 3, 4, 3, 3, 1]
    model = model.cuda()
    summary(model, (3,224,224))
