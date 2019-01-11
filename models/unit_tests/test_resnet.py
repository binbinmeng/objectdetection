from backbones import ResNet
import torchsummary as summary

if __name__ == '__main__':
    model = ResNet(18)
    model =model.cuda()
    summary(model, (3, 224, 224))