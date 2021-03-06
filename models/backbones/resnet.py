import torch.nn as nn
from weights_init import constant_init, kaiming_init
import torch.utils.model_zoo as model_zoo



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,inplanes,planes,stride=1,dilation=1,downsample=None,style='pytorch',):
        super(BasicBlock, self).__init__()

        def conv3x3(in_planes, out_planes, stride=1, dilation=1):
            "3x3 convolution with padding"
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, dilation=dilation,
                             bias=False)

        def conv1x1(in_planes, out_planes, stride=1):
            """1x1 convolution"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes, planes,stride=1,dilation=1,downsample=None,style='pytorch'):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = conv1x1(inplanes,planes, stride=self.conv1_stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=self.conv2_stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation


    def forward(self, x):

       identity = x

       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)

       out = self.conv2(out)
       out = self.bn2(out)
       out = self.relu(out)

       out = self.conv3(out)
       out = self.bn3(out)

       if self.downsample is not None:
           identity = self.downsample(x)

       out += identity
       out = self.relu(out)
       return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    model_urls = {
        18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 num_classes=1000,
                 pretrained=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))

        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.num_classes =num_classes
        self.pretrained = pretrained
        self.pretrained_url = self.model_urls[self.depth]

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        #first several common layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            print(planes)
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1) #bloack_names: layer1,layer2,layer3,layer4
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(len(self.stage_blocks) - 1)#resnet output dimension
        print(self.feat_dim)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            if pretrained:
                if  self.pretrained_url.startswith(('http://', 'https://')):
                    checkpoint = model_zoo.load_url(self.pretrained_url)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


if __name__ == '__main__':
    model = ResNet(152)
    from torchsummary import summary
    summary(model.cuda(),(3,224,224))