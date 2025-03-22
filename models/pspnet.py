import torch
from torch import nn
from torch.nn import functional as F

import models.resnet as extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))

        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages]
        priors=priors+[feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


def get_backbone(in_channels, backbone_type='resnet101'):
    """
    获取PSP-NET的Backbone
    :param in_channels: 输出channels也就是图像的channels
    :param backbone_type: 推荐使用ResNet101或Xception作为DeepLabV3+的Backbone
    :return: 返回backbone，主干特征channels，low-level特征channels
    """
    if backbone_type == 'resnet50':
        backbone = resnet50_atrous(in_channels=in_channels)
        atrous_channels = 2048
        low_level_channels = 256
    elif backbone_type == 'resnet101':
        backbone = resnet101_atrous(in_channels=in_channels)
        atrous_channels = 2048
        low_level_channels = 256
    # elif backbone_type == 'xception':
    #     backbone = xception_backbone(in_channels=in_channels)
    #     atrous_channels = 2048
    #     low_level_channels = 128
    else:
        raise ValueError('backbone type error!')
    return backbone, atrous_channels, low_level_channels



class PSPNet(nn.Module):
    def __init__(self, n_classes=10, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024,
                 backend="resnet101",pretrained=False):
        super().__init__()
        self.feats,_,_ = get_backbone(3,'resnet101')
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f, _ = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)
        feature = p

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        # print("p_final",p.shape)
        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        # print("au",auxiliary.shape)
        p = self.final(p)
        auxiliary = self.classifier(auxiliary)
        if p.size(2) != x.size(2):
            p = F.interpolate(p, size=(x.size(2), x.size(3)))
        return p, feature



def conv_3x3(in_channels, out_channels, stride=1, dilation=1):
    """
    3x3 same 卷积
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param stride: 下采样率。默认stride=1，不下采样；stride=2，下采样2倍
    :param dilation: Atros Conv的rate
    :return: 3x3 same 卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)  # 后面接bn，bias=False;same卷积padding=1


def conv_1x1(in_channels, out_channels, stride=1):
    """
    用于调整维度
    layer调整channel
    project调整channel，spatial
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param stride: 下采样率。默认stride=1，不下采样；stride=2，下采样2倍
    :return: 1x1 卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     bias=False)  # 后面接bn，bias=False


class BasicBlock(nn.Module):
    expansion = 1  # Basic Block 最后一个卷积输出channels是plane的1倍

    def __init__(self, inplanes, planes, stride=1, dilation=1, batch_norm=None,
                 project=None):
        """
        Basic Block 每个block有2个3x3卷积，两个卷积的输出通道数相同，等于planes。
        :param inplanes: 这个basic block的输入通道数，前一个basic block输出通道数。
        :param planes: 两个卷积的输出通道数相同，等于planes。取值64,128,256,512。
        :param stride: stride=1，不下采样；
                       stride=2，第一个卷积下采样；
        :param batch_norm: 外部指定bn，不指定就用默认bn。
        :param project: 外部指定project也就是残差中的+x方法。
        """
        super(BasicBlock, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d  # 外部不指定bn就使用默认bn

        # 第一个3x3卷积，论文中conv2_x不下采样，conv3_x-conv5_x下采样
        self.conv1 = conv_3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = batch_norm(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个3x3卷积，都不下采样
        self.conv2 = conv_3x3(planes, planes, dilation=dilation)
        self.bn2 = batch_norm(planes)

        # 维数一致才能相加， +x 或者 +project(x)
        self.project = project
        pass

    def forward(self, x):
        identity = x  # 记录下输入x

        # 第一个卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积
        out = self.conv2(out)
        out = self.bn2(out)

        # 维数一致才能相加
        # 判断+identity，还是+project(x)
        if self.project is not None:
            identity = self.project(x)
        out += identity  # 残差+
        out = self.relu(out)  # 加上以后再relu
        return out

    pass


class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck最后一个卷积输出channels是planes的4倍

    def __init__(self, inplanes, planes, stride=1, dilation=1, batch_norm=None,
                 project=None):
        """
        Bottleneck
        每个block有3个卷积：1x1降低channel；3x3卷积下采样或不下采样；1x1升高channels；减小计算量。
        前两个卷积的输出channels相同，都等于planes
        最后一个卷积的输出channels是planes的4倍
        :param inplanes: 这个basic block的输入通道数，前一个basic block输出通道数。
        :param planes: 前两个卷积的输出通道数相同，等于planes。取值64,128,256,512。
        :param stride: stride=1，不下采样；
                       stride=2，第一个卷积下采样；
        :param batch_norm: 外部指定bn，不指定就用默认bn。
        :param project: 外部指定project也就是残差中的+x方法。
        """
        super(Bottleneck, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d  # 外部不指定bn就使用默认bn

        # 第一个1x1卷积，降低channels
        self.conv1 = conv_1x1(inplanes, planes)
        self.bn1 = batch_norm(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个3x3卷积，下采样或不下采样
        self.conv2 = conv_3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = batch_norm(planes)

        # 第三个1x1卷积，升高channels到planes的4倍
        self.conv3 = conv_1x1(planes, self.expansion * planes)
        self.bn3 = batch_norm(self.expansion * planes)

        # 维数一致才能相加，+x 或者 +project(x)
        self.project = project
        pass

    def forward(self, x):
        identity = x  # 记录下输入x

        # 第一个1x1卷积，降低channels
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个3x3卷积，下采样或不下采样
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三个1x1卷积，升高channels到planes的4倍
        out = self.conv3(out)
        out = self.bn3(out)

        # 维数一致才能相加
        # +x 或者 +project(x)
        if self.project is not None:
            identity = self.project(x)
        out += identity  # 残差+
        out = self.relu(out)  # 加上以后再relu
        return out

    pass


class ResNetBackBone(nn.Module):
    def __init__(self, block, layers, in_channels=3, batch_norm=None):
        """
        ResNet 18/34/50/101/152
        :param block: ResNet 18/34 用Basic Block
                      ResNet 50/101/152 用Bottleneck
        :param layers: 每种ResNet各个layer中block的数量。
                       取列表前4个数字，依次代表论文Conv2_x至Conv5_x中block的数量
        :param in_channels: 模型输入默认是3通道的
        :param batch_norm: 外部指定bn
        """
        super(ResNetBackBone, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d  # 没有外部指定bn就用默认bn
        self._batch_norm = batch_norm

        self.inplanes = 64  # 各个layer输出通道数，conv1输出64通道，后面再make_layer中更新

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)  # 后面接bn不要bias
        self.bn1 = self._batch_norm(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
        # self._initialize_weights()
        # self.layer5 = self._make_layer(block, layers[3], 512, stride=1,
        #                                dilation=2)  # DeepLabV3中3.2. Going Deeper with Atrous Convolution
        # self.layer6 = self._make_layer(block, layers[3], 512, stride=1, dilation=2)
        pass

    def _make_layer(self, block, n_block, planes, stride=1, dilation=1):
        """
        构造layer1-layer3，也就是论文中的Conv2_x-Conv4_x
        :param block: ResNet 18/34 用Basic Block
                      ResNet 50/101/152 用Bottleneck
        :param n_block: 本层block数量
        :param planes: 本层的基准channels数
        :param stride: stride=1，不下采样；
                       stride=2，第一个卷积下采样；
        :return:
        """
        batch_norm = self._batch_norm

        # 第一个block考虑设置project
        project = None
        if stride != 1 or self.inplanes != block.expansion * planes:
            # stride!=1 下采样，调整spatial
            # block输入channels和输出channels不一致，调整channels=block.expansion * planes
            project = nn.Sequential(
                conv_1x1(self.inplanes, block.expansion * planes,
                         stride=stride),
                batch_norm(block.expansion * planes)  # 调整维数后bn
            )

        # 第一个block考虑是否下采样，单独设置
        layer = [block(self.inplanes, planes, stride=stride, dilation=dilation,
                       batch_norm=batch_norm, project=project)]

        self.inplanes = block.expansion * planes  # 后面几个block输入channel

        # 其余block一样，都不进行下采样，循环添加
        for _ in range(1, n_block):  # 第一个block单独设置了，所以range从1开始
            layer.append(block(self.inplanes, planes, dilation=dilation,
                               batch_norm=batch_norm))
            pass

        return nn.Sequential(*layer)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data,nonlinearity="relu")
    #             if m.bias is not None:
    #                 m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)  # 1/2
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool(x)  # 1/4
        x = self.layer1(x)


        x = self.layer2(x)  # 1/8
        p = x

        x = self.layer3(x)  # 1/16
        low_level_features = x
        x = self.layer4(x)  # 1/32
        # x = self.layer5(x)
        # x = self.layer6(x)

        return x, low_level_features, p  # 输出主干特征x和low-level特征

    pass


def resnet50_atrous(in_channels=3, batch_norm=None):
    return ResNetBackBone(Bottleneck, [3, 4, 6, 3], in_channels=in_channels,
                          batch_norm=batch_norm)


def resnet101_atrous(in_channels=3, batch_norm=None):
    model = ResNetBackBone(Bottleneck, [3, 4, 23, 3], in_channels=in_channels,
                          batch_norm=batch_norm)
    state_dict = torch.load(r"/home/ljk/image_captioning/models/resnet101-5d3b4d8f.pth")
    op = model.state_dict()
    # for op_num, op_key in enumerate(op.keys()):
    #     if "num_batches_tracked" in op_key:
    #         continue
    #     if "project" in op_key:
    #         op_key = op_key.replace("project", "downsample")
    items = state_dict
    for new_state_dict_num, new_state_dict_key in enumerate(state_dict.keys()):
        if new_state_dict_key in op.keys():

            op[new_state_dict_key] = items.setdefault(new_state_dict_key)
        elif "downsample" in new_state_dict_key:
            op_key = new_state_dict_key.replace("downsample", "project")
            op[op_key] = items.setdefault(new_state_dict_key)
        else:
            op_key = "module." + new_state_dict_key
            op[op_key] = items.setdefault(new_state_dict_key)

    model.load_state_dict(op, strict=False)
    return model