import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from typing import Callable, Any, Optional, List
from torchvision import models
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url  # for torchvision with a lower version

from src.repvgg import get_RepVGG_func_by_name
from src.tiny_vit import tiny_vit_11m_224, tiny_vit_21m_224

from pytorchcv.model_provider import get_model  # pip install pytorchcv

def get_network(config):
    if config.network == 'mobilenet':
        net = MobileNet(num_classes=config.num_classes)
    elif config.network == 'resnet18':
        net = get_ResNet(config)
    elif config.network == 'resnet50':
        net = get_ResNet(config)
    elif config.network == 'repvgg':
        net = RepVggNet(num_classes=config.num_classes)  # only RepVGG-B1g2
    elif config.network == 'effinetv2':
        net = get_EfficientNet_V2(config, model_name="S")  # S / M / L
    elif config.network == 'effinet':
        net = get_EfficientNet(config, model_name="b4")  # b0~b7
    elif config.network == 'tinyvit':
        net = get_TinyViT(config, model_name="21m")  # 11m or 21m
    else:
        raise NotImplementedError()
    net.cuda()
    return net


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            expand_ratio: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNet, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU6(inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, num_classes),
        )
    
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x_feat = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)

        out = self.classifier(x_feat)
        return out
            
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

class RepVggNet(nn.Module):
    def __init__(self,
                 backbone_name='RepVGG-B1g2',
                 backbone_file='weights/RepVGG-B1g2-train.pth',
                 deploy=False,
                 pretrained=True,
                 num_classes=9):
        super(RepVggNet, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel  # 2048
        # self.linear_reg = nn.Linear(fea_dim, num_classes)
        
        self.linear_reg = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fea_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU6(inplace=True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU6(inplace=True),
            nn.Linear(128, num_classes),
        )

        for m in self.linear_reg :
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        feat = torch.flatten(x, 1)
        x = self.linear_reg(feat)
        return x

"""
# https://arxiv.org/abs/2104.00298
# https://github.com/google/automl/tree/master/efficientnetv2
# https://pytorch.org/vision/main/models/efficientnetv2.html
ImageNet1K	Top1 Acc.	Params(TF)  FLOPs       acc@1   File size   GFLOPS  
EffNetV2-S	83.9%	    21.5M       8.4B        84.228  82.7 MB     8.37
EffNetV2-M	85.2%	    54.1M       24.7B       85.112  208.0 MB    24.58
EffNetV2-L	85.7%	    119.5M      56.3B       85.808  454.6 MB    56.08
"""
def get_EfficientNet_V2(config, model_name="S"):
    if model_name=="S":
        model = models.efficientnet_v2_s(weights='DEFAULT')
    if model_name=="M":
        model = models.efficientnet_v2_m(weights='DEFAULT')
    if model_name=="L":
        model = models.efficientnet_v2_l(weights='DEFAULT')
    
    out_dim = 1280  # always be the same dim number
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(out_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU6(inplace=True),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU6(inplace=True),
        nn.Linear(128, config.num_classes),
    )
    
    for m in model.classifier:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    return model


"""
# https://github.com/osmr/imgclsmob/blob/master/pytorch/README.md
# https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py
# https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/efficientnet.py#L356
Model               Top1	Top5	Params	    FLOPs/2
Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385
ResNet-50	        22.28	6.33	25,557,032	3,877.95M
ResNet-50b	        22.39	6.38	25,557,032	4,110.48M
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, https://arxiv.org/abs/1704.04861
MobileNet x1.0	    26.61	8.95	4,231,976	579.80M
MobileNetb x1.0	    25.45	8.16	4,222,056	574.97M
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, https://arxiv.org/abs/1905.11946
EfficientNet-B0b	23.88	7.02	5,288,548	414.31M
EfficientNet-B1b	21.60	5.94	7,794,184	732.54M  <----
EfficientNet-B2b	20.31	5.27	9,109,994	1,051.98M
EfficientNet-B3b	18.83	4.45	12,233,232	1,928.55M
EfficientNet-B4b	17.45	3.89	19,341,616	4,607.46M  <----
EfficientNet-B5b	16.56	3.37	30,389,784	10,695.20M
EfficientNet-B6b	16.29	3.23	43,040,704	19,796.24M
EfficientNet-B7b    15.94   3.22	66,347,960	39,010.98M

(ICML2021) EfficientNetV2: Smaller Models and Faster Training, https://github.com/google/automl/tree/master/efficientnetv2
(CVPR2022) A ConvNet for the 2020s, https://github.com/facebookresearch/ConvNeXt
"""
def get_EfficientNet(config, pretrain=True, model_name="b1"):
    if model_name=="b0":
        model = get_model("efficientnet_b0b", pretrained=pretrain)
        out_dim = 1280  # x 1.0
    if model_name=="b1":
        model = get_model("efficientnet_b1b", pretrained=pretrain)
        out_dim = 1280  # x 1.0
    if model_name=="b2":
        model = get_model("efficientnet_b2b", pretrained=pretrain)
        out_dim = 1408  # x 1.1
    if model_name=="b3":
        model = get_model("efficientnet_b3b", pretrained=pretrain)
        out_dim = 1536  # x 1.2
    if model_name=="b4":
        model = get_model("efficientnet_b4b", pretrained=pretrain)
        out_dim = 1792  # x 1.4
    # print(model)
    
    model.output = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(out_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU6(inplace=True),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU6(inplace=True),
        nn.Linear(128, config.num_classes),
    )
    
    for m in model.output:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    return model

    
# https://github.com/microsoft/Cream/tree/main/TinyViT
# Model	        Pretrain	Input	Acc@1	Acc@5	#Params	MACs	FPS
# TinyViT-5M	IN-1k	    224x224	79.1	94.8	5.4M	1.3G	3,060
# TinyViT-11M	IN-1k	    224x224	81.5	95.8	11M	    2.0G	2,468
# TinyViT-21M	IN-1k	    224x224	83.1	96.5	21M	    4.3G	1,571
def get_TinyViT(config, pretrain=True, model_name="11m"):
    if model_name=="11m":
        model = tiny_vit_11m_224(pretrained=pretrain)
        out_dim = 448
    if model_name=="21m":
        model = tiny_vit_21m_224(pretrained=pretrain)
        out_dim = 576
    
    # print(model)  # for debug
    # model.head = nn.Linear(embed_dims[-1], num_classes)  # maybe (448, 1000) for tiny_vit_11m_224
    # model.head = nn.Linear(embed_dims[-1], num_classes)  # maybe (576, 1000) for tiny_vit_21m_224
    
    model.head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(out_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU6(inplace=True),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU6(inplace=True),
        nn.Linear(64, config.num_classes),
    )
    
    for m in model.head:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    return model

def get_ResNet(config, pretrain=True):
    model = models.__dict__[config.network]()
    if pretrain:
        print(f"Loading pretrained model...")
        model.load_state_dict(load_state_dict_from_url(models.resnet.model_urls[config.network], map_location='cuda'))
    
    if config.network == 'resnet18':
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU6(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, config.num_classes),
        )
    
    if config.network == 'resnet50':
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU6(inplace=True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU6(inplace=True),
            nn.Linear(128, config.num_classes),
        )
        
    for m in model.fc:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    if not pretrain:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    return model


class TestConfig:
    def __init__(self):
        self.num_classes = 9
        self.network = 'resnet18'



if __name__ == '__main__':
    config = TestConfig()
    net = get_network(config)
    img = torch.randn(2, 3, 227, 227).cuda()
    output = net(img)
    print(output.shape)
