# Pytorch 공식 문서를 가져왔으며, 필요한 부분에 주석문을 더했음.
import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
#from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url


__all__ = ['GoogLeNet', 'googlenet', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])   #namedTuple -> 키값으로 접근 가능.
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}                          #output 변수의 자료형을 지정. Optional -> None 이 허용됨.

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat                                            #backwards compat -> 하위 호환성. _GoogleNetOutputs 는 하위호환성을 위해 존재.
_GoogLeNetOutputs = GoogLeNetOutputs


def googlenet(pretrained=False, progress=True, **kwargs):   # pretrain 인데 aux_ligits 가 False 이면 , pretrain 파라미터개수에 맞추기 위해서, aux_logits 을 True로 만들고 모델 생성, 파라미터 로드 후에 다시 하이퍼파라미터를 변경함.
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:    #Pretrain 이 True 라면 ~
        if 'transform_input' not in kwargs:  # transform_input 기본값 == True
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:       # aux_logits 기본값 == False
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:             # aux_logits 가 존재한다면 경고문을 준다. -> "프리트레인 모델에는 보조분류기가 훈련되어 있지 않다!"    보조분류기는 훈련때만 쓰이고, inference시에는 쓰이지 않는다.
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']   # original_aux_logits 에 aux_logits 의 원래 값을 저장해놓음.
        kwargs['aux_logits'] = True                  # kwargs[aux_logits] 에 True 를 대입
        kwargs['init_weights'] = False               # 프리트레인 모델이니까 가중치 초기화는 안함.
        model = GoogLeNet(**kwargs)                  # kwargs 를 넘겨서 모델 생성. ( 일단 보조분류기가 있다고 가정하고, 모델 생성)  밑에 if 문에서 모델 내의 하이퍼파라미터 변경
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)      # url 에서 파라미터 가져옴.
        model.load_state_dict(state_dict)                             # 모델에 파라미터 load
        if not original_aux_logits:                  # 만약 original_aux_logits 가 False 라면..
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
        return model                                # pretrain 이 True 면 model 을 넘김.

    return GoogLeNet(**kwargs)                      # pretrain 이 True 가 아니면 **kwargs 그대로 넘겨서 모델 생성.


class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input'] #__constants__ 는 CUDA 명령어.

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=None,
                 blocks=None):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux] # 세 개의 블록 클래스를 리스트에 넣음.
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)        # googlenet 파라미터 초기화 방식이 바뀔 예정. 만약 scipy 패키지로 인해 긴 초기화 시간을 가진 예전 방식을 원한다면 init_weights = True 로 설정할 것.
            init_weights = True
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits   # inception_aux_block -> Auxiliary Classifiers (보조 분류기)를 맨 마지막에 추가할 지를 결정.
        self.transform_input = transform_input # transfrom_input -> 채널별로 imageNet 의 Mean 값과 Var 값을 기준으로 transform

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  #ceil -> 소숫점 올림. // padding 개수를 올림.   # Maxpooling 다음 레이어에 LRN 이 논문에 등장하지만, 요즘엔 잘 안쓰이는 계층.
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

                                                                    # class inception : def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, conv_block=None):
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self): #다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함#다시 봐야함
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats                   #scipy.stats 는 확률분포를 지원한다.
                X = stats.truncnorm(-2, 2, scale=0.01)        #truncnorm -> truncated normal distribution
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5 # 배치 전체에 대해 각 채널로 나누는 과정에서 차원수가 하나 줄음. 원상복귀 위해서 unsqeeze 함수 실행. # unsqeeze -> 해당 차원으로 차원수를 늘림.
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5 # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]". # FROM IMAGENET
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1 = torch.jit.annotate(Optional[Tensor], None)       # jit.annotate()를 aux1 변수에 할당.
        if self.aux1 is not None:                               # 보조분류기1
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2 = torch.jit.annotate(Optional[Tensor], None)       # jit.annotate()를 aux2 변수에 할당.
        if self.aux2 is not None:                               # 보조분류기2
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)                                     # 1,1 adaptiveAveragePooling
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

    @torch.jit.unused                                           # 이 데코레이터는 함수 또는 메서드를 무시하고 예외 발생으로 대체해야 함을 컴파일러에 나타냅니다. 이를 통해 아직 TorchScript와 호환되지 않는 코드를 모델에 남겨두고 모델을 내보낼 수 있습니다.
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x   # type: ignore[return-value]

    def forward(self, x):                                                # 다른 갈래로 갈라지는 // if 문으로 생기는 branch 를 고려하기 위해, _forward 와 forward 가 구분됨.
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits                  # 훈련중이고, 보조 분류기를 쓴다 -> aux_defined
        if torch.jit.is_scripting():                                     # is_scripting() -> 컴파일시에 True 를 반환한다. TorchScript와 호환되지 않는 코드를 모델에 남겨두기 위해 @unused 데코레이터와 함께 특히 유용합니다.
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)                       # 컴파일 시에는 GoogleNetOutputs 반환.
        else:
            return self.eager_outputs(x, aux2, aux1)                     # 만약 컴파일 시가 아니라면 eager_output -> _GoogleNetOutPuts() 반환. // 하위 호환성을 위해 존재.


class Inception(nn.Module):
#inception_block(192, 64, 96, 128, 16, 32, 32)
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)   # w,h 그대로 , d -> ch1X1 = 64        #nn.Sequential 함수를 이용하면 forward 함수가 간결해짐.

        self.branch2 = nn.Sequential(                                  # w,h 그대로 , d -> ch3X3 = 128
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(                                  # w,h 그대로 , d -> ch5X5 = 32
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(                                  # w,h 그대로 , d -> pool_proj = 32
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):   # aux1 -> in_channels = 512 / aux2 -> in_channels = 528
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

