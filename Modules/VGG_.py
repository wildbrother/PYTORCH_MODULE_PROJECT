# PYTORCH 공식 문서의 코드를 참조하였고, 주석문으로 설명을 더했다.
import torch
import torch.nn as nn

class VGG(nn.Module) :
    def __init__(self, features , num_classes=1000, init_layers = True):
        super(VGG,self).__init__()
        self.feautures = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7)) # nn.AvgPool2d 와는 다르게 출력크기만 입력하면, kernel,stride 를 알아서 할당해줍니다.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True), #inplace == True -> input으로 들어온 값 자체를 수정 // 메모리 usage 가 좋아짐.
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,num_classes)
        )
        if init_layers :
            self._initialize_weights()

    def forward(self,x):
        x = self.feautures(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)  #reshape 와 동일한 기능. 하지만 view 는 동일한 메모리를 공유하며 그로 인해 continguous 한 경우에 쓸 수 있다. -> 그렇지 않으면 에러발생)
        x = self.classifier(x)

        return x


    def _initialize_weights(self):
        for m in self.modules() : # nn.Module().modules() 함수는 각 레이어를 iterable 하게 돌려준다. 이 함수는 각 레이어의 종류를 확인하고, initialize 하기 위해 사용된다.
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu') # _ underbar 가 있는 함수들은 대부분 새로운 tensor를 리턴하지 않고, 기존 tensor를 변경함. # Kaiming_normal(fan_out , leaky_relu) -> He initialization
                if m.bias is not None :
                    nn.init.constant_(m.bias,0) # bias 가 있다면 0으로 초기화

            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)


def make_layers(cfg,batch_norm = False):
    layers = []
    in_channels = 1
    for v in cfg :
        if v == 'M' :
            layers += [nn.MaxPool2d(kernel_size=2, stride= 2)]
        else :
            conv2d = nn.Conv2d(in_channels=in_channels,out_channels=v,kernel_size=3,padding=1)
            if batch_norm :
                layers += [conv2d,nn.BatchNorm2d(v), nn.ReLU(True)]
            else :
                layers += [conv2d,nn.ReLU(True)]

            in_channels = v

    return nn.Sequential(*layers) # 파이썬의 에스터리스크는 복수개의 인자를 받을 때 사용.

cfgs = {
    'MNIST' : [64,'M',128,'M',256,'M',512,512,"M"], # 기존의 cfg 로는 MNIST 의 해상도에서 Maxpooling 을 반복하면 남는 픽셀이 존재하지 않는다.
    'A' : [64,'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ],
    'B' : [64, 64,'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ],
    'D' : [64, 64,'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M' ],
    'E' : [64, 64,'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M' ],
}
