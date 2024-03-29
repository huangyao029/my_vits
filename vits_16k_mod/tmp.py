from thop import profile, clever_format
from torchvision.models import resnet50
import torch
from torchsummary import summary
from torchstat import stat
from ptflops import get_model_complexity_info

import models

# if __name__ == '__main__':
#     net = models.TextEncoder(191, 192, 192, 768, 2, 6, 3, 0.1)
#     # stat(net, [(513, 100), (100,)])
#     #help(stat)
    
#     model = resnet50()
#     flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings = True, print_per_layer_stat = True)
#     print(flops, params)


# import torch


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(32, 64, kernel_size = 1, stride = 1)
        self.conv2 = torch.nn.Conv1d(320, 64, kernel_size = 1, stride = 1)
        
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        return x1 + x2
    
if __name__ == '__main__':
    # net = models.TextEncoder(191, 192, 192, 768, 2, 6, 3, 0.1)
    # x1 = torch.randint(0, 190, (2, 34))
    # x2 = torch.LongTensor([34, 30])
    
    net = models.PosteriorEncoder(513, 192, 192, 5, 1, 16, 256)
    x1 = torch.randn(2, 513, 650)
    x2 = torch.LongTensor([650, 400])
    x3 = torch.randn(2, 256, 1)
    
    torch.onnx.export(
        net,
        (x1, x2, x3),
        './tmp/posteriorEncoder.onnx',
        input_names = ['spec', 'spec_length', 'g'],
        output_names = ['posterior_encoder_output']
    )
    
