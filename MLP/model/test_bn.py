import torch
import torch.nn as nn

import extension as ext
from extension.my_modules.normalization import *

class ConvBN(nn.Module):
    def __init__(self, depth=3, width=64, input_size=32*32*3, output_size=10, num_classes=10):
        super(ConvBN, self).__init__()
        layers = [nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, stride=1, padding=1),
                  ext.Norm(width), ext.Activation(width)]
        for index in range(depth-1):
            layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1))
            layers.append(ext.Norm(width))
            layers.append(ext.Activation(width))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.net = nn.Sequential(*layers)
        w = width * input_size / ((2**(depth-1))**2)
        w = int(w / 3)

        # 全连接层
        self.fc1 = nn.Linear(w, num_classes)

    def forward(self, x): 
        x = self.net(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        return x
    

class ConvBNPre(nn.Module):
    def __init__(self, depth=3, width=64, input_size=32*32*3, output_size=10, num_classes=10):
        super(ConvBNPre, self).__init__()
        layers = [nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, stride=1, padding=1),
                  ext.Activation(width)]
        for index in range(depth-1):
            layers.append(ext.Norm(width))
            layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1))
            layers.append(ext.Activation(width))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        
        # layers.append(ext.Norm(width))

        self.net = nn.Sequential(*layers)
        w = width * input_size / ((2**(depth-1))**2)
        w = int(w / 3)

        # 全连接层
        
        self.fc1 = nn.Linear(w, num_classes)

    def forward(self, x): 
        x = self.net(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        return x


# 示例：创建网络并进行前向传播
# if __name__ == "__main__":
#     # 创建网络
#     model = ConvBN(num_classes=10)
    
#     # 输入一个示例张量（假设输入是 32x32 的 RGB 图像）
#     input_tensor = torch.randn(1, 3, 32, 32)  # (batch_size, channels, height, width)
    
#     # 前向传播
#     output = model(input_tensor)
    
#     print("输入张量形状:", input_tensor.shape)
#     print("输出张量形状:", output.shape)