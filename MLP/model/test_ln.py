import torch
import torch.nn as nn

import extension as ext
from extension.normalization import *

class ConvLN(nn.Module):
    def __init__(self, depth=3, width=64, input_size=32*32*3, output_size=10, num_classes=10):
        super(ConvLN, self).__init__()
        norm_width = 32
        layers = [nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, stride=1, padding=1),
                  ext.Norm(norm_width), ext.Activation(width)]
                #   nn.BatchNorm2d(width), nn.ReLU()]

        if depth > 3:
            for index in range(2):
                layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1))
                # layers.append(nn.BatchNorm2d(width))
                # layers.append(nn.ReLU())
                layers.append(ext.Norm(int(norm_width/(2**index))))
                layers.append(ext.Activation(width))
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            for index in range(depth-3):
                layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1))
                # layers.append(nn.BatchNorm2d(width))
                # layers.append(nn.ReLU())
                layers.append(ext.Norm(int(norm_width/4)))
                layers.append(ext.Activation(width))

            self.net = nn.Sequential(*layers)
            w = width * input_size / ((2**(2))**2)
            w = int(w / 3)
        else:
            for index in range(depth-1):
                layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1))
                # layers.append(nn.BatchNorm2d(width))
                # layers.append(nn.ReLU())
                layers.append(ext.Norm(int(norm_width/(2**index))))
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
    

class ConvLNPre(nn.Module):
    def __init__(self, depth=3, width=64, input_size=32*32*3, output_size=10, num_classes=10):
        super(ConvLNPre, self).__init__()
        norm_width = 32
        layers = [nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, stride=1, padding=1),
                  ext.Activation(width)]
                #   nn.ReLU()]

        if depth > 3:
            for index in range(2):
                # layers.append(nn.BatchNorm2d(width))
                layers.append(ext.Norm(int(norm_width/(2**index))))
                layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1))
                # layers.append(nn.ReLU())
                layers.append(ext.Activation(width))
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            for index in range(depth-3):
                # layers.append(nn.BatchNorm2d(width))
                layers.append(ext.Norm(int(norm_width/4)))
                layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1))
                # layers.append(nn.ReLU())
                layers.append(ext.Activation(width))

            self.net = nn.Sequential(*layers)
            w = width * input_size / ((2**(2))**2)
            w = int(w / 3)
        else:
            for index in range(depth-1):
            #     layers.append(nn.BatchNorm2d(width))
                layers.append(ext.Norm(int(norm_width/(2**index))))
                layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1))
                # layers.append(nn.ReLU())
                layers.append(ext.Activation(width))
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            self.net = nn.Sequential(*layers)
            w = width * input_size / ((2**(depth-1))**2)
            w = int(w / 3)
        # layers.append(ext.Norm(width))

        # 全连接层
        
        self.fc1 = nn.Linear(w, num_classes)

    def forward(self, x): 
        x = self.net(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,norm_width):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = ext.Norm(norm_width)
        self.act1 = ext.Activation(out_channels)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        # self.act1 = nn.ReLU()
        
        
        # 如果输入和输出通道数不同，使用1x1卷积调整通道数
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        # 添加残差连接
        out = out + identity
        return out
    

class ConvLNRes(nn.Module):
    def __init__(self, depth=3, width=64, input_size=32*32*3, output_size=10, num_classes=10):
        super(ConvLNRes, self).__init__()
        norm_width = 32
        layers = [nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, stride=1, padding=1),
                  ext.Norm(norm_width), ext.Activation(width)]
                #   nn.BatchNorm2d(width), nn.ReLU()]


        if depth > 3:
            for index in range(2):
                layers.append(ResidualBlock(in_channels=width, out_channels=width, norm_width=int(norm_width/(2**index))))
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            for index in range(depth-3):
                layers.append(ResidualBlock(in_channels=width, out_channels=width, norm_width=int(norm_width/4)))

            self.net = nn.Sequential(*layers)
            w = width * input_size / ((2**(2))**2)
            w = int(w / 3)
        else:
            for index in range(depth-1):
                layers.append(ResidualBlock(in_channels=width, out_channels=width,norm_width=int(norm_width/(2**index))))
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





class ResidualBlockPre(nn.Module):
    def __init__(self, in_channels, out_channels, norm_width):
        super(ResidualBlockPre, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = ext.Norm(norm_width)
        self.act1 = ext.Activation(out_channels)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        # self.act1 = nn.ReLU()
        
        
        # 如果输入和输出通道数不同，使用1x1卷积调整通道数
    
    def forward(self, x):
        identity = x
        out = self.norm1(out)
        out = self.conv1(x)
        out = self.act1(out)

        # 添加残差连接
        out = out + identity
        return out
    

class ConvLNResPre(nn.Module):
    def __init__(self, depth=3, width=64, input_size=32*32*3, output_size=10, num_classes=10):
        super(ConvLNResPre, self).__init__()
        norm_width = 32
        layers = [nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, stride=1, padding=1),
                    ext.Activation(width)]
                #   nn.BatchNorm2d(width), nn.ReLU()]


        if depth > 3:
            for index in range(2):
                layers.append(ResidualBlock(in_channels=width, out_channels=width, norm_width=int(norm_width/(2**index))))
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            for index in range(depth-3):
                layers.append(ResidualBlock(in_channels=width, out_channels=width,norm_width=int(norm_width/4)))

            self.net = nn.Sequential(*layers)
            w = width * input_size / ((2**(2))**2)
            w = int(w / 3)
        else:
            for index in range(depth-1):
                layers.append(ResidualBlock(in_channels=width, out_channels=width,norm_width=int(norm_width/(2**index))))
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

# 示例：创建网络并进行前向传播
if __name__ == "__main__":
    # 创建网络
    model = ConvLNRes(depth=5, num_classes=10)
    
    # 输入一个示例张量（假设输入是 32x32 的 RGB 图像）
    input_tensor = torch.randn(1, 3, 32, 32)  # (batch_size, channels, height, width)
    
    # 前向传播
    output = model(input_tensor)
    
    print(model)
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output.shape)