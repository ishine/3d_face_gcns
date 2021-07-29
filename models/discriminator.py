import torch.nn as nn
from torch.nn import init


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNorm = nn.BatchNorm2d(num_features=out_channels)
        self.leakyRelu = nn.LeakyReLU(0.2, True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        init.xavier_normal_(self.conv.weight, gain=init.calculate_gain('leaky_relu', 0.2))
        init.constant_(self.conv.bias, 0.0)
        init.normal_(self.batchNorm.weight, 1.0, 0.02)
        init.constant_(self.batchNorm.bias, 0.0)

    def forward(self, input):
        output = self.conv(input)
        output = self.batchNorm(output)
        output = self.leakyRelu(output)
        output = self.maxPool(output)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = [
            BasicBlock(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            BasicBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            BasicBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            BasicBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            BasicBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=7, stride=1, padding=0)
        ]

        self.net = nn.Sequential(*self.net)

        init_weights(self.net)

    def forward(self, inputs):
        return self.net(inputs)  # input: B x 3 x 224 x 224


def init_weights(net, init_type='xavier', init_gain=0.02):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
