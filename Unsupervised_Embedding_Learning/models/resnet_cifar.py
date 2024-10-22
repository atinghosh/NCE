'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.normalize import Normalize
import math
from torch.autograd import Variable
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, pool_len=4, low_dim=128, fixed_classifier=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_embedding = nn.Linear(512 * block.expansion, low_dim)
        self.linear_class = nn.Linear(low_dim, 10, bias=False)
        if fixed_classifier:
            M = np.random.normal(0, 1, size=(low_dim, low_dim))
            ortho, _, _ = np.linalg.svd(M)
            ortho = ortho[:10]
            self.linear_class.weight.data = torch.tensor(ortho, dtype=torch.float).cuda()
            self.linear_class.weight.detach_()
        # with torch.no_grad():
        #     self.linear_class.weight.div_(torch.norm(self.linear_class.weight, dim=1, keepdim=True))
        self.l2norm_for_feature = Normalize()
        # self.l2norm_for_weight = Normalize(temp=.1)
        self.pool_len = pool_len
        # w = torch.nn.Parameter(torch.randn(128, 10))
        # for m in self.modules():
        # if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        # elif isinstance(m, nn.BatchNorm2d):
        # m.weight.data.fill_(1)
        # m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.pool_len)
        out = out.view(out.size(0), -1)
        out_embedding = self.linear_embedding(out)
        out_embedding = self.l2norm_for_feature(out_embedding)
        # with torch.no_grad():
        #     self.linear_class.weight.div_(torch.norm(self.linear_class.weight, dim=1, keepdim=True))
        # self.linear_class.weight.div_(torch.norm(self.linear_class.weight, dim=1, keepdim=True))
        # type(self.l2norm_for_weight(self.linear_class.weight))
        # self.linear_class.weight = self.l2norm_for_weight(self.linear_class.weight)
        # self.linear_class.weight = torch.nn.Parameter(10 * self.linear_class.weight/torch.norm(self.linear_class.weight, dim=1, keepdim=True))
        # out_class = 10*self.linear_class(out_embedding)/torch.norm(self.linear_class.weight, dim=1, keepdim=True).transpose(1,0)
        out_class = self.linear_class(out_embedding)
        return out_embedding, out_class


def ResNet18(pool_len = 4, low_dim=128):
    return ResNet(BasicBlock, [2,2,2,2], pool_len, low_dim)

def ResNet34(pool_len = 4, low_dim=128):
    return ResNet(BasicBlock, [3,4,6,3], pool_len, low_dim)

def ResNet50(pool_len = 4, low_dim=128):
    return ResNet(Bottleneck, [3,4,6,3], pool_len, low_dim)

def ResNet101(pool_len = 4, low_dim=128):
    return ResNet(Bottleneck, [3,4,23,3], pool_len, low_dim)

def ResNet152(pool_len = 4, low_dim=128):
    return ResNet(Bottleneck, [3,8,36,3], pool_len, low_dim)


def test():
    net = ResNet18()
    # y = net(Variable(torch.randn(1,3,32,32)))
    # pdb.set_trace()
    y, _ = net(Variable(torch.randn(1,3,96,96)))
    # pdb.set_trace()
    print(y.size())

# test()
