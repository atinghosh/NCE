import torch
from torch import nn
import torch.nn.functional as F


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return {"lr": param_group["lr"], "momentum": param_group["momentum"]}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
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
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def calculate_size(input_size, kernel_size, stride, padding=0, transposed=False):
    import math

    if transposed:
        output_size = stride * (input_size - 1) + kernel_size - 2 * padding
    else:
        output_size = math.floor((input_size + 2 * padding - kernel_size) / stride) + 1
    return output_size


class ResNet_(nn.Module):
    def __init__(self, block, num_blocks, pool_len=4, low_dim=128, normalize_flag=True):
        super(ResNet_, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, low_dim)
        self.pool_len = pool_len
        self.log_norm_cst = torch.nn.Parameter(torch.tensor([0.0]))
        self.normalize_flag = normalize_flag
        # self.log_tau = torch.nn.Parameter(torch.tensor([-2.0]))

    def norm_const(self):
        # for NCE estimation
        return torch.exp(self.log_norm_cst)

    # def tau(self):
    #     return torch.exp(self.log_tau)

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
        out = self.linear(out)
        # if self.normalize_flag:
        return self.normalize(out)

    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(0.5)
        return x.div(norm)


def resnet_original(num_class=10, fixed_classifier=False):
    pool_len = 4
    low_dim = 128
    return ResNet_(BasicBlock, [2, 2, 2, 2], pool_len, low_dim)


class CifarAutoencoder(nn.Module):
    def __init__(self):
        super(CifarAutoencoder, self).__init__()
        self.encoder = resnet_original()
        self.linear_decoder = nn.Linear(128, 48 * 4 * 4)
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            # nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        b_size = x.size(0)
        encoded = self.encoder(x)
        # encoded = self.linear_encoder(encoded.view(b_size, -1))
        decoded = self.linear_decoder(encoded).view(-1, 48, 4, 4)
        decoded = self.decoder(decoded)

        return encoded, decoded


class STL_ResNet_(nn.Module):
    def __init__(self, block, num_blocks, pool_len=4, low_dim=128):
        super(STL_ResNet_, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, low_dim)
        self.pool_len = pool_len
        self.log_norm_cst = torch.nn.Parameter(torch.tensor([0.0]))

    def norm_const(self):
        # for NCE estimation
        return torch.exp(self.log_norm_cst)

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
        out = self.layer5(out)
        out = F.avg_pool2d(out, self.pool_len)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return self.normalize(out)

    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(0.5)
        return x.div(norm)


def stl_resnet_original(num_class=10, fixed_classifier=False, low_dim=128):
    pool_len = 6
    return STL_ResNet_(BasicBlock, [2, 2, 2, 2, 2], pool_len, low_dim)


class BatchCriterion(nn.Module):
    """ Compute the loss within each batch  
    """

    def __init__(self, batchSize):
        super(BatchCriterion, self).__init__()
        self.negM = 1
        self.T = 0.1

    def forward(self, f1, f2, dd):
        x = torch.cat((f1, f2), 0)
        batchSize = x.size(0)
        diag_mat = 1 - torch.eye(batchSize).cuda()

        # get positive innerproduct
        reordered_x = torch.cat(
            (
                x.narrow(0, batchSize // 2, batchSize // 2),
                x.narrow(0, 0, batchSize // 2),
            ),
            0,
        )
        # reordered_x = reordered_x.data
        pos = (x * reordered_x.data).sum(1).div_(self.T).exp_()

        # get all innerproduct, remove diag
        all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * diag_mat
        if self.negM == 1:
            all_div = all_prob.sum(1)
        else:
            # remove pos for neg
            all_div = (all_prob.sum(1) - pos) * self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize, 1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = -(lnPmtsum + lnPonsum) / batchSize
        return loss

