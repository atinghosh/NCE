# Define temparature
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torchvision

from progress.bar import Bar as Bar
import os
import math
import time
import numpy as np


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

def train_autoencoder(model, dl, device, lr=.1):

    model.train()
    train_loss_list = []
    # start = time.time()

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)

    def lr_lambda_ae(epoch):
        if epoch < 50:
            return 1
        elif epoch >= 50 and epoch < 70:
            return 0.16666667
        elif epoch >= 70 and epoch < 90:
            return 0.0335
        else:
            return 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_ae)

    for ep in range(args.nb_epoch_ae):
        bar = Bar("Autoencoder Training", max=len(dl))
        for i, (inputs, _) in enumerate(dl):
            inputs = inputs.to(device)
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_list.append(loss.item())

            bar.suffix = (
                f"({i}/{len(dl)}) | ETA: {bar.eta_td} | Loss: {loss.item():.4f} | "
                f"Avg. Loss: {np.mean(np.array(train_loss_list)):.4f}"
            )
            bar.next()
        bar.finish()

base = "./data/"
download = True
test_augment = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
cifar_train_dataset_noaugment = torchvision.datasets.CIFAR10(
    base, train=True, transform=test_augment, download=download
)
dataloader_trainnoaugment = torch.utils.data.DataLoader(
        cifar_train_dataset_noaugment,
        batch_size=512,
        shuffle=True,
        num_workers=15,
    )

device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
ae_model = CifarAutoencoder().to(device)
train_autoencoder(ae_model, dataloader_trainnoaugment, device)

