from __future__ import print_function

import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import numpy as np

import torchvision.datasets as datasets
import math
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler


from PIL import Image
import matplotlib.pyplot as plt
import umap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix, identity, diags

import faiss
from progressbar import ProgressBar

def generate_subset_of_CIFAR_for_ssl(samples_per_class, label_sample_per_class=25, seed=1):
    '''Generate label and unalabel index for CIFAR10, total unlabel index is same as no of samples
    
    Args:
        
        samples_per_class(int): no. of images to be considered per class, maximum posisble is 5000
        
        label_sample_per_class(int): no. of label images per class, must be less than equal to samples_per_class, 
        default is 25
        
    
    Returns:
        index for unlabel and label
    '''
 
    trainset_cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset_cifar, batch_size=1024, shuffle=False, num_workers=20)
    
    list_of_target = []
    for i, (_, target) in enumerate(trainloader):
        list_of_target.append(target)
    list_of_target = torch.cat(list_of_target).numpy()
    
    np.random.seed(seed)
    sampled_index = []
    index_for_label = []
    for i in range(10):
        ii = np.where(list_of_target == i)[0]
        cls_sample = list(np.random.choice(ii, samples_per_class, replace=False))
        sampled_index.extend(cls_sample)
        cls_sample_labeled = list(np.random.choice(cls_sample, label_sample_per_class, replace=False))
        index_for_label.extend(cls_sample_labeled)
    
    sampled_index = list(set(sampled_index)-set(index_for_label))
    return sampled_index, index_for_label

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
   


def sem_sup_feature(conv_model: nn.Module, dl: torch.utils.data.dataloader):
    '''Extracts features from images using pretrained model

    Args:
        dl: dataloader
        conv_model: pretrained nn.Module object

    Returns:
        Tuple[list1, list2]
        list1: 2D torch tensor of size (no. of observation, embedding dim)
        list2: 1D array of corresponding labels
    '''
    conv_model.eval()
    semi_supervised_feature_list = []
    label_list = []
    for b in dl:
        data, label = b
        data = data.cuda()
        with torch.no_grad():
            out, _ = conv_model(data)
            b_size = out
            semi_supervised_feature_list.append(out)
            label_list.append(label)
    final_list = torch.cat(semi_supervised_feature_list, dim=0)
    final_label_list = torch.cat(label_list, dim=0)
    return final_list.cpu().numpy(), final_label_list.cpu().numpy()



def buildGraph(MatX, trns, knn_num_neighbors=20, dim=128) -> csr_matrix:
    '''creates the affinity matrix from the fearure matrix
    
    MatX has a particular structure. 
    
    Args:
        Matx: feature matrix of shape (num_observation, embedding_dimension). It must have the following structure.
        The first 250 rows (assuming our SSL is trained on 250 labels) will have features corresponding to 250
        labelled examples and remaining rows will be features for unlabelled examples.
        
        trns(function object): transformation for every element of affinity matrix e.g. lambda x: 0 if x < 0 else x**4
        knn_num_neighbors: # of nearest neighbors
        
        dim: embedding dimension
    
    Returns:
        sparse affinity matrix to be used for label propagation later, where labelled examples are stacked at the
        front rows. This is required for label propagation function to work properly. 
    '''
    
    num_samples = MatX.shape[0]
    index = faiss.IndexFlatIP(dim)
    index.add(MatX)
    distances, indices = index.search(MatX, knn_num_neighbors)

    trns = np.vectorize(trns)

    row = np.repeat(np.arange(num_samples), knn_num_neighbors)
    col = indices.flatten()
    data = distances.flatten()
    data = trns(data)
    sp_affinity_matrix = csr_matrix((data, (row, col)), shape=(num_samples, num_samples))
    sp_affinity_matrix = (sp_affinity_matrix + sp_affinity_matrix.transpose())/2
    return sp_affinity_matrix


def labelPropagation(sp_affinity, Mat_Label, Mat_Unlabel, labels, alpha=.1, n_iter=100) -> np.array:
    '''Propagates the label to get the prefiction for all unlabelled observations
    
    Args:
        sp_affinity: Sparse affinity matrix of shape (num_observation, num_observation).It must have the following structure.
        The first 250 rows (assuming our SSL is trained on 250 labels) will be corresponding to 250
        labelled examples and remaining rows will be corresponding to unlabelled examples.
        
        Mat_Label: Feature matrix corresponding to lablled obs.
        
        Mat_Unlabel: Feature matrix corresponding to unlablled obs.
        
        labels: labels for each row in Mat_Label
    
    
    Returns:
        predicted labels for all rows of Mat_Unlabel
    '''
    
    # initialize
    num_label_samples = Mat_Label.shape[0]
    num_unlabel_samples = Mat_Unlabel.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)

    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)
    for i in range(num_label_samples):
        clamp_data_label[i][labels[i]] = 1.0

    label_function = np.zeros((num_samples, num_classes), np.float32)
    label_function[0: num_label_samples] = clamp_data_label
    label_function[num_label_samples: num_samples] = 0
    
    degree_vec = np.sum(sp_affinity, axis=1)
    degree_vec_to_the_power_minus_half = 1/np.sqrt(degree_vec)
    sp_degree_matrix_2_the_power_minus_half = diags(np.array(degree_vec_to_the_power_minus_half).flatten())

    sp_d_minus_half_w_d_minus_half = sp_degree_matrix_2_the_power_minus_half @ sp_affinity @ sp_degree_matrix_2_the_power_minus_half

    sparse_matrix = identity(num_samples, format="csr") - alpha * sp_d_minus_half_w_d_minus_half
    
    
    
    
    normalization_diag = diags(np.array(1. / degree_vec).flatten())
    P = normalization_diag @ sp_affinity
    label_function_prop = np.copy(label_function)
    for k in range(n_iter):
        label_function_prop = P @ label_function_prop
        label_function_prop[:num_label_samples] = clamp_data_label
        unlabel_data_labels = np.argmax(label_function_prop, axis=1)
        unlabel_data_labels = unlabel_data_labels[num_label_samples:]


    return unlabel_data_labels, label_function_prop

def get_acc(predicted_labels, true_labels):
    '''returns accuracy'''
    
    corrects = 0
    num_samples = len(predicted_labels)
    for i in range(num_samples):
        if predicted_labels[i] == true_labels[i]:
            corrects +=1
    return corrects/num_samples

# to use attributes from dataparallel 
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


        
# Wide Resnet Model

class Normalize(nn.Module):

    def __init__(self, power=2, temp=1):
        super(Normalize, self).__init__()
        self.power = power
        self.temp = temp
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out/self.temp
    
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
    def __init__(self, block, num_blocks, pool_len =4, low_dim=128, fixed_classifier=False, temp=1):
        super(ResNet, self).__init__()
        self.temp = temp
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_embedding = nn.Linear(512*block.expansion, low_dim)
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
        strides = [stride] + [1]*(num_blocks-1)
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
        return out_embedding, out_class/self.temp
        # return out_embedding, out_class

    # def normalize_weight(self):
    #     with torch.no_grad():
    #         self.linear_class.weight.div_(torch.norm(self.linear_class.weight, dim=1, keepdim=True))



def ResNet18(pool_len = 4, low_dim=128, fixed_weight=True, temperature=1):
    return ResNet(BasicBlock, [2,2,2,2], pool_len, low_dim, fixed_classifier= fixed_weight, temp=temperature)







# Fast Resnet 

class Flatten(torch.nn.Module):
    """ simply flatten a feature """
    def forward(self, x):
        return x.view(x.size(0), -1)
    
def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, 
            groups=1, batchnorm=True, activation=True, bias=True):
    op = [
            torch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, 
                            padding=padding, groups=groups, 
                            bias=bias),
    ]
    if batchnorm:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
    
class resnet(torch.nn.Module):
    def __init__(self, num_class,  fixed_classifier = False):
        """
        fix_classifier: if True the last dense layer is fixed to a random orthogonal matrix
        """
        super(resnet, self).__init__()
        self.num_class = num_class
        dim_feature = 128
        
        self.feature_extractor = torch.nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            # torch.nn.MaxPool2d(2),

            Residual(torch.nn.Sequential(
                conv_bn(128, 128),
                conv_bn(128, 128),
            )),

            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2),

            Residual(torch.nn.Sequential(
                conv_bn(256, 256),
                conv_bn(256, 256),
            )),

            conv_bn(256, dim_feature, kernel_size=3, stride=1, padding=0),

            torch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            )
        self.classifier = torch.nn.Linear(dim_feature, num_class, bias=False)
        self.log_temperature = torch.nn.Parameter(torch.tensor([0.]))
        
        # FIX ORTHOGONAL CLASSIFIER
        if fixed_classifier:
            M = np.random.normal(0,1,size=(dim_feature, dim_feature))
            ortho,_,_ = np.linalg.svd(M)
            ortho = ortho[:num_class]
            self.classifier.weight = torch.nn.Parameter(torch.tensor(ortho,dtype=torch.float))
            self.classifier.weight.detach_()
            
    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(0.5)
        return x.div(norm)
    
    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = self.normalize(feature)*10. 
        #feature = feature / torch.exp(self.log_temperature)
        out = self.classifier(feature)
        return out
    
    
    
class BatchCriterion(nn.Module):  # Unsupervised Loss
    ''' Compute the unsupervised loss within each batch  
    '''
    def __init__(self, T, batchSize):
        super(BatchCriterion, self).__init__()
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize*2).cuda()
        
    def forward(self, x, targets):
        batchSize = x.size(0)
        
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2), x.narrow(0,0,batchSize//2)), 0)
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat
        all_div = all_prob.sum(1)
        

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize,1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        loss = - (lnPmtsum + lnPonsum)/batchSize
        return loss

def kNN(epoch, net, trainloader, testloader, K, sigma, ndata, low_dim=128):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    correct_t = 0
    testsize = testloader.dataset.__len__()

    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        try:
            trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
        except:
            trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
    trainFeatures = np.zeros((low_dim, ndata))
    C = trainLabels.max() + 1
    C = np.int(C)
    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=10)
        for batch_idx, (inputs, _, targets, indexes) in enumerate(temploader):
            inputs, targets = inputs.cuda(), targets.cuda()
            batchSize = inputs.size(0)
            features, _ = net(inputs)
            #
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t().cpu().numpy()

    trainloader.dataset.transform = transform_bak
    #

    trainFeatures = torch.Tensor(trainFeatures).cuda()
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            inputs, targets = inputs.cuda(), targets.cuda()
            batchSize = inputs.size(0)
            features, _ = net(inputs)
            total += targets.size(0)

            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)),
                              1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

    print(top1 * 100. / total)

    return top1 * 100. / total

def return_train_labels_index(Labels_loc):
    label_list = []
    with open(Labels_loc, 'r') as df:
        for l in df.readlines():
            a = l.rstrip().split("_")
            label_list.append(int(a[0]))

    return np.array(label_list)

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def linear_rampup_new(current, x1, y1, x2, y2):
    '''functional value of line between (x1, y1) and (x2, y2)'''
    
    m = (y1-y2)/(x1-x2)
    c = (x1*y2-x2*y1)/(x1-x2)
    
    return m*current + c

def exp_ramp_up(current, x1, y1, x2, y2):
    l = linear_rampup_new(current, x1, math.log(y1), x2, math.log(y2))
    return math.exp(l)




