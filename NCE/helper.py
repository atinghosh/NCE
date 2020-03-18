import numpy as np
import torch
import time
import faiss
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from progress.bar import Bar as Bar
import torch.nn as nn
from imgaug import augmenters as iaa
import imgaug as ia
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time


def generate_subset_of_CIFAR_for_ssl(
    samples_per_class, label_sample_per_class=25, seed=1
):
    """Generate label and unalabel index for CIFAR10, total unlabel index is same as no of samples

    Args:

        samples_per_class(int): no. of images to be considered per class, maximum posisble is 5000

        label_sample_per_class(int): no. of label images per class, must be less than equal to samples_per_class,
        default is 25


    Returns:
        index for unlabel and label
    """

    trainset_cifar = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    trainloader = torch.utils.data.DataLoader(
        trainset_cifar, batch_size=1024, shuffle=False, num_workers=20
    )

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
        cls_sample_labeled = list(
            np.random.choice(cls_sample, label_sample_per_class, replace=False)
        )
        index_for_label.extend(cls_sample_labeled)

    sampled_index = list(set(sampled_index) - set(index_for_label))
    labels = list_of_target[index_for_label]
    labels_test = list_of_target[sampled_index]

    return (
        np.array(index_for_label),
        np.array(labels),
        np.array(sampled_index),
        np.array(labels_test),
    )


# Custom dataset


class CIFARNegativeMining(torchvision.datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """

    def __init__(self, nn_index_mat, *args, **kwargs):
        super(CIFARNegativeMining, self).__init__(*args, **kwargs)
        if self.train:
            self.index_mat = nn_index_mat

    def __getitem__(self, index):
        if self.train:
            # img, target = self.train_data[index], self.train_labels[index]
            index1, index2 = self.index_mat[index, 49], self.index_mat[index, 99]
            img1, target = self.data[index], self.targets[index]
            img2, img3 = self.data[index1], self.data[index2]
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img1, img2, img3 = (
                Image.fromarray(img1),
                Image.fromarray(img2),
                Image.fromarray(img3),
            )

            img1_v1, img2_v1, img3_v1 = (
                self.transform(img1),
                self.transform(img2),
                self.transform(img3),
            )
            img1_v2, img2_v2, img3_v2 = (
                self.transform(img1),
                self.transform(img2),
                self.transform(img3),
            )

            return img1_v1, img2_v1, img3_v1, img1_v2, img2_v2, img3_v2, target, index

        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)

            return img1, target, index


class CIFAR10_pairs(torchvision.datasets.CIFAR10):
    """
    produces pairs of augmented images
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        return self.transform(img), self.transform(img), target


class STL10_pairs(torchvision.datasets.STL10):
    """
    produces pairs of augmented images
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        return self.transform(img), self.transform(img), target


# Augmentation

# class ImgAugTransform:
#     def __init__(self):
#         self.aug = iaa.Sequential([
#             iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
#             iaa.Fliplr(0.5),
#             iaa.Affine(rotate=(-20, 20), mode='symmetric'),
#             iaa.Sometimes(0.25,
#                           iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
#                                      iaa.CoarseDropout(0.1, size_percent=0.5)])),
#             iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
#         ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)

# tfs = torchvision.transforms.Compose([
#     torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
#     ImgAugTransform(),
#     torchvision.transforms. ToPILImage(),
#     torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
#     torchvision.transforms.ToTensor()
# ])


sometimes = lambda aug: iaa.Sometimes(0.5, aug)


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential(
            [
                #             iaa.Resize((224, 224)),
                # iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.Fliplr(0.5),
                #             iaa.Flipud(0.1), # vertically flip 20% of all images
                iaa.Affine(
                    rotate=(-5, 5),
                    shear=(-3, 3),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    mode="symmetric",
                ),
                # iaa.SaltAndPepper(p=(0, 0.03)),
                #             iaa.Sometimes(0.1,
                #                           iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                #                                      iaa.CoarseDropout(0.1, size_percent=0.5)])),
                #             iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                #             sometimes(
                #                     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                #                 ),
                #             iaa.Grayscale(alpha=(0.0, 1.0)),
                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                #             iaa.Multiply((0.8, 1.2), per_channel=0.2),
                #             iaa.ContrastNormalization((0.75, 1.5)),
                #             iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
            ],
            random_order=True,
        )
        self.aug2 = iaa.Sequential(
            [
                #             iaa.Scale((224, 224)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-20, 20), mode="symmetric"),
                iaa.Sometimes(
                    0.25,
                    iaa.OneOf(
                        [
                            iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(0.1, size_percent=0.5),
                        ]
                    ),
                ),
                iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            ]
        )

        self.aug3 = iaa.Sequential(
            [
                #             iaa.Scale((224, 224)),
                iaa.RandAugment(n=2, m=9)
            ]
        )

    def __call__(self, img):
        img = np.array(img)
        return self.aug3(image=img)


#         return self.aug1.augment_image(img)
# return self.aug.augment_image(img)

tfs = torchvision.transforms.Compose(
    [
        ImgAugTransform(),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        # torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
    ]
)


class CutOut(object):
    """
    CutOut augmentation -- replace a box by its mean (coordinate-wise)
    """

    def __init__(self, side):
        self.side = side

    def __call__(self, image):
        xx = np.random.randint(0, 32 - self.side)
        yy = np.random.randint(0, 32 - self.side)
        for c in range(3):
            image[c, xx : (xx + self.side), yy : (yy + self.side)] = torch.mean(
                image[c, xx : (xx + self.side), yy : (yy + self.side)]
            )
        return image


class PixelJitter(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, power_min=0.4, power_max=1.4):
        assert 0 < power_min
        assert power_min < power_max
        self.power_min = power_min
        self.power_max = power_max

    def __call__(self, image):
        p = self.power_min + (self.power_max - self.power_min) * np.random.rand()
        image = image ** p
        for c in range(3):
            alpha = -0.1 + 0.2 * np.random.rand()
            strech = 0.8 + 0.4 * np.random.rand()
            image[c, :, :] = alpha + strech * image[c, :, :]
        image = torch.clamp(image, min=0.0, max=1.0)
        return image


# compute all the representations
def extract_features(dataloader, model, device):
    """
    extract all the features from a given dataset
    """
    model.eval()
    features_array = None
    labels_array = None
    for k, batch in enumerate(dataloader):
        image, label = batch
        feature = model(image.to(device)).data.cpu().numpy()

        if k == 0:
            features_array = feature
            labels_array = label
        else:
            features_array = np.concatenate([features_array, feature], axis=0)
            labels_array = np.concatenate([labels_array, label], axis=0)
    return features_array, labels_array


def feature_loss(model, features_0, features_1, args, device, log_temp):
    """
    features_0, features_1: batch of features
    """
    threshold = 0.1  # for triplet loss
    approach = args.approach
    # approach = "spreading_instance_feature"

    batchsize = len(features_0)

    if args.allow_grad_inner:
        sim_00 = features_0 @ features_0.t()
        sim_01 = features_0 @ features_1.t()
        sim_10 = features_1 @ features_0.t()  # redundant -- clean-up
        sim_11 = features_1 @ features_1.t()  # redundant -- clean-up
    else:
        sim_00 = features_0 @ features_0.data.t()
        sim_01 = features_0 @ features_1.data.t()
        sim_10 = features_1 @ features_0.data.t()  # redundant -- clean-up
        sim_11 = features_1 @ features_1.data.t()  # redundant -- clean-up

    sim_positive = torch.diag(sim_10)

    dist_00 = 1.0 - sim_00
    dist_01 = 1.0 - sim_01
    dist_positive = torch.diag(dist_00)

    #
    #  BASIC TRIPLET LOSS
    #
    if args.approach == "max_margin":
        triplet = (threshold + dist_positive).view(batchsize, 1) - dist_00
        loss_full = torch.clamp_min(triplet, 0)
        n_positive_terms = torch.sum(loss_full > 0)

        if n_positive_terms >= 1:
            loss = torch.sum(loss_full) / n_positive_terms
        else:
            loss = 0.0
        return loss

    #
    # Improved Deep Metric Learning with Multi-class N-pair Loss Objective
    #
    elif args.approach == "N_pairs_soft_plus":
        triplet = (dist_positive).view(batchsize, 1) - dist_00
        loss = torch.mean(torch.log(1.0 + torch.exp(triplet)))
        return loss

    #
    #  Noise Contrastive Estimation
    #

    elif args.approach == "sim_CLR":
        temp = 0.1  # temparture patameter
        proba_unorm_10 = torch.exp(sim_10 / temp)
        proba_unorm_01 = torch.exp(sim_01 / temp)
        proba_unorm_00 = torch.exp(sim_00 / temp)
        proba_unorm_11 = torch.exp(sim_11 / temp)

        norm_constant_10 = (
            torch.sum(proba_unorm_10, axis=1)
            + torch.sum(proba_unorm_11, axis=1)
            - torch.diag(proba_unorm_11)
            - torch.diag(proba_unorm_10)
        )
        norm_constant_01 = (
            torch.sum(proba_unorm_01, axis=1)
            + torch.sum(proba_unorm_00, axis=1)
            - torch.diag(proba_unorm_00)
            - torch.diag(proba_unorm_01)
        )

        # Z = model.norm_const()
        proba_norm_10 = proba_unorm_10 / (1 * norm_constant_10.view(-1, 1))
        proba_norm_01 = proba_unorm_01 / (1 * norm_constant_01.view(-1, 1))
        # loss = -torch.sum(torch.log(torch.diag(proba_norm_10)))
        # loss += -torch.sum(torch.log(torch.diag(proba_norm_01)))
        loss = torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_01)))
        return loss / batchsize

    elif args.approach == "NCE":
        temp = 0.1  # temparture patameter
        proba_unorm_10 = torch.exp(sim_10 / temp)
        proba_unorm_01 = torch.exp(sim_01 / temp)
        proba_unorm_00 = torch.exp(sim_00 / temp)
        proba_unorm_11 = torch.exp(sim_11 / temp)

        # with torch.no_grad():
        #     norm_constant_shared = torch.sum(proba_unorm_10, axis=1)
        #     norm_constant_10 = norm_constant_shared + torch.sum(proba_unorm_11, axis=1) - torch.diag(proba_unorm_11)
        #     norm_constant_01 = norm_constant_shared + torch.sum(proba_unorm_00, axis=1) - torch.diag(proba_unorm_00)

        with torch.no_grad():
            norm_constant_1 = torch.sum(proba_unorm_10, axis=1)
            norm_constant_0 = torch.sum(proba_unorm_01, axis=1)
            norm_constant_10 = (
                norm_constant_1
                + torch.sum(proba_unorm_11, axis=1)
                - torch.diag(proba_unorm_11)
            )
            norm_constant_01 = (
                norm_constant_0
                + torch.sum(proba_unorm_00, axis=1)
                - torch.diag(proba_unorm_00)
            )

        Z = model.norm_const()
        proba_norm_10 = proba_unorm_10 / (Z * norm_constant_10.view(-1, 1))
        proba_norm_11 = proba_unorm_11 / (Z * norm_constant_10.view(-1, 1))
        proba_norm_01 = proba_unorm_01 / (Z * norm_constant_01.view(-1, 1))
        proba_norm_00 = proba_unorm_00 / (Z * norm_constant_01.view(-1, 1))

        loss = torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + proba_norm_10))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + proba_norm_11))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_11)))

        loss += torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_01)))
        loss += torch.sum(torch.log(1.0 + proba_norm_01))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_01)))
        loss += torch.sum(torch.log(1.0 + proba_norm_00))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_00)))

        return loss / batchsize

    # DIFFERENT VARIANT OF NCE

    # Allow gradient through denominator
    elif args.approach == "NCE_all_grad":
        temp = 0.1  # temparture patameter
        proba_unorm_10 = torch.exp(sim_10 / temp)
        proba_unorm_01 = torch.exp(sim_01 / temp)
        proba_unorm_00 = torch.exp(sim_00 / temp)
        proba_unorm_11 = torch.exp(sim_11 / temp)

        norm_constant_1 = torch.sum(proba_unorm_10, axis=1)
        norm_constant_0 = torch.sum(proba_unorm_01, axis=1)
        norm_constant_10 = (
            norm_constant_1
            + torch.sum(proba_unorm_11, axis=1)
            - torch.diag(proba_unorm_11)
        )
        norm_constant_01 = (
            norm_constant_0
            + torch.sum(proba_unorm_00, axis=1)
            - torch.diag(proba_unorm_00)
        )

        Z = model.norm_const()
        proba_norm_10 = proba_unorm_10 / (Z * norm_constant_10.view(-1, 1))
        proba_norm_11 = proba_unorm_11 / (Z * norm_constant_10.view(-1, 1))
        proba_norm_01 = proba_unorm_01 / (Z * norm_constant_01.view(-1, 1))
        proba_norm_00 = proba_unorm_00 / (Z * norm_constant_01.view(-1, 1))

        loss = torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + proba_norm_10))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + proba_norm_11))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_11)))

        loss += torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_01)))
        loss += torch.sum(torch.log(1.0 + proba_norm_01))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_01)))
        loss += torch.sum(torch.log(1.0 + proba_norm_00))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_00)))

        return loss / batchsize

    # Remove Z from NCE
    elif approach == "NCE_without_z":
        temp = 0.1  # temparture patameter
        proba_unorm_10 = torch.exp(sim_10 / temp)
        proba_unorm_01 = torch.exp(sim_01 / temp)
        proba_unorm_00 = torch.exp(sim_00 / temp)
        proba_unorm_11 = torch.exp(sim_11 / temp)

        # with torch.no_grad():
        #     norm_constant_shared = torch.sum(proba_unorm_10, axis=1)
        #     norm_constant_10 = norm_constant_shared + torch.sum(proba_unorm_11, axis=1) - torch.diag(proba_unorm_11)
        #     norm_constant_01 = norm_constant_shared + torch.sum(proba_unorm_00, axis=1) - torch.diag(proba_unorm_00)

        with torch.no_grad():
            norm_constant_1 = torch.sum(proba_unorm_10, axis=1)
            norm_constant_0 = torch.sum(proba_unorm_01, axis=1)
            norm_constant_10 = (
                norm_constant_1
                + torch.sum(proba_unorm_11, axis=1)
                - torch.diag(proba_unorm_11)
            )
            norm_constant_01 = (
                norm_constant_0
                + torch.sum(proba_unorm_00, axis=1)
                - torch.diag(proba_unorm_00)
            )

        # Z = model.norm_const()
        proba_norm_10 = proba_unorm_10 / (norm_constant_10.view(-1, 1))
        proba_norm_11 = proba_unorm_11 / (norm_constant_10.view(-1, 1))
        proba_norm_01 = proba_unorm_01 / (norm_constant_01.view(-1, 1))
        proba_norm_00 = proba_unorm_00 / (norm_constant_01.view(-1, 1))

        loss = torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + proba_norm_10))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + proba_norm_11))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_11)))

        loss += torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_01)))
        loss += torch.sum(torch.log(1.0 + proba_norm_01))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_01)))
        loss += torch.sum(torch.log(1.0 + proba_norm_00))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_00)))

        return loss / batchsize

    elif approach == "learnable_tau":
        temp = torch.exp(log_temp)
        # temp = model.tau()  # temparture patameter
        proba_unorm_10 = torch.exp(sim_10 / temp)
        proba_unorm_01 = torch.exp(sim_01 / temp)
        proba_unorm_00 = torch.exp(sim_00 / temp)
        proba_unorm_11 = torch.exp(sim_11 / temp)

        # with torch.no_grad():
        #     norm_constant_shared = torch.sum(proba_unorm_10, axis=1)
        #     norm_constant_10 = norm_constant_shared + torch.sum(proba_unorm_11, axis=1) - torch.diag(proba_unorm_11)
        #     norm_constant_01 = norm_constant_shared + torch.sum(proba_unorm_00, axis=1) - torch.diag(proba_unorm_00)

        with torch.no_grad():
            norm_constant_1 = torch.sum(proba_unorm_10, axis=1)
            norm_constant_0 = torch.sum(proba_unorm_01, axis=1)
            norm_constant_10 = (
                norm_constant_1
                + torch.sum(proba_unorm_11, axis=1)
                - torch.diag(proba_unorm_11)
            )
            norm_constant_01 = (
                norm_constant_0
                + torch.sum(proba_unorm_00, axis=1)
                - torch.diag(proba_unorm_00)
            )

        Z = model.norm_const()
        proba_norm_10 = proba_unorm_10 / (Z * norm_constant_10.view(-1, 1))
        proba_norm_11 = proba_unorm_11 / (Z * norm_constant_10.view(-1, 1))
        proba_norm_01 = proba_unorm_01 / (Z * norm_constant_01.view(-1, 1))
        proba_norm_00 = proba_unorm_00 / (Z * norm_constant_01.view(-1, 1))

        loss = torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + proba_norm_10))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_10)))
        loss += torch.sum(torch.log(1.0 + proba_norm_11))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_11)))

        loss += torch.sum(torch.log(1.0 + 1.0 / torch.diag(proba_norm_01)))
        loss += torch.sum(torch.log(1.0 + proba_norm_01))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_01)))
        loss += torch.sum(torch.log(1.0 + proba_norm_00))
        loss -= torch.sum(torch.log(1.0 + torch.diag(proba_norm_00)))

        return loss / batchsize

    #  paper: Unsupervised Embedding Learning via Invariant and Spreading Instance Feature
    #
    elif approach == "spreading_instance_feature":
        temp = 0.1

        x = torch.cat((features_0, features_1), 0)
        batchSize = x.size(0)
        diag_mat = 1 - torch.eye(batchSize).to(device)

        # get positive innerproduct
        reordered_x = torch.cat(
            (
                x.narrow(0, batchSize // 2, batchSize // 2),
                x.narrow(0, 0, batchSize // 2),
            ),
            0,
        )
        # reordered_x = reordered_x.data
        pos = (x * reordered_x.data).sum(1).div_(temp).exp_()

        # get all innerproduct, remove diag
        all_prob = torch.mm(x, x.t().data).div_(temp).exp_() * diag_mat
        all_div = all_prob.sum(1)

        lnPmt = torch.div(pos, all_div)  # alex: OK

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
        lnPonsum = lnPonsum
        loss = -(lnPmtsum + lnPonsum) / batchSize

        del diag_mat
        return loss


def train_autoencoder(model, dl, args, device, model_path):

    model.train()
    start = time.time()
    threshold = 0.1

    criterion = nn.BCELoss().to(device)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00001)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.ae_lr, weight_decay=0.00001
    )

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
        train_loss_list = []
        bar = Bar(f"AE Training, epoch: {ep}", max=len(dl))
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
    print("Saving AE model..")
    state = {"model": model.encoder.state_dict(), "acc": 0, "epoch": -1}
    torch.save(state, model_path)


def train(model, pair_dataloader, optimizer, scheduler, args, device, log_temp):

    bar = Bar("UEL Training", max=len(pair_dataloader))

    model.train()
    train_loss_list = []
    start = time.time()
    threshold = 0.1

    # features_array = -1
    # labels_array = -1
    # first_batch = True

    for batch_id, batch in enumerate(pair_dataloader):
        if args.nm:
            batch_0 = torch.cat((batch[0], batch[1], batch[2]), 0).to(device)
            batch_1 = torch.cat((batch[3], batch[4], batch[5]), 0).to(device)
            labels = batch[6]
        else:
            batch_0 = batch[0].to(device)
            batch_1 = batch[1].to(device)
            labels = batch[2]

        if args.no_mixup:
            batch_1_mixup = batch_1
        elif args.mixup_uniform:
            with torch.no_grad():
                batch_sz = len(batch_0)

                # random permutation
                index = np.random.choice(batch_sz, replace=False, size=batch_sz)
                batch_0_shuffled = batch_0[index, :, :, :]

                beta = np.random.uniform(size=batch_sz)
                beta_array = torch.tensor(beta, dtype=torch.float).to(device)
                batch_1_mixup = (
                    1.0 - beta_array.view(batch_sz, 1, 1, 1)
                ) * batch_1 + beta_array.view(batch_sz, 1, 1, 1) * batch_0_shuffled
        else:
            with torch.no_grad():
                batch_sz = len(batch_0)

                # random permutation
                index = np.random.choice(batch_sz, replace=False, size=batch_sz)
                batch_0_shuffled = batch_0[index, :, :, :]

                beta_param_1, beta_param_2 = 1, 2  # parameter of the beta distribution
                beta = np.random.beta(beta_param_1, beta_param_2, size=batch_sz)
                beta = np.minimum(
                    1.0 - beta, beta
                )  # dont want beta to be larger than 0.5
                beta_array = torch.tensor(beta, dtype=torch.float).to(device)
                batch_1_mixup = (
                    1.0 - beta_array.view(batch_sz, 1, 1, 1)
                ) * batch_1 + beta_array.view(batch_sz, 1, 1, 1) * batch_0_shuffled

        # features_0 = model.normalize(model.feature_extractor(batch_0))
        # combined_batch = torch.cat((batch_0, batch_1_mixup), 0)
        #
        # features = model(combined_batch)
        features_0 = model(batch_0)
        features_1 = model(batch_1_mixup)
        # print(features.size())
        # features_0, features_1 = features[:batch_sz], features[batch_sz: ]
        # print(features_0.shape), print(features_1.shape)
        # features_0 = model.normalize(model.feature_extractor(batch_0))
        # features_1 = model.normalize(model.feature_extractor(batch_1))
        # features_1 = model.normalize(model.feature_extractor(batch_1_mixup))

        loss = feature_loss(model, features_0, features_1, args, device, log_temp)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_list.append(loss.item())
        with torch.no_grad():
            temp = torch.exp(log_temp)
        bar.suffix = (
            f"({batch_id}/{len(pair_dataloader)}) | ETA: {bar.eta_td} | Loss: {loss.item():.4f} | "
            f"Avg. Loss: {np.mean(np.array(train_loss_list)):.4f} | temp:{temp.item():.2f}"
        )
        bar.next()
        # save the features for later analysis
        # if first_batch:
        #     first_batch = False
        #     features_array = features_0.data.cpu().numpy()
        #     labels_array = labels.data.cpu().numpy()
        # else:
        #     features_array = np.concatenate([features_array, features_0.data.cpu().numpy()], axis=0)
        #     labels_array = np.concatenate([labels_array, labels.data.cpu().numpy()], axis=0)

    # update learning rate
    # scheduler.step()
    bar.finish()
    compute_time = time.time() - start
    avg_loss = np.mean(np.array(train_loss_list))
    return {"avg_loss": avg_loss, "compute_time": compute_time}


def knn_accuracy(feature_nn, label_nn, feature_test, label_test):
    num_nearest_neighbor = 50
    _, embedding_dim = feature_nn.shape
    index = faiss.IndexFlatIP(embedding_dim)  # cosine similarity
    index.add(feature_nn)
    distances, indices = index.search(feature_test, num_nearest_neighbor)
    proba_knn = np.zeros((len(indices), 10))
    for k in range(10):
        proba_knn[:, k] = np.mean(label_nn[indices] == k, axis=1)
    pred_knn = np.argmax(proba_knn, axis=1)
    accuracy = np.mean(pred_knn == label_test)
    return accuracy


def logistic_accuracy(feature_nn, label_nn, feature_test, label_test):

    start = time.time()
    sc_x = StandardScaler()
    feature_nn = sc_x.fit_transform(feature_nn)
    feature_test = sc_x.transform(feature_test)
    clf = LogisticRegression(
        random_state=0,
        n_jobs=-1,
        solver="sag",
        multi_class="multinomial",
        max_iter=200,
    )
    clf.fit(feature_nn, label_nn)
    pred_logistic = clf.predict(feature_test)
    accuracy = np.mean(pred_logistic == label_test)
    time_taken = time.time() - start
    return accuracy, time_taken


def mlp_accuracy(feature_nn, label_nn, feature_test, label_test):

    start = time.time()
    sc_x = StandardScaler()
    feature_nn = sc_x.fit_transform(feature_nn)
    feature_test = sc_x.transform(feature_test)
    clf = MLPClassifier(
        solver="adam", alpha=1e-5, hidden_layer_sizes=(20,), random_state=1
    )
    clf.fit(feature_nn, label_nn)
    pred_logistic = clf.predict(feature_test)
    accuracy = np.mean(pred_logistic == label_test)
    time_taken = time.time() - start
    return accuracy, time_taken


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Logger(object):
    """Save training process to log file with simple plot function."""

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = "" if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, "r")
                name = self.file.readline()
                self.names = name.rstrip().split("\t")
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split("\t")
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, "a")
            else:
                self.file = open(fpath, "w")

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write("\t")
            self.numbers[name] = []
        self.file.write("\n")
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), "Numbers do not match names"
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write("\t")
            self.numbers[self.names[index]].append(num)
        self.file.write("\n")
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + "(" + name + ")" for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + "(" + name + ")" for name in names]


class LoggerMonitor(object):
    """Load and visualize multiple logs."""

    def __init__(self, paths):
        """paths is a distionary with {name:filepath} pair"""
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.grid(True)
