from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.transform is not None:
            img1 = self.transform(img)
            if self.train:
                img2 = self.transform(img)

        if self.train:
            return img1, img2, target, index
        else:
            return img1, target, index

class CIFAR10InstanceLabelled(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """

    def __init__(self, label_index_list, *args, **kwargs):
        super(CIFAR10InstanceLabelled, self).__init__(*args, **kwargs)
        self.label_index_list = label_index_list

    def __getitem__(self, index):

        ind = self.label_index_list[index]
        img, target = self.data[ind], self.targets[ind]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.label_index_list)

