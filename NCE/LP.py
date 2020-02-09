import faiss
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from numpy.linalg import multi_dot
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import cg


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
        trainset_cifar, batch_size=1024, shuffle=False, num_workers=0
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


def build_affinity(feature_train):
    knn_num_neighbors = 20
    n_train, dim = feature_train.shape
    index = faiss.IndexFlatIP(dim)
    index.add(feature_train)
    distances, indices = index.search(feature_train, knn_num_neighbors)

    row = np.repeat(np.arange(n_train), knn_num_neighbors)
    col = indices.flatten()
    data = distances.flatten()
    data = np.exp(data)
    sp_affinity_matrix = csr_matrix((data, (row, col)), shape=(n_train, n_train))
    sp_affinity_matrix = (sp_affinity_matrix + sp_affinity_matrix.transpose()) / 2
    # sp_affinity_matrix = np.maximum(sp_affinity_matrix, sp_affinity_matrix.transpose())
    return sp_affinity_matrix


def create_clamp_vector(sp_affinity_matrix, labels_index, labels):
    num_classes = 10
    n_total, _ = sp_affinity_matrix.shape
    clamp_data_label = np.zeros((n_total, num_classes))
    for i in range(len(labels_index)):
        clamp_data_label[labels_index[i], labels[i]] = 1.0
    return clamp_data_label


def label_prop(sp_affinity_matrix, labels_index, labels, alpha):
    num_classes = 10
    n_total, _ = sp_affinity_matrix.shape
    degree_vec = np.sum(sp_affinity_matrix, axis=1)
    degree_vec_to_the_power_minus_half = 1 / np.sqrt(degree_vec)
    sp_degree_matrix_2_the_power_minus_half = diags(
        np.array(degree_vec_to_the_power_minus_half).flatten()
    )
    sp_d_minus_half_w_d_minus_half = (
        sp_degree_matrix_2_the_power_minus_half
        @ sp_affinity_matrix
        @ sp_degree_matrix_2_the_power_minus_half
    )
    sparse_matrix = (
        identity(n_total, format="csr") - alpha * sp_d_minus_half_w_d_minus_half
    )
    clamp_data_label = create_clamp_vector(sp_affinity_matrix, labels_index, labels)

    label_pred = np.zeros((n_total, num_classes))
    # solve with CG
    for i in range(num_classes):
        # print(i, label_pred.shape, sparse_matrix.shape, clamp_data_label.shape)
        label_pred[:, i] = cg(sparse_matrix, clamp_data_label[:, i])[0]
    return label_pred


def compute_accuracy(label_pred, labels_index_test, labels_test):
    labels_test_pred = np.argmax(label_pred, axis=1)[labels_index_test.astype(int)]
    corrects = 0
    for i in range(len(labels_test_pred)):
        if labels_test_pred[i] == labels_test[i]:
            corrects += 1
    return corrects / len(labels_test_pred)
