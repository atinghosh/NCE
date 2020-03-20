import argparse
import math
import logging

# import NCE.unsupervised_feature_model as unsupervised_feature_model
# from NCE.helper import *
# from NCE.LP import *
import os
import sys

# Define temparature
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# sys.path.append('/data2/atin/Pycharmprojects/SSL_LP/')
# sys.path.append('/data02/Atin/LATEST/SSL_LABEL_PROPAGATION/')
import unsupervised_feature_model
from helper import *
from LP import *

import copy
import os 
os.environ['USE_DAAL4PY_SKLEARN'] = 'YES'


parser = argparse.ArgumentParser(description="unsupervised embedding training with NCE")
parser.add_argument(
    "--name",
    default="cifar_nce_pretrain",
    type=str,
    help="give a suitable name of the experiment",
)
parser.add_argument(
    "--dataset",
    default="cifar",
    help='dataset name: "cifar": cifar-10 datasetor "stl": stl-10 dataset]',
)
parser.add_argument(
    "--gpu", default="0,2", type=str, help="gpu device ids for CUDA_VISIBLE_DEVICES"
)
parser.add_argument(
    "--approach",
    default="NCE",
    choices=[
        "max_margin",
        "N_pairs_soft_plus",
        "NCE",
        "spreading_instance_feature",
        "NCE_all_grad",
        "NCE_without_z",
        "learnable_tau",
        "sim_CLR"
    ],
    help="choice of loss to be used for embedding learning",
)

parser.add_argument("--no_mixup", action="store_true", help="turn off mix up aug")
parser.add_argument(
    "--allow_grad_inner",
    action="store_true",
    help="choice of loss to be used for embedding learning",
)

parser.add_argument("--dim", type=int, default=128, help="embedding dimension")

parser.add_argument(
    "--resume", "-r", default="", type=str, help="resume from checkpoint"
)
parser.add_argument("--log-dir", default="log/", type=str, help="log save path")
parser.add_argument(
    "--batch-size", default=512, type=int, metavar="B", help="training batch size"
)

# Learning rate arguments
parser.add_argument("--lr", default=0.03, type=float, help="learning rate")
parser.add_argument(
    "--lr-patience", default=10, type=int, help="learning rate patience"
)
parser.add_argument(
    "--lr-factor", default=0.5, type=float, help="learning rate patience"
)
parser.add_argument(
    "--lr-gamma", default=0.5, type=float, help="learning rate patience"
)


parser.add_argument(
    "--n-epoch", type=int, default=2000, help="number of epochs for model to train"
)
parser.add_argument(
    "--n-label",
    type=int,
    default=250,
    help="number of labels, should be multiple of 10",
)

parser.add_argument("--seed", default=1, help="seed value for calculative LP")
parser.add_argument("--m-gpu", action="store_false")

parser.add_argument("--n-worker", type=int, default=15, help="number of worker")
parser.add_argument("--rand_aug", action="store_true", help="use random augmentation")

parser.add_argument("--ae_pretrain", action="store_true", help="apply ae pretraining")
parser.add_argument("--nm", action="store_true", help="do negative mining")
parser.add_argument("--mixup_uniform", action="store_true", help="use uniform random for mixup")

parser.add_argument(
    "--nb_epoch_ae",
    default=100,
    type=int,
    help="number of epochs for auto encoder training",
)
parser.add_argument("--ae_lr", type=float, default=0.001)

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

base_path = args.log_dir + args.name + "/"  # example --  "./log/nce_vanila/"
log_path = os.path.join(base_path, "log.txt")  # example --  "./log/nce_vanila/log.txt"


model_path = os.path.join(
    base_path, "model_ckpt.t"
)  # example --  "./log/nce_vanila/model_ckpt.t"

print(f"no_mix up flag: {args.no_mixup}")
print(f"rand_aug flag: {args.rand_aug}")

if not os.path.isdir(base_path):
    os.makedirs(base_path)
# writer = SummaryWriter(base_path)
all_log_path = os.path.join(base_path, "log_all.txt")
logging.basicConfig(level=logging.INFO,filename=all_log_path)
logger_all = logging.getLogger(__name__)

base = "./data/"
download = True
train_dataset = torchvision.datasets.CIFAR10(root=base, train=True, download=download)

test_dataset = torchvision.datasets.CIFAR10(root=base, train=False, download=download)

if args.dataset == "cifar":
    img_size = 32
if args.dataset == "stl":
    img_size = 96

if args.rand_aug:
    # train_augment = transforms.Compose(
    #     [
    #         transforms.RandomCrop(32, padding=4),
    #         torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    #         torchvision.transforms.RandomGrayscale(p=0.1),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.ToTensor(),
    #     ]
    # )

    # # Add RandAugment with N, M(hyperparameter)
    # train_augment.transforms.insert(0, RandAugment(3, 1))
    train_augment = tfs
else:
    train_augment = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    )

test_augment = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# DEFINE DATASET and DaTALOADER
if args.dataset == "cifar":
    cifar_train_dataset_noaugment = torchvision.datasets.CIFAR10(
        base, train=True, transform=test_augment, download=False
    )
    cifar_test_dataset = torchvision.datasets.CIFAR10(
        base, train=False, transform=test_augment, download=False
    )

    # create a batch loader that outputs pairs of augmented images
    cifar_train_pair_dataset = CIFAR10_pairs(
        base, train=True, transform=train_augment, download=False
    )
    # CREATE  BATCHLOADER
    dataloader_trainnoaugment = torch.utils.data.DataLoader(
        cifar_train_dataset_noaugment,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_worker,
    )
    dataloader_test = torch.utils.data.DataLoader(
        cifar_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_worker,
    )
    dataloader_train_pairs = torch.utils.data.DataLoader(
        cifar_train_pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_worker,
    )

if args.dataset == "stl":
    stl_train_dataset_noaugment = torchvision.datasets.STL10(
        base, split="train", transform=test_augment, download=True
    )
    stl_test_dataset = torchvision.datasets.STL10(
        base, split="test", transform=test_augment, download=True
    )
    # stl_train_dataset_noaugment = torch.utils.data.ConcatDataset([stl_train_dataset_noaugment, stl_test_dataset])
    stl_train_pair_dataset = STL10_pairs(
        base, split="train+unlabeled", transform=train_augment, download=True
    )
    # stl_train_pair_dataset = STL10_pairs(base, split='train', transform=train_augment, download=True)

    # CREATE BATCHLOADER
    dataloader_trainnoaugment = torch.utils.data.DataLoader(
        stl_train_dataset_noaugment,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_worker,
    )
    dataloader_test = torch.utils.data.DataLoader(
        stl_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_worker,
    )
    dataloader_train_pairs = torch.utils.data.DataLoader(
        stl_train_pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_worker,
    )


if args.dataset == "cifar":
    if args.ae_pretrain:
        ae_model = unsupervised_feature_model.CifarAutoencoder().to(device)
        train_autoencoder(ae_model, dataloader_trainnoaugment, args, device, model_path)
        model = unsupervised_feature_model.resnet_original()
        model = copy.deepcopy(ae_model.encoder)
    else:
        model = unsupervised_feature_model.resnet_original()  # model from the paper
else:
    model = unsupervised_feature_model.stl_resnet_original(
        low_dim=args.dim
    )  # model from the paper

if device == "cuda" and args.m_gpu:
    # print(torch.cuda.device_count())
    model = MyDataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# def lr_lambda(epoch):
#     if epoch <= 800:
#         return 1
#     elif epoch > 800 and epoch <= 1100:
#         return 0.16666667
#     elif epoch > 1100 and epoch <= 1400:
#         return 0.0335
#     else:
#         return 0.01
# def lr_lambda(epoch):
#     if epoch < 1210:
#         return 1
#     elif epoch >= 1210 and epoch < 1510:
#         return 0.16666667
#     elif epoch >= 1510 and epoch < 1810:
#         return 0.0335
#     else:
#         return 0.01

if args.dataset == "cifar":

    if args.nm:
        def lr_lambda(epoch):
            if epoch < 403:
                return 1
            elif epoch >= 403 and epoch < 503:
                return 0.16666667
            elif epoch >= 503 and epoch < 603:
                return 0.0335
            else:
                return 0.01
    else:
        def lr_lambda(epoch):
            if epoch < 1210:
                return 1
            elif epoch >= 1210 and epoch < 1510:
                return 0.16666667
            elif epoch >= 1510 and epoch < 1810:
                return 0.0335
            else:
                return 0.01

if args.dataset == "stl":

    def lr_lambda(epoch):
        if epoch < 60:
            return 1
        elif epoch >= 60 and epoch < 100:
            return 0.05
        elif epoch >= 100 and epoch < 120:
            return 0.01
        elif epoch >= 120 and epoch < 140:
            return 0.005
        elif epoch >= 120 and epoch < 140:
            return 0.001
        else:
            return 0.0001

if args.approach == "learnable_tau":
    # log_temp = torch.tensor(-2., device = device, requires_grad=True)
    log_temp = torch.randn(1, device=device, requires_grad=True)
    optimizer = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": log_temp, "lr": 0.001}],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

else:
    log_temp = torch.log(torch.tensor(0.1)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if len(args.resume) > 0:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    old_base_path = args.log_dir + args.resume + "/"
    old_model_path = os.path.join(old_base_path, "model_ckpt.t")
    assert os.path.isfile(old_model_path), "Error: no model file found!"
    model_path = old_model_path
    checkpoint = torch.load(old_model_path)
    print(old_model_path)
    print(old_base_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"We are starting from epoch: {start_epoch} and best acc is {best_acc}")

    title = "CIFAR-10_NCE"
    old_log_path = os.path.join(old_base_path, "log.txt")
    assert os.path.isfile(old_log_path), "Error: no log file found!"
    logger = Logger(old_log_path, title=title, resume=True)
else:
    best_acc = 0
    start_epoch = 0
    title = "CIFAR-10_NCE"
    logger = Logger(log_path, title=title)
    if args.dataset == "cifar":
        logger.set_names(
            ["epoch", "LR", "Train Loss", "KNN acc", "LP acc", "time", "total_time"]
        )
    else:
        logger.set_names(["epoch", "LR", "Train Loss", "KNN acc", "time", "total_time"])

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor,
#                                                        patience=args.lr_patience, threshold=.0005)

# if args.dataset == "cifar":
#     milestones = list(range(20, args.n_epoch, 10))
# else:
#     milestones = list(range(2, args.n_epoch, 2))
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lr_gamma)

train_loss_list = []
knn_acc_list = []
total_train_time = 0

if args.dataset == "cifar":
    (
        labels_index,
        labels,
        labels_index_test,
        labels_test,
    ) = generate_subset_of_CIFAR_for_ssl(5000, args.n_label // 10, 1)


for epoch in range(start_epoch, args.n_epoch):
    if args.nm:
        # if epoch <=1:
        #     train_results = train(model, dataloader_train_pairs, optimizer, scheduler, args, device, log_temp)
        if epoch % 2 == 0 or len(args.resume) > 0 :
            args.resume = []
            feature_mat, label_arr = extract_features(dataloader_trainnoaugment, model, device)
            index = faiss.IndexFlatIP(128)
            index.add(feature_mat)
            distances, indices = index.search(feature_mat, 100)

            trainset_nm = CIFARNegativeMining(indices, root='./data', train=True, download=True,
                                                transform=train_augment)
            trainloader_nm = torch.utils.data.DataLoader(trainset_nm, batch_size=args.batch_size, shuffle=True, num_workers=20,
                                                    drop_last=True)
        train_results = train(model, trainloader_nm, optimizer, scheduler, args, device, log_temp)
        
    else:
        train_results = train(model, dataloader_train_pairs, optimizer, scheduler, args, device, log_temp)
    
    scheduler.step(epoch)
    # scheduler.step(train_results["avg_loss"])
    total_train_time += train_results["compute_time"]

    optim_param = unsupervised_feature_model.get_lr(optimizer)
    norm_const = model.norm_const().item()

    # KNN accuracy
    feature_train, label_train = extract_features(
        dataloader_trainnoaugment, model, device
    )
    feature_test, label_test = extract_features(dataloader_test, model, device)
    # acc_knn = knn_accuracy(train_results["features"], train_results["labels"],
    #                       feature_test, label_test)
    acc_knn = knn_accuracy(feature_train, label_train, feature_test, label_test)
    if epoch == args.n_epoch-1:
        acc_logistic = logistic_accuracy(feature_train, label_train, feature_test, label_test)
        logger_all.info(f"logistic accuracy is {acc_logistic[0]:.4f} and time taken {acc_logistic[1]:.4f}")

        acc_mlp = mlp_accuracy(feature_train, label_train, feature_test, label_test)
        logger_all.info(f"logistic accuracy is {acc_mlp[0]:.4f} and time taken {acc_mlp[1]:.4f}")

    

    if args.dataset == "cifar":
        sp_affinity_matrix = build_affinity(feature_train)
        n_total, _ = sp_affinity_matrix.shape
        label_pred = label_prop(sp_affinity_matrix, labels_index, labels, 0.95)
        lp_accuracy = compute_accuracy(label_pred, labels_index_test, labels_test)

    # if acc_knn > best_acc:
    print("Saving..")
    state = {"model": model.state_dict(), "acc": acc_knn, "epoch": epoch, "optimizer": optimizer.state_dict()}
    torch.save(state, model_path)
    best_acc = acc_knn

    # if epoch == 299:
    #     state = {"model": model.state_dict(), "acc": acc_knn, "epoch": epoch, "optimizer": optimizer.state_dict()}
    #     torch.save(state, model_path+'_300')


    if args.dataset == "cifar":
        with torch.no_grad():
            temp = torch.exp(log_temp)
        print(
            "Epoch:[{epoch}/{n_epoch}] \t "
            "Loss:{train_loss:.2f} \t "
            "Z:{norm_const:.2f} \t"
            "lr/mmt:{lr:.3f}/{momentum:.2f} \t"
            "time:{time:.0f}s({total:.0f}s) \t"
            "knnt:{acc_knn:.1f} % \t"
            "LP acc:{LP:.2f} %".format(
                epoch=epoch,
                n_epoch=args.n_epoch,
                train_loss=train_results["avg_loss"],
                lr=optim_param["lr"],
                momentum=optim_param["momentum"],
                time=train_results["compute_time"],
                total=total_train_time,
                acc_knn=100 * acc_knn,
                norm_const=norm_const,
                temp=temp.item(),
                LP=100 * lp_accuracy,
            )
        )
        logger.append(
            [
                epoch,
                optim_param["lr"],
                train_results["avg_loss"],
                acc_knn,
                lp_accuracy,
                train_results["compute_time"],
                total_train_time,
            ]
        )
    else:
        print(
            "Epoch:[{epoch}/{n_epoch}] \t "
            "Loss:{train_loss:.2f} \t "
            "Z:{norm_const:.2f} \t"
            "lr/mmt:{lr:.3f}/{momentum:.2f} \t"
            "time:{time:.0f}s({total:.0f}s) \t"
            "knnt:{acc_knn:.1f} % ".format(
                epoch=epoch,
                n_epoch=args.n_epoch,
                train_loss=train_results["avg_loss"],
                lr=optim_param["lr"],
                momentum=optim_param["momentum"],
                time=train_results["compute_time"],
                total=total_train_time,
                acc_knn=100 * acc_knn,
                norm_const=norm_const,
            )
        )
        logger.append(
            [
                epoch,
                optim_param["lr"],
                train_results["avg_loss"],
                acc_knn,
                train_results["compute_time"],
                total_train_time,
            ]
        )
