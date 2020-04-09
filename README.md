## Variants of NCE for unsupervised embedding learning 

This repository contains code about implementation of different variants of NCE for representation learning on CIFAR10 and STL10. Below we give description of all type of loss implemented. **To render the math equations correctly use mathjax plugin for github Chrome App store**. 

## List of Losses

1. NCE - vanila NCE, with both positive and negative term
2. [sim_CLR](https://arxiv.org/pdf/2002.05709.pdf) - Geoff Hinton's paper without the negative term in NCE, very similar to CPC loss
3. [Spreading_instance_feature](https://arxiv.org/pdf/1904.03436.pdf) - This is taken from this paper, which also includes negative term in NCE loss
4. NCE_all_grad - We allow gradient through normalization constant, vanila version keeps it fixed
5. NCE_without_z - we omit the z term in vanila NCE
6. learnable_tau - we make the temparature learnable parameter, this doesn't work!
7. no_norm_const - NCE without the normalization constant, we replace the normalization constant by 1.
8. nm - NCE with negative mining, we do hard negative mining . So for every image in the batch, we include the 50th and 100th nearest neighbor also in the batch. This makes the batch size 3x the original batch size, accordingly we adjust the number of epochs.
9. no_mixup - We run without the mixup data augmentation. 
10. rand_aug -  we use [Randaugment](https://arxiv.org/pdf/1909.13719.pdf), with 2 randomly chosen augmentation from a list of 15 augmentations and intensity of each augmentation is chosen to be 9.
11. ae_pretrain - We pretrain our feature extractor resnet model (which extracts a feature of dimension 128 for CIFAR10) with an auto encoder (by adding decoder network) for 100 epochs and then start our original training with weights initialized from this pretrained auto encoder.

#### Two main variants of NCE

**Vanila NCE**- 
$$
J_{N C E}^{\prime}(\theta)=\frac{1}{n} \sum_{i=1}^{n}\left[\log \left(1+\frac{z \sum_{j=1}^{k} \varphi\left(y_{i j}, x ; \theta\right)}{\varphi\left(y_{i 0}, x ; \theta\right)}\right)+\sum_{j=1}^{k} \log \left(1+\frac{\varphi\left(y_{i j}, x ; \theta\right)}{z \sum_{j=1}^{k} \varphi\left(y_{i j}, x ; \theta\right)}\right)\right]
$$
where, $\varphi(y, x ; \theta)=\exp \left(<f_{\theta}(y), f_{\theta}(x)>/ \tau\right)$

$f_{\theta}(x)$ is 128 dimensional output from our resnet model for image $x$. 

For **Sim_CLR** we do not use the second term or the negative term, and the loss becomes
$$
J_{N C E}(\theta)=-\frac{1}{n} \sum_{i=1}^{n} \log \left(\frac{\varphi\left(y_{i 0}, x ; \theta\right)}{\sum_{j=1}^{k} \varphi\left(y_{i j}, x ; \theta\right)}\right)
$$


## Usage

To train sim_CLR model in (2) we need to execute the following, this will create a folder name `cifar_sim_clr` in the log folder in the current directory and inside the log folder model and the log.txt will be saved. log.txt will have information about knn accuracy and lp accuracy for every epoch. 

```bash
python NCE/CIFAR_unsup.py --name cifar_sim_clr --gpu 0,1 --approach sim_CLR
```

To resume training from the last epoch use

```bash
python NCE/CIFAR_unsup.py --resume cifar_sim_clr --gpu 0,1 --approach sim_CLR
```

Note that we need to use same name we chose in the `--name` argument for the `--resume` argument too.



- `--name` : name of the experiment 

- `--resume` : resume flag for the experiment, takes the name of the experiment from `--name` as argument

- `--approach` : name of the loss to be used with one of following choices as described in the list of loss section. 

  ```python
  choices=["max_margin",
          "N_pairs_soft_plus",
          "NCE",
          "spreading_instance_feature",
          "NCE_all_grad",
          "NCE_without_z",
          "learnable_tau",
          "sim_CLR",
          "no_norm_const"],
  ```

- `--gpu` : gpu to be used for the experiment

- `--no_mixup` : turns off mixup augmentation, by default mixup will be used.

- `--mixup_uniform` : Instead of using weight which is close to 1, we use weight drawn from Uniform(0,1) - Note it doesn't work.
- `--rand_aug` : turn on randaugment, default is False
- `--nm` : turn on negative mining, by default it is False.
- `--ae_pretrain` - turn on auto encoder pretraining, by default it is False
- `--nb_epoch_ae` : if the ae_pretrain flag is True, then how many epochs of autoencoder pretraining need to be run
- `--ae_lr` : learning rate for autoencoder training, default is .001

- `--dim` :  dimension of the embedding, default is 128 for CIFAR
- `--batch-size` : default batch size is 512 (batch size needs to be tuned with learning rate)
- `--lr`  : default learning rate is .03
- `--n-epoch` : default no. of epoch is 2000
- `--n-label` : no. of labeled images for label propagation for SSL, default is 250
- `--n-worker` : no of worker CPU thread to use for data loader, default is 15
- `--dataset` : "cifar": cifar-10 dataset, "stl": stl-10 dataset. default is cifar
- `--log-dir` : location of log directory relative to current directory, default is `log/`. The results and model object will be saved in this location inside the name of a folder with same name as passed in the argument of `--name`.


## Citation

1. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)

2. [Unsupervised Embedding Learning via Invariant and Spreading Instance Feature](https://arxiv.org/pdf/1904.03436.pdf)

3. [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)

4. [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/pdf/1909.13719.pdf)

   













