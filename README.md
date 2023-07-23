# Deep Active Learning for Defect Classification in PyTorch

This repo is inspired from Akshay L Chandra [DAL-pytorch](https://github.com/acl21/deep-active-learning-pytorch) and modified to suit the defect classification usecase.  

## Introduction

The origianl repo demonstrated the deep active learning on CIFAR dataset. This repo is modified to goal of this repository is to provide a simple and flexible codebase for deep active learning. It is designed to support rapid implementation and evaluation of research ideas. We also provide a results on CIFAR10 below.

The codebase currently only supports single-machine single-gpu training. We will soon scale it to single-machine multi-gpu training, powered by the PyTorch distributed package.

## Using the Toolkit

Please see [`GETTING_STARTED`](docs/GETTING_STARTED.md) for brief instructions on installation, adding new datasets, basic usage examples, etc.

## Active Learning Methods used
* Entropy
* Random
* Deep Bayesian Active Learning (DBAL)
* Bayesian Active Learning by Disagreement (BALD)


## Datasets Supported
The dataset used in this research was published in Kaggle by (Dhabi, 2019) owners of the casting industry Pilot Techno Cast.
* [Casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/reallife-industrial-dataset-of-casting-product)
* 

## Results on CIFAR10 and CIFAR100 

The following are the results on CIFAR10 and CIFAR100, trained with hyperameters present in `configs/cifar10/al/RESNET18.yaml` and `configs/cifar100/al/RESNET18.yaml` respectively. All results were averaged over 3 runs. 

<!-- ![alt text](docs/AL_results.png) -->

<div align="center">
<img src="docs/AL_results.png"/>
</div>

###  CIFAR10 at 60%
```
|    AL Method     |        Test Accuracy        |
|:----------------:|:---------------------------:|
|            DBAL  |       91.670000 +- 0.230651 |
| Least Confidence |       91.510000 +- 0.087178 |
|            BALD  |       91.470000 +- 0.293087 |
|         Coreset  |       91.433333 +- 0.090738 |
|     Max-Entropy  |       91.373333 +- 0.363639 |
|      Min-Margin  |       91.333333 +- 0.234592 |
|   Ensemble-varR  |       89.866667 +- 0.127410 |
|          Random  |       89.803333 +- 0.230290 |
|            VAAL  |       89.690000 +- 0.115326 |
```

### CIFAR100 at 60%
```
|    AL Method     |        Test Accuracy        |
|:----------------:|:---------------------------:|
|            DBAL  |       55.400000 +- 1.037931 |
|         Coreset  |       55.333333 +- 0.773714 |
|     Max-Entropy  |       55.226667 +- 0.536128 |
|            BALD  |       55.186667 +- 0.369639 |
| Least Confidence |       55.003333 +- 0.937248 |
|      Min-Margin  |       54.543333 +- 0.611583 |
|   Ensemble-varR  |       54.186667 +- 0.325628 |
|            VAAL  |       53.943333 +- 0.680686 |
|          Random  |       53.546667 +- 0.302875 |
```

## Citing this Repository

If you find this repo helpful in your research, please consider citing us and the owners of the original toolkit:

```
@article{Annapoorni Pranesh,
    Author = {Annapoorni Pranesh, and Vineeth N Balasubramanian},
    Title = {Deep Active Learning for Defect Classification in PyTorch},
    Journal = {https://github.com/acl21/deep-active-learning-pytorch},
    Year = {2023}
}

@article{Munjal2020TowardsRA,
  title={Towards Robust and Reproducible Active Learning Using Neural Networks},
  author={Prateek Munjal and N. Hayat and Munawar Hayat and J. Sourati and S. Khan},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.09564}
}
```

## License

This toolkit is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## References

[1] Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep bayesian active learning with image data. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1183–1192. JMLR. org, 2017.

[2] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018.

[3] Sinha, Samarth et al. Variational Adversarial Active Learning. 2019 IEEE/CVF International Conference on Computer Vision (ICCV) (2019): 5971-5980.

[4] William H. Beluch, Tim Genewein, Andreas Nürnberger, and Jan M. Köhler. The power of ensembles for active learning in image classification. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9368–9377, 2018.