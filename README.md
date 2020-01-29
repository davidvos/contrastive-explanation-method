# Re - Explanations based on the missing

This repo contains a reproduction of the method defined in the paper "Explanations based on the Missing"[https://arxiv.org/pdf/1802.07623].

The implementation on this github repository is given for two datasets: MNIST and FashionMNIST, but allows for easy extensions to new datasets (see below). 

## Requirements

See requirement.txt.

## Installation

Download the github repo and navigate to its root folder.

## Usage of the python package

An example of the usage of the python implementation on the MNIST dataset is given below. 

### Load required modules
```python
from datasets.mnist import MNIST

from models.cae_model import CAE
from models.conv_model import CNN
from train import train_ae, train_cnn

from cem import ContrastiveExplanationMethod
```
### Dataset and models 

This repo comes with two pretrained sets of models. These models are contained in `models/saved_models/`. By default, the MNIST models will be loaded. To instead train the classifier and autoencoder from scratch, specify the `load_path` argument as an empty string: `""`.

```python
dataset = MNIST()

cnn = CNN
train_cnn(cnn, dataset)

# load / train autoencoder model and weights
cae = CAE()
train_ae(cae, dataset)


```

## Usage of the command line implementation

Experiments can also be ran from the command line by calling 'main.py'. For an overview of all the arguments see below.

An example of the usage of this script for the FashionMNIST dataset is given below.

```bash
python main.py --verbose -mode PP -dataset FashionMNIST \
  -cnn_load_path ./models/saved_models/fashion-mnist-cnn.h5\
  -cae_load_path ./models/saved_models/fashion-mnist-cae.h5
```

## Extending to new datasets

To extend this implementation to a new dataset, inherit the 'Dataset' class specified in 'datasets.dataset.py' and overwrite the initialisation by specifying a train_data and test_data attribute containing a Pytorch Dataset, train_loader and test_loader attributes containing Pytorch Dataloaders and train_list and test_list attributes containing a list of samples.

## License
[MIT](https://choosealicense.com/licenses/mit/)
