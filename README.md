# Re - Explanations based on the missing

This repo contains a reproduction of the method defined in the paper ["Explanations based on the Missing"](https://arxiv.org/pdf/1802.07623).

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

This repo comes with two pretrained sets of models. These models are contained in `models/saved_models/`. By default, the MNIST models will be loaded. To instead train the classifier and autoencoder from scratch, specify the `load_path` argument for the `train_cnn` and `train_ae` functions as an empty string: `""`.

```python
dataset = MNIST()

# load / train classifier model and weights
cnn = CNN
train_cnn(cnn, dataset)

# load / train autoencoder model and weights
cae = CAE()
train_ae(cae, dataset)
```
### Initialising the Contrastive Explanation Method class
The `ContrastiveExplanationMethod` class takes a classifier as positional argument. For a full overview of all arguments and their uses, see `cem/cem.py`.
```python

CEM = ContrastiveExplanationMethod(
    cnn,
    cae,
    iterations=1000,
    n_searches=9,
    kappa=10.0,
    gamma=1.0
    beta=0.1,
    learning_rate=0.1,
    c_init=10.0
)

```

### Producing pertinent positives or negatives
The explain function takes as input a single sample, and returns the perturbed image that satisfies the classification objective with the lowest corresponding overall loss as defined in [equation 1](https://arxiv.org/pdf/1802.07623.pdf). To obtain the delta for said perturbed image, see the following example code.
```python

# obtain a sample from the dataset, discard the label
sample, _ = dataset.get_sample()

# obtaining the PP
perturbed_image = CEM.explain(sample, mode="PP")
pp_delta = sample - perturbed_image

# obtaining the PN
perturbed_image = CEM.explain(sample, mode="PP")
pn_delta = perturbed_image - sample
```

## Usage of the command line implementation

Experiments can also be ran from the command line by calling 'main.py'. For an overview of all the arguments see below.

An example of the usage of this script for the FashionMNIST dataset is given below.

```bash
python main.py --verbose -mode PP -dataset FashionMNIST \
  -cnn_load_path ./models/saved_models/fashion-mnist-cnn.h5\
  -cae_load_path ./models/saved_models/fashion-mnist-cae.h5
```

### Command line arguments
```
usage: main.py [-h] [-dataset DATASET] [-cnn_load_path CNN_LOAD_PATH]                                                                                      [--no_cae] [-cae_load_path CAE_LOAD_PATH]                                                                                                   [-sample_from_class SAMPLE_FROM_CLASS] [--discard_images]                                                                                   [-mode MODE] [-kappa KAPPA] [-beta BETA] [-gamma GAMMA]                                                                                     [-c_init C_INIT] [-c_converge C_CONVERGE]                                                                                                   [-iterations ITERATIONS] [-n_searches N_SEARCHES]                                                                                           [-learning_rate LEARNING_RATE] [-input_shape INPUT_SHAPE]                                                                                   [--verbose] [-print_every PRINT_EVERY] [-device DEVICE]                                                                                                                                                                                                                  optional arguments:                                                                                                                           -h, --help            show this help message and exit                                                                                       -dataset DATASET      choose a dataset (MNIST or FashionMNIST) to apply the                                                                                       contrastive explanation method to.                                                                                    -cnn_load_path CNN_LOAD_PATH                                                                                                                                      path to load classifier weights from.                                                                                 --no_cae              disable the autoencoder                                                                                               -cae_load_path CAE_LOAD_PATH                                                                                                                                      path to load autoencoder weights from.                                                                                -sample_from_class SAMPLE_FROM_CLASS                                                                                                                              specify which class to sample from for pertinent                                                                                            negative or positive                                                                                                  --discard_images      specify whether or not to save the created images                                                                     -mode MODE            Either PP for pertinent positive or PN for pertinent                                                                                        negative.                                                                                                             -kappa KAPPA          kappa value used in the CEM attack loss.                                                                              -beta BETA            beta value used as L1 regularisation coefficient.                                                                     -gamma GAMMA          gamma value used as reconstruction regularisation                                                                                           coefficient                                                                                                           -c_init C_INIT        initial c value used as regularisation coefficient for                                                                                      the attack loss                                                                                                       -c_converge C_CONVERGE                                                                                                                                            c value to amend the value of c towards if no solution                                                                                      has been found in the current iterations                                                                              -iterations ITERATIONS                                                                                                                                            number of iterations per search                                                                                       -n_searches N_SEARCHES                                                                                                                                            number of searches                                                                                                    -learning_rate LEARNING_RATE                                                                                                                                      initial learning rate used to optimise the slack                                                                                            variable                                                                                                              -input_shape INPUT_SHAPE                                                                                                                                          shape of a single sample, used to reshape input for                                                                                         classifier and autoencoder input                                                                                      --verbose             print loss information during training                                                                                -print_every PRINT_EVERY                                                                                                                                          if verbose mode is enabled, interval to print the                                                                                           current loss                                                                                                          -device DEVICE        device to run experiment on 
```

## Extending to new datasets

To extend this implementation to a new dataset, inherit the 'Dataset' class specified in 'datasets.dataset.py' and overwrite the initialisation by specifying a train_data and test_data attribute containing a Pytorch Dataset, train_loader and test_loader attributes containing Pytorch Dataloaders and train_list and test_list attributes containing a list of samples.

## License
[MIT](https://choosealicense.com/licenses/mit/)
