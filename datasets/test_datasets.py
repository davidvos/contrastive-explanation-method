from mnist import MNIST
from fashion_mnist import FashionMNIST

dataset = FashionMNIST(download=False)

print(dataset.get_sample(train=True, show_image=True))