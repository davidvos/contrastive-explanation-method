from mnist_dataset import MNISTDataset

dataset = MNISTDataset(download=True)

print(dataset.get_sample(train=True))