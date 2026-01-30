from torchvision import datasets

# Download datasets
datasets.MNIST(root = "/home/jupyteruser/.ml-datasets/mnist/train", train = True, download = True)
datasets.MNIST(root = "/home/jupyteruser/.ml-datasets/mnist/test", train = False, download = True)
datasets.FashionMNIST(root = "/home/jupyteruser/.ml-datasets/fashion/train", train = True, download = True)
datasets.FashionMNIST(root = "/home/jupyteruser/.ml-datasets/fashion/test", train = False, download = True)
datasets.CIFAR10(root = "/home/jupyteruser/.ml-datasets/cifar/train", train = True, download = True)
datasets.CIFAR10(root = "/home/jupyteruser/.ml-datasets/cifar/test", train = False, download = True)
