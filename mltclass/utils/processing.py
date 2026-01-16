import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import warnings
from tqdm import tqdm
import warnings
from numpy.exceptions import VisibleDeprecationWarning

def load_dataset(dataset_name: str, download: bool = True) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    # PyTorch
    transform = transforms.ToTensor()
    def data2numpy(x) -> np.ndarray:
        if hasattr(x, "numpy"):
            return x.numpy()
        else:
            return np.asarray(x)
    
    if dataset_name == "MNIST":
        trainDataset =  datasets.MNIST(root = "~/datasets/mnist/train", train = True, download = download, transform = transform)
        testDataset = datasets.MNIST(root = "~/datasets/mnist/test", train = False, download = download, transform = transform)
    elif dataset_name == "Fashion":
        trainDataset = datasets.FashionMNIST(root = "~/datasets/fashion/train", train = True, download = download, transform = transform) 
        testDataset = datasets.FashionMNIST(root = "~/datasets/fashion/test", train = False, download = download, transform = transform)
    elif dataset_name == "CIFAR":
        trainDataset = datasets.CIFAR10(root = "~/datasets/cifar/train", train = True, download = download, transform = transform) 
        testDataset = datasets.CIFAR10(root = "~/datasets/cifar/test", train = False, download = download, transform = transform)
        warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
    else: raise ValueError("Dataset not available.")
    
    # TensorFlow
    # (X, Y), (XAll, YAll) = tf.keras.datasets.mnist.load_data() # MNIST
    # (X, Y), (XAll, YAll)= tf.keras.datasets.fashion_mnist.load_data() # Fashion MNIST
    # (X, Y), (XAll, YAll) = tf.keras.datasets.cifar10.load_data() # CIFAR-10

    X, XAll = data2numpy(trainDataset.data), data2numpy(testDataset.data)
    Y, YAll = data2numpy(trainDataset.targets), data2numpy(testDataset.targets)
    return ((X, Y), (XAll, YAll))

def split_dataset(user_X: np.ndarray, user_Y: np.ndarray, test_X: np.ndarray, test_Y: np.ndarray, mode: str, balanced_dataset: bool, trainval_ratio: float, generator, show_population: bool = True) -> tuple[int, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    user_X = user_X.astype(np.float64)
    user_Y = user_Y.astype(np.float64)[:,np.newaxis]
    test_X = test_X.astype(np.float64)
    test_Y = test_Y.astype(np.float64)[:,np.newaxis]

    # Squeeze extra dimension on the target labels
    if user_Y.ndim == 3: user_Y = np.squeeze(user_Y, axis=-1) 
    if test_Y.ndim == 3: test_Y = np.squeeze(test_Y, axis=-1) 

    # Convert to grayscale (if needed) and perform amplitude encoding (L2)
    tmpList = []
    for elX in [user_X, test_X]:
        if elX.ndim > 4:
            raise ValueError('Dataset not supported')
        elif elX.ndim == 4:
            elX = np.mean(elX, axis = -1) # Grayscale
        ampX = np.sqrt(elX).reshape(elX.shape[0], -1)
        normX = np.linalg.norm(ampX, axis = 1)
        if (normX == 0).any(): 
            raise ValueError('Normalization is zero')
        tmpList.append(ampX/normX[:, np.newaxis]) # Normalization
    user_X = tmpList[0]
    Xtest = torch.tensor(tmpList[1], dtype=torch.float64)
    Ytest = torch.tensor(test_Y, dtype=torch.float64)
    del tmpList

    # Split dataset
    X, Y = [], []
    num_classes = np.unique(user_Y).size
    if mode == "one_vs_rest":
        num_models = num_classes
        for idx in range(num_classes):
            one_idx = np.where(user_Y[:,0] == idx)[0] # One
            zero_idx = np.where(user_Y[:,0] != idx)[0] # Rest
        
            if balanced_dataset == True:
                num_samples = min(len(one_idx), len(zero_idx)) # Balanced   
                one_sample = generator.choice(one_idx, size = num_samples, replace = False)
                zero_sample = generator.choice(zero_idx, size = num_samples, replace = False)
                tmp_X = np.vstack((user_X[one_sample], user_X[zero_sample]))
                tmp_Y = np.vstack((np.ones((num_samples, 1)), np.zeros((num_samples, 1))))
            else:
                tmp_X = np.vstack((user_X[one_idx], user_X[zero_idx]))
                tmp_Y = np.vstack((np.ones((len(one_idx), 1)), np.zeros((len(zero_idx), 1))))
            
            shuffled = generator.permutation(tmp_X.shape[0])
            X.append(torch.tensor(tmp_X[shuffled], dtype=torch.float64))
            Y.append(torch.tensor(tmp_Y[shuffled], dtype=torch.float64))
            
    elif mode == "one_vs_one": 
        num_models = num_classes*(num_classes-1)//2
        comb = [(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)]
        for idx0, idx1 in comb:
            one_idx = np.where(user_Y[:,0] == idx1)[0] # One
            zero_idx = np.where(user_Y[:,0] == idx0)[0] # One 

            if balanced_dataset == True: warnings.warn("Ignoring balanced_dataset.")
            tmp_X = np.vstack((user_X[one_idx], user_X[zero_idx]))
            tmp_Y = np.vstack((np.ones((len(one_idx), 1)), np.zeros((len(zero_idx), 1))))
            
            shuffled = generator.permutation(tmp_X.shape[0])
            X.append(torch.tensor(tmp_X[shuffled], dtype=torch.float64))
            Y.append(torch.tensor(tmp_Y[shuffled], dtype=torch.float64))
                
    else: raise ValueError("Rule not available.")

    # Split train and test datasets
    Xtrain, Ytrain, Xval, Yval = [], [], [], []
    for x, y in zip(X, Y):
        split = int(trainval_ratio * x.shape[0])
        idx = torch.randperm(x.shape[0])
        train_idx, val_idx = idx[:split], idx[split:]
        
        Xtrain.append(x[train_idx])
        Ytrain.append(y[train_idx])
        Xval.append(x[val_idx])
        Yval.append(y[val_idx])

    # Print shapes
    rows = []
    idx0, idx1 = 1, 0
    if show_population == True:
        for idx in range(num_models):
            shape_Xtrain = tuple(Xtrain[idx].shape)
            shape_Ytrain = tuple(Ytrain[idx].shape)
            shape_Xval = tuple(Xval[idx].shape)
            shape_Yval = tuple(Yval[idx].shape)
            if mode == "one_vs_rest":
                rows.append({"Dataset": f"{idx}s vs. Rest", "Xtrain": shape_Xtrain, "Ytrain": shape_Ytrain,\
                             "Xval": shape_Xval, "Yval": shape_Yval})
            elif mode == "one_vs_one":
                idx1, idx0 = comb[idx]
                rows.append({"Dataset": f"{idx1}s vs. {idx0}s", "Xtrain": shape_Xtrain, "Ytrain": shape_Ytrain,\
                             "Xval": shape_Xval, "Yval": shape_Yval})
            
        df = pd.DataFrame(rows) # Create DataFrame
        print(df.to_string(index=False))
    
    return ((num_classes, num_models), (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest))

def normalize_dataset(user_X: np.ndarray, user_Y: np.ndarray, test_X: np.ndarray, test_Y: np.ndarray):
    user_X = user_X.astype(np.float64)
    user_Y = user_Y.astype(np.float64)[:,np.newaxis]
    test_X = test_X.astype(np.float64)
    test_Y = test_Y.astype(np.float64)[:,np.newaxis]
    num_models = np.unique(user_Y).size
    num_classes = np.unique(user_Y).size

    if user_Y.ndim == 3: user_Y = np.squeeze(user_Y, axis=-1) # Squeeze extra dimension on the target labels
    if test_Y.ndim == 3: test_Y = np.squeeze(test_Y, axis=-1) 

    tmpList = []
    for elX in [user_X, test_X]:
        if elX.ndim > 4:
            raise ValueError('Dataset not supported')
        elif elX.ndim == 4:
            elX = np.mean(elX, axis = -1) # Grayscale conversion
        ampX = np.sqrt(elX).reshape(elX.shape[0], -1)
        normX = np.linalg.norm(ampX, axis = 1)
        if (normX == 0).any(): 
            raise ValueError('Normalization is zero')
        tmpList.append(ampX/normX[:, np.newaxis]) # Normalize
        
    Xtrain = torch.tensor(tmpList[0], dtype=torch.float64)
    Xtest = torch.tensor(tmpList[1], dtype=torch.float64)
    
    Ytrain = torch.tensor(user_Y, dtype=torch.float64)
    Ytest = torch.tensor(test_Y, dtype=torch.float64)
    del tmpList

    return ((num_classes, num_models), (Xtrain, Ytrain), (Xtest, Ytest))
    
