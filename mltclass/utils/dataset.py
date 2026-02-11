import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import warnings
from tqdm import tqdm
import warnings
from numpy.exceptions import VisibleDeprecationWarning

def load_dataset(dataset_name: str, download: bool = True, labels: list[int] | None = None) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    # PyTorch
    transform = transforms.ToTensor()
    def data2numpy(x) -> np.ndarray:
        if hasattr(x, "numpy"):
            return x.numpy()
        else:
            return np.asarray(x)
    
    if dataset_name == "MNIST":
        trainDataset =  datasets.MNIST(root = "~/.ml-datasets/mnist/train", train = True, download = download, transform = transform)
        testDataset = datasets.MNIST(root = "~/.ml-datasets/mnist/test", train = False, download = download, transform = transform)
    elif dataset_name == "Fashion":
        trainDataset = datasets.FashionMNIST(root = "~/.ml-datasets/fashion/train", train = True, download = download, transform = transform) 
        testDataset = datasets.FashionMNIST(root = "~/.ml-datasets/fashion/test", train = False, download = download, transform = transform)
    elif dataset_name == "CIFAR":
        trainDataset = datasets.CIFAR10(root = "~/.ml-datasets/cifar/train", train = True, download = download, transform = transform) 
        testDataset = datasets.CIFAR10(root = "~/.ml-datasets/cifar/test", train = False, download = download, transform = transform)
        warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
    else: raise ValueError("Dataset not available.")
    
    # TensorFlow
    # (X, Y), (XAll, YAll) = tf.keras.datasets.mnist.load_data() # MNIST
    # (X, Y), (XAll, YAll)= tf.keras.datasets.fashion_mnist.load_data() # Fashion MNIST
    # (X, Y), (XAll, YAll) = tf.keras.datasets.cifar10.load_data() # CIFAR-10

    #X, XAll = data2numpy(trainDataset.data), data2numpy(testDataset.data)
    #Y, YAll = data2numpy(trainDataset.targets), data2numpy(testDataset.targets)

    X, XAll = data2numpy(trainDataset.data), data2numpy(testDataset.data)
    Y, YAll = data2numpy(trainDataset.targets), data2numpy(testDataset.targets)
    
    # Filter the labels
    if labels is not None:
        labels = np.asarray(labels)
        trainX, testX = np.copy(X), np.copy(XAll)
        trainY, testY = np.copy(Y), np.copy(YAll)
        train_mask = np.isin(trainY, labels)
        test_mask = np.isin(testY, labels)

        X = trainX[train_mask]
        Y = trainY[train_mask]
        XAll = testX[test_mask]
        YAll = testY[test_mask]
    
    return ((X, Y), (XAll, YAll))

def split_versus_dataset(user_X: np.ndarray, user_Y: np.ndarray, test_X: np.ndarray, test_Y: np.ndarray, mode: str, balanced_dataset: bool, trainval_ratio: float, generator, show_population: bool = True, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> tuple[int, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:

    # Preliminaries
    device = torch.device(device) if device is not None else torch.device("cpu")
    np_dtype = torch.tensor([], dtype=dtype).numpy().dtype # Match numpy and torch dtype
    
    user_X = user_X.astype(np_dtype, copy = False)
    user_Y = user_Y.astype(np_dtype, copy = False)[:,np.newaxis]
    test_X = test_X.astype(np_dtype, copy = False)
    test_Y = test_Y.astype(np_dtype, copy = False)[:,np.newaxis]

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
    Xtest = torch.tensor(tmpList[1], dtype = dtype, device = device)
    Ytest = torch.tensor(test_Y, dtype = dtype, device = device)
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
            X.append(torch.tensor(tmp_X[shuffled], dtype = dtype, device = device))
            Y.append(torch.tensor(tmp_Y[shuffled], dtype = dtype, device = device))
            
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
            X.append(torch.tensor(tmp_X[shuffled], dtype = dtype, device = device))
            Y.append(torch.tensor(tmp_Y[shuffled], dtype = dtype, device = device))
                
    else: raise ValueError("Rule not available.")

    # Split train and test datasets
    Xtrain, Ytrain, Xval, Yval = [], [], [], []
    for x, y in zip(X, Y):
        split = int(trainval_ratio * x.shape[0])
        idx = torch.randperm(x.shape[0], device = x.device)
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

def split_tree_dataset(user_X: np.ndarray, user_Y: np.ndarray, test_X: np.ndarray, test_Y: np.ndarray, tree, depth: float,\
generator, verbose: bool = True, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32):
    
    X, Y = [], []
    istraining = True # Verbose only for training
    
    for data_X, data_Y in [(user_X, user_Y), (test_X, test_Y)]:
        tmplist_X , tmplist_Y = [], []
        counter, legend = 0, {}
        for d in range(1, depth): # Main training loop
            if istraining and verbose: print(f"Depth: {d} -> {tree[d]}")
            for idx in range(0, len(tree[d]) - 1, 2): # Loop over leaves
                zeros = tree[d][idx]
                ones = tree[d][idx + 1]
                zero_idx = np.where(np.isin(data_Y[:,0], zeros))[0]
                one_idx = np.where(np.isin(data_Y[:,0], ones))[0]

                # Create binary models. Keep track of the original multinomial labels for navigating the decision tree with flattened models
                if not istraining: legend[counter] = zeros + ones # Multinomial labels included in the node, e.g. [3,[4,5]] (Test only)
                tmp_X = np.vstack((data_X[zero_idx], data_X[one_idx]))
                tmp_Y = np.vstack((np.zeros((len(zero_idx), 1)), np.ones((len(one_idx), 1)))) # Binarized labels, e.g. [3,[4,5]] -> [0,1]
                if istraining and verbose:
                    print('Class 0:', np.unique(data_Y[zero_idx]))
                    print('Class 1:', np.unique(data_Y[one_idx]))
                    print('---')
                
                shuffled = generator.permutation(tmp_X.shape[0])
                tmplist_X.append(torch.tensor(tmp_X[shuffled], device = device, dtype = dtype))
                tmplist_Y.append(torch.tensor(tmp_Y[shuffled], device = device, dtype = dtype))
                counter += 1
                
        istraining = False       
        X.append(tmplist_X)
        Y.append(tmplist_Y)

    # Print tree structure
    # print('Legend') # Map from indexes to labels
    # for key in legend.keys():
    #    print(f"Models {key} -> {legend[key]}")
    tree_map = {tuple(v): k for k, v in legend.items()} # Inverse map from labels to indexes
    
    return (len(X[0]), legend, tree_map), (X[0], Y[0]), (X[1], Y[1])

def normalize_dataset(user_X: np.ndarray, user_Y: np.ndarray, test_X: np.ndarray, test_Y: np.ndarray, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32):

    # Preliminaries
    device = torch.device(device) if device is not None else torch.device("cpu")
    np_dtype = torch.tensor([], dtype=dtype).numpy().dtype # Match numpy and torch dtype
    
    user_X = user_X.astype(np_dtype, copy = False)
    user_Y = user_Y.astype(np_dtype, copy = False)[:,np.newaxis]
    test_X = test_X.astype(np_dtype, copy = False)
    test_Y = test_Y.astype(np_dtype, copy = False)[:,np.newaxis]
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
        
    Xtrain = torch.tensor(tmpList[0], dtype = dtype, device = device)
    Xtest = torch.tensor(tmpList[1], dtype = dtype, device = device)
    
    Ytrain = torch.tensor(user_Y, dtype = dtype, device = device)
    Ytest = torch.tensor(test_Y, dtype = dtype, device = device)
    del tmpList

    return ((num_classes, num_models), (Xtrain, Ytrain), (Xtest, Ytest))
    
