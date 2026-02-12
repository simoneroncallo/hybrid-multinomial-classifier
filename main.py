#!/usr/bin/env python
# coding: utf-8

# --- PRELIMINARIES --- #
import os
import argparse
import time
import warnings
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

from mltclass.utils import load_dataset, split_versus_dataset
from mltclass.utils import split_tree_dataset, normalize_dataset, plot_history
from mltclass.utils import plot_weights, get_accuracy, get_bisection
from mltclass.utils import get_leaves, get_nodes, get_tree, get_multinomial

# Suppress CIFAR dataset warning
warnings.filterwarnings("ignore", 
                        category=np.exceptions.VisibleDeprecationWarning, 
                        message=".*align should be passed as Python*")

path = "./data"
if not os.path.exists(path):
    os.makedirs(path) # Create folder for output data

def parse_args():
    """ Parse CLI arguments """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, 
                        choices = ["MNIST", "Fashion", "CIFAR"])
    parser.add_argument("--mode", type = str,
                        choices = ["one_vs_rest", "one_vs_one", "tree"])
    parser.add_argument("--arch", type = str,
                        choices = ["classical", "quantum"])

    args, _ = parser.parse_known_args()
    return args

def run_simulation(arch, mode, dataset):
    """ Simulation runtime called by main() """

    # --- PARAMETERS --- #
    num_epochs = 450
    num_hidden = 20
    num_layers = 20 # For ClassicalNetwork
    batch_size = 128
    learning_rate = 0.05
    
    optimizer = "SGD" # Available {SGD, Adam}
    balanced_dataset = False
    use_bias_sigmoid = True
    trainval_ratio = 0.8 # Ratio 4:1
    labelmask = [0,1,2,3,4,5] # Example [0,1,5,8]
    
    download = True
    rng = np.random.default_rng(2025)
    torch.manual_seed(2025)
    tree, partition, tree_map = None, None, None # Compatibility with tree
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 # Floating point precision
    start = time.perf_counter() # Stopwatch

    # --- ARCHITECTURE --- #
    if arch == "classical":
        from mltclass import ClassicalNetwork as Model # Classical model
    elif arch == "quantum":
        from mltclass import QuantumNetwork as Model # Quantum model
    else:
        raise ValueError("Architecture not available")

    # --- DATASET --- #
    (X, Y), (XAll, YAll) = load_dataset(
        dataset, download = download, labels = labelmask
    )

    if mode == "one_vs_rest" or mode == "one_vs_one":
        # Split dataset
        (num_classes, num_models), train, val, test = split_versus_dataset(
            X, Y, XAll, YAll, mode, balanced_dataset, trainval_ratio, rng, 
            show_population = False, device = device, dtype = dtype
        )

    elif mode == "tree":
        # Normalize dataset
        (num_classes, _), (X0, Y0), test = normalize_dataset(
            X, Y, XAll, YAll, device = device, dtype = dtype
        )
        tree, partition, depth = get_tree(labelmask, num_classes, rng) 
        (Xtest, Ytest) = test
        
        # Split dataset
        (num_models, idx2labels, tree_map), train, val = split_tree_dataset(
            X0, Y0, Xtest, Ytest, tree, depth, rng, verbose = False
        )

    else: 
        raise ValueError("Mode not available")

    # Build dataloader
    (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest) = train, val, test
    train_loader = [
        DataLoader(TensorDataset(X,Y), batch_size=batch_size, shuffle=True)
        for X,Y in zip(Xtrain, Ytrain)
    ]
    val_loader = [
        DataLoader(TensorDataset(X,Y), batch_size=batch_size, shuffle=False) 
        for X,Y in zip(Xval, Yval)
    ]

    # --- TRAINING --- #
    history_train = torch.zeros(
        (num_models,num_epochs,2), device = "cpu", dtype = torch.float32
    )
    history_val = torch.zeros(
        (num_models,num_epochs,2), device = "cpu", dtype = torch.float32
    )
    hidden_weights, output_weights, models, optimizers = [], [], [], []
    for idx in tqdm(range(num_models), ascii=' ='):
        models.append(Model(
            Xtrain[idx].shape[1], num_hidden, 
            use_bias_sigmoid = use_bias_sigmoid, num_layers = num_layers,
            device = device, dtype = dtype)
        )
        loss_function = torch.nn.BCELoss() # Loss
        method = getattr(torch.optim, optimizer)
        optimizers.append(
            method(models[idx].parameters(), lr = learning_rate)
        ) # Optimizer
        (_,_), history_train[idx,:,:], history_val[idx,:,:] = models[idx].fit(
            train_loader[idx], val_loader[idx], num_epochs, loss_function, 
            optimizers[idx]
        ) # Train

    # Plot
    figure = plot_history(history_train, history_val, show_legend = False)
    figure.savefig(f"./data/{arch}_{mode}_{dataset}_history.png")

    # --- INFERENCE --- #
    objects = models, tree, partition, tree_map
    acc = get_accuracy(
        objects, Xtest, Ytest, num_classes, num_models, mode, 
        device = device, dtype = dtype
    ) # Accuracies
    df = pd.DataFrame.from_dict(acc)
    df = df.rename(index={df.index[-1]: 'Avg.'})
    df.to_string(f"./data/{arch}_{mode}_{dataset}_accuracy.txt")

    runtime = time.perf_counter() - start # Stopwatch
    print(f"Average accuracy of {df[mode].iloc[-1]:.1f}%")
    print(f"Runtime of {runtime:.2f}s ({runtime/3600:.2f}h)")

def main():
    """ Main function """

    args = parse_args()  
    arch = args.arch
    mode = args.mode
    dataset = args.dataset

    # Verbose
    print("# --- START --- #")
    print(f"Settings <- {arch}, {mode}, {dataset}")
    run_simulation(arch, mode, dataset)
    print("# ---  END  --- #\n")

if __name__ == "__main__":
    main()
    
