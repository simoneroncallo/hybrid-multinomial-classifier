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
    parser.add_argument("--labelmask", type = int, nargs = "+",
                        default = [0, 1, 2, 3],
                        help = "Class labels to use (e.g. --labelmask 0 1 8)")                    
    
    args, _ = parser.parse_known_args()

    if len(args.labelmask) < 2:
        parser.error("--labelmask requires at least 2 integers")
    
    return args

def run_simulation(arch, mode, dataset, labelmask):
    """ Simulation runtime called by main() """

    # ================================================================ #
    #                          PARAMETERS                              #
    # ================================================================ #
    
    num_epochs = 450
    num_hidden = 20
    num_layers = 10 # For ClassicalNetwork
    batch_size = 128
    learning_rate = 0.05
    
    optimizer = "SGD" # Available {SGD, Adam}
    standardization = True
    balanced_dataset = False
    use_bias_sigmoid = True
    trainval_ratio = 0.8 # Ratio 4:1

    # --- NEW: number of trees for ensemble (hyperparameter; increase if ties are frequent) ---
    num_runs = 3
    # ------------------------------------------------------------------------------------

    download = True
    torch.manual_seed(2025)  # --- CHANGED: single fixed seed; diversity comes from partitions, not stochasticity ---
    tree, partition, tree_map = None, None, None # Compatibility with tree
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 # Floating point precision
    start = time.perf_counter() # Stopwatch

    # ================================================================ #
    #                         ARCHITECTURE                             #
    # ================================================================ #
    
    if arch == "classical":
        from mltclass import ClassicalNetwork as Model # Classical model
    elif arch == "quantum":
        from mltclass import QuantumNetwork as Model # Quantum model
    else:
        raise ValueError("Architecture not available")

    # ================================================================ #
    #                           DATASET                                #
    # ================================================================ #
    
    (X, Y), (XAll, YAll) = load_dataset(
        dataset, download = download, labels = labelmask, 
        standardization = standardization
    )

    if mode == "one_vs_rest" or mode == "one_vs_one":

        rng = np.random.default_rng(2025)  # Single rng for non-tree modes

        # Split dataset
        (num_classes, num_models), train, val, test = split_versus_dataset(
            X, Y, XAll, YAll, mode, balanced_dataset, trainval_ratio, rng, 
            show_population = False, device = device, dtype = dtype
        )
        (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest) = train, val, test

    elif mode == "tree":

        # --- CHANGED: rng is instantiated once; sequential get_tree calls advance its state ---
        rng = np.random.default_rng(2025)

        # Normalize dataset once; test set is shared across all runs
        (num_classes, _), (X0, Y0), test = normalize_dataset(
            X, Y, XAll, YAll, device = device, dtype = dtype
        )
        (Xtest, Ytest) = test

    else: raise ValueError("Mode not available")

    # ================================================================ #
    #              TRAINING  [CHANGED: per-run tree + splits]          #
    # ================================================================ #

    # --- NEW: stores (models, tree, partition, tree_map) for each run ---
    all_runs = []
    active_runs = num_runs if mode == "tree" else 1

    for run_idx in range(active_runs):

        if mode == "tree":
            # --- NEW: each call advances rng state, yielding a different bisection ---
            tree, partition, depth = get_tree(labelmask, num_classes, rng)

            # Split dataset according to this run's partition
            (num_models, idx2labels, tree_map), train, val = split_tree_dataset(
                X0, Y0, Xtest, Ytest, tree, depth, rng, verbose = False
            )
            (Xtrain, Ytrain), (Xval, Yval), _ = train, val, test

        # Create dataloaders for this run's partition
        train_loader = [
            DataLoader(TensorDataset(Xb, Yb), batch_size=batch_size, shuffle=True)
            for Xb, Yb in zip(Xtrain, Ytrain)
        ]
        val_loader = [
            DataLoader(TensorDataset(Xb, Yb), batch_size=batch_size, shuffle=False)
            for Xb, Yb in zip(Xval, Yval)
        ]

        models, optimizers = [], []
        history_train = torch.zeros(
            (num_models, num_epochs, 2), device="cpu", dtype=torch.float32
        )
        history_val = torch.zeros(
            (num_models, num_epochs, 2), device="cpu", dtype=torch.float32
        )

        for idx in tqdm(range(num_models), ascii=' =',
                        desc=f"Run {run_idx+1}/{active_runs}"):

            # Create model
            models.append(Model(
                Xtrain[idx].shape[1], num_hidden, 
                use_bias_sigmoid = use_bias_sigmoid, num_layers = num_layers,
                device = device, dtype = dtype)
            )

            # Define loss function
            loss_function = torch.nn.BCELoss()

            # Create optimizer
            method = getattr(torch.optim, optimizer) 
            optimizers.append(method(
                models[idx].parameters(), lr = learning_rate,
                momentum = 0.9, weight_decay = 1e-4)
            ) 

            # Define optimizer scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[idx], T_max = num_epochs, eta_min=1e-6
            )

            # Start training
            (_,_), history_train[idx,:,:], history_val[idx,:,:] = models[idx].fit(
                train_loader[idx], val_loader[idx], num_epochs, loss_function, 
                optimizers[idx], scheduler
            )

        # --- NEW: store all run-specific objects together ---
        all_runs.append((models, tree, partition, tree_map))

    # Plot histories for run 0
    figure = plot_history(history_train, history_val, show_legend = False)
    maskinfo = ''.join(str(i) for i in labelmask)
    figure.savefig(f"./data/{arch}_{mode}_{dataset}_{maskinfo}_history.png")

    # ================================================================ #
    #                          INFERENCE                               #
    # ================================================================ #

    models_run0, tree_run0, partition_run0, tree_map_run0 = all_runs[0]

    if mode == "tree":
        # --- NEW: full-tree voted accuracy loop ---

        accuracy = {mode: []}
        num_ties = 0  # --- NEW: counts samples where all 3 trees disagree ---

        for idx in range(num_classes):
            mask = (Ytest[:, 0] == idx).squeeze()
            X = Xtest[mask].reshape(mask.sum(), -1)
            res = []
            for x in X:
                x = x.unsqueeze(0)
                with torch.no_grad():
                    # One full prediction per run (each with its own tree + models)
                    preds = [
                        get_multinomial(x, models, tree, partition, tree_map)
                        for models, tree, partition, tree_map in all_runs
                    ]

                # Majority vote; torch.mode breaks ties toward the smallest label
                preds_tensor = torch.tensor(preds)
                vote, _ = torch.mode(preds_tensor)
                voted_label = vote.item()

                # Detect full disagreement (all predictions distinct) -> fall back to run 0
                if len(set(preds)) == len(preds):
                    num_ties += 1
                    voted_label = preds[0]

                res.append(voted_label)

            mdl_pred = torch.tensor(res).to(torch.long)
            y_true = Ytest[mask, 0].to(torch.long)
            tmp = (mdl_pred == y_true).float().mean() * 100
            accuracy[mode].append(float(tmp))

        accuracy[mode].append(np.mean(accuracy[mode]))
        print(f"Total ties (fell back to run 0): {num_ties} / {Ytest.shape[0]}")
        # ---------------------------------------------------------------

    else:
        # Unchanged: original inference for one_vs_rest and one_vs_one
        objects = models_run0, tree_run0, partition_run0, tree_map_run0
        accuracy = get_accuracy(
            objects, Xtest, Ytest, num_classes, num_models, mode, 
            device = device, dtype = dtype
        )

    df = pd.DataFrame.from_dict(accuracy)
    df = df.rename(index={df.index[-1]: 'Avg.'})
    maskinfo = ''.join(str(i) for i in labelmask)
    df.to_string(f"./data/{arch}_{mode}_{dataset}_{maskinfo}_accuracy.txt")

    runtime = time.perf_counter() - start # Stopwatch
    print(f"Average accuracy of {df[mode].iloc[-1]:.1f}%")
    print(f"Runtime of {runtime:.2f}s ({runtime/3600:.2f}h)")

def main():
    """ Main function """

    args = parse_args()  
    arch = args.arch
    mode = args.mode
    dataset = args.dataset
    labelmask = args.labelmask

    # Verbose
    print("# --- START --- #")
    print(f"Settings <- {arch}, {mode}, {dataset}, {labelmask}")
    run_simulation(arch, mode, dataset, labelmask)
    print("# ---  END  --- #\n")

if __name__ == "__main__":
    main()
