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

    # --- NEW: ensemble settings (only active for mode == "tree") ---
    num_runs = 3
    seeds = [2025, 2026, 2027]
    # ----------------------------------------------------------------

    download = True
    rng = np.random.default_rng(2025)
    torch.manual_seed(2025)
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

        # Generate tree
        tree, partition, depth = get_tree(labelmask, num_classes, rng) 
        (Xtest, Ytest) = test
        
        # Split dataset
        (num_models, idx2labels, tree_map), train, val = split_tree_dataset(
            X0, Y0, Xtest, Ytest, tree, depth, rng, verbose = False
        )

    else: raise ValueError("Mode not available")

    (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest) = train, val, test

    # Create dataloaders
    train_loader = [
        DataLoader(TensorDataset(X,Y), batch_size=batch_size, shuffle=True)
        for X,Y in zip(Xtrain, Ytrain)
    ]
    val_loader = [
        DataLoader(TensorDataset(X,Y), batch_size=batch_size, shuffle=False) 
        for X,Y in zip(Xval, Yval)
    ]

    # ================================================================ #
    #                           TRAINING                               #
    # ================================================================ #

    # --- CHANGED: outer loop over runs; 3 runs for tree, 1 for everything else ---
    all_models_runs = []
    active_runs = num_runs if mode == "tree" else 1

    for run_idx in range(active_runs):
        torch.manual_seed(seeds[run_idx])  # --- NEW: per-run seed ---

        models, optimizers = [], []
        history_train = torch.zeros(
            (num_models, num_epochs, 2), device="cpu", dtype=torch.float32
        )
        history_val = torch.zeros(
            (num_models, num_epochs, 2), device="cpu", dtype=torch.float32
        )

        for idx in tqdm(range(num_models), ascii=' =',
                        desc=f"Run {run_idx+1}/{active_runs}"):  # --- CHANGED: added desc ---

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

        all_models_runs.append(models)  # --- NEW: store this run's models ---
    # -------------------------------------------------------------------------

    # Plot histories for run 0
    figure = plot_history(history_train, history_val, show_legend = False)
    maskinfo = ''.join(str(i) for i in labelmask)
    figure.savefig(f"./data/{arch}_{mode}_{dataset}_{maskinfo}_history.png")

    # ================================================================ #
    #                          INFERENCE                               #
    # ================================================================ #

    models_run0 = all_models_runs[0]  # --- NEW: shorthand for run 0 ---

    if mode == "tree":
        # --- NEW: voted accuracy loop for tree mode ---

        def flatten(x):
            """ Flatten nested tuples, e.g. ((1,2),(3,(4,5))) -> (1,2,3,4,5) """
            for item in x:
                if isinstance(item, tuple):
                    yield from flatten(item)
                else:
                    yield item

        def voted_multinomial(x, all_models_runs, tree, partition, tree_map):
            """
            Like get_multinomial, but uses majority vote across all runs at the
            root node only. Deeper nodes always use run 0 models.
            """
            left, right = partition

            # Identify root node model index via tree_map
            root_key = tuple(flatten(partition))
            root_idx = tree_map[root_key]

            # Majority vote: threshold each run at 0.5, count votes for right branch
            votes = sum(
                1 if all_models_runs[r][root_idx](x).item() > 0.5 else 0
                for r in range(len(all_models_runs))
            )
            next_partition = right if votes > len(all_models_runs) / 2 else left

            # Continue down the tree using run 0 models
            return get_multinomial(x, models_run0, tree, next_partition, tree_map)

        # Mirror the tree branch of get_accuracy (metrics.py) with voted_multinomial
        accuracy = {mode: []}
        for idx in range(num_classes):
            mask = (Ytest[:, 0] == idx).squeeze()
            X = Xtest[mask].reshape(mask.sum(), -1)
            res = []
            for x in X:
                with torch.no_grad():
                    res.append(voted_multinomial(
                        x.unsqueeze(0), all_models_runs, tree, partition, tree_map
                    ))
            mdl_pred = torch.tensor(res).to(torch.long)
            y_true = Ytest[mask, 0].to(torch.long)
            tmp = (mdl_pred == y_true).float().mean() * 100
            accuracy[mode].append(float(tmp))
        accuracy[mode].append(np.mean(accuracy[mode]))
        # ---------------------------------------------------------------

    else:
        # Unchanged: original inference for one_vs_rest and one_vs_one
        objects = models_run0, tree, partition, tree_map
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
