import torch
import numpy as np
import pandas as pd
from typing import Tuple
from . import get_multinomial

def get_accuracy(objects: Tuple, test_X, test_Y, num_classes, num_models, mode, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32):

    # Unpack None for full compatibility between tree andversus (which requires only models)
    models, tree, partition, tree_map = objects 
    
    # Label combinatorics (for mode = "one_vs_one")
    comb = torch.tensor([(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)], dtype = dtype, device = device)

    accuracy = {mode: []}
    for idx in range(num_classes):
        mask = (test_Y[:, 0] == idx).squeeze() # Mask classes
        X = test_X[mask].reshape(mask.sum(), -1) # Mask dataset
        out = torch.zeros(X.shape[0], num_models, dtype = dtype, device = device) # Binary model outputs
        bin_pred = torch.zeros(X.shape[0], num_models, dtype = torch.long, device = device) 
        label_pred = torch.zeros(X.shape[0], num_models, dtype = torch.long, device = device) 

        if mode == "tree":
            # Compute the predictions of each binary model
            res = []
            for x in X:
                x = x.unsqueeze(0)
                with torch.no_grad():
                    out = get_multinomial(x, models, tree, partition, tree_map)
                    res.append(out)
            mdl_pred = torch.tensor(res).to(torch.long)

        elif mode == "one_vs_rest" or mode == "one_vs_one":
            # Compute the predictions of each binary model
            for j in range(num_models):
                with torch.no_grad():
                    out[:, j] = models[j](X).squeeze()
    
            # Compute multinomial predictions
            if mode == "one_vs_rest":
                mdl_pred = torch.argmax(out, dim=1).to(torch.long) # Label assigned with torch.argmax
                
            elif mode == "one_vs_one": 
                for j in range(num_models):
                    bin_pred[:, j] = (out[:, j] >= 0.5).to(torch.long) # Thresholded predictions
                    label_pred[:, j] = comb[j, bin_pred[:,j]] # Multinomial predictions
                mdl_pred, _ = torch.mode(label_pred, dim=1) # Majority voting with torch.mode, ties are assigned to the smallest label. Noise?
            
        else: 
            raise ValueError("Rule not available.")

        y_true = test_Y[mask, 0].to(torch.long)
        tmp = (mdl_pred == y_true).float().mean() * 100
        accuracy[mode].append(float(tmp)) # Accuracy per class

    accuracy[mode].append(np.mean(accuracy[mode])) # Accuracy averaged over classes
    return accuracy