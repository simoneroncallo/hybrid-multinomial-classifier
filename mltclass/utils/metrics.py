import torch
import numpy as np
import pandas as pd

def get_accuracy(models, test_X, test_Y, num_classes, num_models, mode):
    accuracy = {mode: []}
    comb = torch.tensor([(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)]) # Label combinatorics (for mode = "one_vs_one")
    for idx in range(num_classes):
        mask = (test_Y[:, 0] == idx).squeeze() # Mask classes
        X = test_X[mask].reshape(mask.sum(), -1) # Mask dataset
        out = torch.zeros(X.shape[0], num_models) # Binary model outputs
        bin_pred = torch.zeros(X.shape[0], num_models, dtype = int) 
        label_pred = torch.zeros_like(bin_pred) 
    
        # Compute the predictions of each binary model
        for j in range(num_models):
            with torch.no_grad():
                out[:, j] = models[j](X).detach().squeeze()

        # Compute multinomial predictions
        if mode == "one_vs_rest":
            mdl_pred = torch.argmax(out, dim=1) # Label assigned with torch.argmax
        elif mode == "one_vs_one": 
            for j in range(num_models):
                bin_pred[:, j] = (out[:, j] >= 0.5).int() # Thresholded predictions
                label_pred[:, j] = comb[j, bin_pred[:,j]] # Multinomial predictions
            mdl_pred, _ = torch.mode(label_pred, dim=1) # Majority voting with torch.mode, ties are assigned to the smallest label. Noise?
        else: raise ValueError("Rule not available.")
        tmp = (mdl_pred == test_Y[mask, 0]).float().mean() * 100
        accuracy[mode].append(float(tmp)) # Accuracy per class

    accuracy[mode].append(np.mean(accuracy[mode])) # Accuracy averaged over classes
    return accuracy