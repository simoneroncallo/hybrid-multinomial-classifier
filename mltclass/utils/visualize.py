import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_history(history_train, history_val, show_legend: str = True) -> Figure:
    """ History plot for training and validation dataset """
    num_classes = history_train.shape[0]
    num_plots = 4
    
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 4), dpi = 400)
    for idx in range(num_classes):
        axs[0].plot(history_train[idx,:,0], label = f'{idx}')
        axs[1].plot(history_val[idx,:,0], label = f'{idx}')
        axs[2].plot(history_train[idx,:,1], label = f'{idx}')
        axs[3].plot(history_val[idx,:,1], label = f'{idx}')
    
    titles = ['Loss (Train)', 'Loss (Val)', 'Accuracy (Train)', 'Accuracy (Val)']
    for idx in range(num_plots):
        axs[idx].set_title(titles[idx])
        if show_legend == True:
            axs[idx].legend(ncol=num_classes//3)
    
    for idx in [2, 3]:
        axs[idx].yaxis.grid(True, linestyle='--', alpha=0.3) # Grid
    
    ylim_loss = [min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]), max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])]
    ylim_acc = [min(axs[2].get_ylim()[0], axs[3].get_ylim()[0]), max(axs[2].get_ylim()[1], axs[3].get_ylim()[1])]
    axs[0].set_ylim(ylim_loss), axs[1].set_ylim(ylim_loss) # Loss axes
    axs[2].set_ylim(ylim_acc), axs[3].set_ylim(ylim_acc) # Accuracy axes
    fig.tight_layout()
    plt.close()

    return fig

def plot_weights(weights):
    """ Show weights as images """
    probs = np.array([np.abs(w.detach().numpy())**2 for w in weights])  # shape (num_classes, num_features)
    num_classes = probs.shape[0]
    img_size = int(np.sqrt(probs.shape[1]))
    num_rows, num_cols = 2, 5
    
    vmax = probs.max()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2.5, num_rows*2.5), dpi=200)
    for img_idx in range(num_classes):
        row = img_idx // num_cols
        col = img_idx % num_cols
        ax = axes[row, col]
        im = ax.imshow(probs[img_idx].reshape((img_size, img_size)), cmap='viridis', vmin = 0, vmax = vmax)
        ax.set_title(f'{img_idx}s vs Rest')
        ax.axis('off')
    
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.019, pad=0.02)
    cbar.set_label('Probabilities', rotation=270, labelpad=15)
    plt.show()