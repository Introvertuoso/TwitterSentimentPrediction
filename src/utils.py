# define some functions for plotting purposes
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_y_continous(y, bins=10, show=True, title=None):
    fig, ax = plt.subplots(1, 1)
    _ = ax.hist(y, bins=bins)
    if isinstance(title, str):
        ax.set_title(title)
    plt.tight_layout()
    if show: plt.show()


def plot_scatter(x, y, show=True, x_label=None, y_label=None, title=None):
    fig, ax = plt.subplots(1, 1)
    _ = ax.scatter(x, y)
    if isinstance(title, str):
        ax.set_title(title)
    if isinstance(x_label, str):
        ax.set_xlabel(x_label)
    if isinstance(y_label, str):
        ax.set_ylabel(y_label)
    plt.tight_layout()
    if show: plt.show()


def plot_y_discrete(y, colors, show=True, title=None):
    fig, ax = plt.subplots(1, 1)
    sns.countplot(x=y, palette=colors, ax=ax)
    if isinstance(title, str):
        ax.set_title(title)
    plt.tight_layout()
    if show: plt.show()


def save_sub(model, X_test, split, task, enc=None, team_id=40):
    # Run this to save a file with your predictions on the test set to be submitted

    y_hat = model.predict(X_test)
    if task == 'clf' and enc is not None:
        y_hat = enc.inverse_transform(y_hat)
    print(y_hat.shape)
    print(y_hat[:5])

    # Save the results with the format <TEAM_ID>__<SPLIT>_reg_pred.npy

    folder = '../predictions'
    np.save(os.path.join(folder, f'{team_id}__{split}__{task}_pred.npy'), y_hat)
