#!/usr/bin/env python3
"""
Refactored script from the original Jupyter Notebook for:
1. Parsing filenames of training logs.
2. Processing directories to compute average validation accuracies.
3. Filtering and plotting results for various conditions (lr=0.1, rf10 vs rf20, rescale vs non-rescale).
4. Checking for missing epochs and reindexing.

Author: (Your Name)
Created: (Date)
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# -------------------------------------------------------------------------
# 1. Global Directories / Constants
# -------------------------------------------------------------------------
LOG_DIR = '../logs/resnet18'
CIFAR10_DIR = os.path.join(LOG_DIR, 'cifar10')
IMAGENET_DIR = os.path.join(LOG_DIR, 'imagenet')
CIFAR100_DIR = os.path.join('../../pytorch-cifar100/logs/resnet18/cifar100')

# If you want a specific CIFAR-100 path:
# CIFAR100_DIR = os.path.join('../../pytorch-cifar100/logs/resnet18/cifar100')

def extract_numeric_from_tensor_string(s):
    """
    Extracts the numeric value from a string formatted like "tensor(1.1200, device='cuda:0')"
    Returns a float if successful, or NaN if extraction fails.
    """
    # Define a regex pattern to capture the number inside 'tensor(...)'
    pattern = r'tensor\(([\d.]+),'
    match = re.search(pattern, s)
    if match:
        return float(match.group(1))
    else:
        # If the pattern does not match, return NaN
        return float('nan')



# -------------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------------
def parse_filename(filename):
    """
    Parses the filename to extract algorithm, hyperparameters, and seed.
    E.g. "sgd_lr0.1_bs128_s1.csv" => 
         algorithm="sgd", hyperparams={"lr":"0.1","bs":"128"}, seed=1, rescale=False

    Args:
        filename (str): The name of the file (e.g. "sgd_lr0.1_bs128_s1.csv").

    Returns:
        tuple: (algorithm (str), hyperparams (dict), seed (int), rescale (bool))
    """
    # Remove the file extension
    name = filename.replace('.csv', '')
    parts = name.split('_')

    # The first part is the algorithm
    algorithm = parts[0]
    
    hyperparams = {}
    seed = None
    rescale = False

    # Regex to match key-value pairs (e.g., "lr0.1", "bs128", "rf10")
    pattern = re.compile(r'([a-zA-Z]+)([\d.]+)')

    for part in parts[1:]:
        if part == 'rescale':
            rescale = True
        elif part.startswith('s') and part[1:].isdigit():
            seed = int(part[1:])
        else:
            match = pattern.match(part)
            if match:
                key, value = match.groups()
                hyperparams[key] = value
    
    return algorithm, hyperparams, seed, rescale



def process_directory(directory_path):
    average_metrics_dict = defaultdict(dict)  # Initialize the nested dictionary

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):  # Adjust the file extension as needed
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(file_path)
                
                # Check if 'val_acc' column exists
                if 'val_acc' not in df.columns:
                    print(f"File '{filename}' does not contain 'val_acc' column. Skipping.")
                    continue  # Skip to the next file

                val_acc = df['val_acc']
                algorithm = filename.split('_')[0]  # Extract the algorithm from the filename
                hyperparam_str = filename.split('_')[1]  # Extract the hyperparameters from the filename
                hyperparam_str = hyperparam_str.replace('.csv', '')

                # Check if 'val_acc' is numeric
                if pd.api.types.is_numeric_dtype(val_acc):
                    val_acc_numeric = val_acc
                else:
                    # Attempt to parse tensor-formatted strings
                    # print(f"Non-numeric 'val_acc' column found in file: {filename}. Attempting to parse.")
                    # Apply the extraction function to each entry
                    val_acc_numeric = val_acc.apply(extract_numeric_from_tensor_string)

                    # Check if extraction was successful for all entries
                    num_nans = val_acc_numeric.isna().sum()
                    if num_nans > 0:
                        print(f"Warning: {num_nans} 'val_acc' entries in file '{filename}' could not be parsed and were set to NaN.")

                # Compute the mean of 'val_acc'
                val_acc_mean = val_acc_numeric.mean()

                # Similarly handle 'val_loss' if necessary
                if 'val_loss' in df.columns:
                    val_loss = df['val_loss']
                    if not pd.api.types.is_numeric_dtype(val_loss):
                        print(f"Non-numeric 'val_loss' column found in file: {filename}. Setting 'val_loss_mean' to None.")
                        val_loss_mean = None
                    else:
                        val_loss_numeric = pd.to_numeric(val_loss, errors='coerce')
                        val_loss_mean = val_loss_numeric.mean()
                else:
                    val_loss_mean = None  # 'val_loss' column not present

                # Store the computed means in the nested dictionary
                average_metrics_dict[algorithm][hyperparam_str] = {
                    "val_acc": val_acc_mean,
                    "val_loss": val_loss_mean
                }

            except Exception as e:
                print(f"Error processing file '{filename}': {e}. Skipping this file.")
                continue  # Skip to the next file in case of any other errors

    return average_metrics_dict

def plot_metric(
    data, 
    metric_key="val_acc", 
    title="Validation Accuracy", 
    xlabel="Epoch", 
    ylabel="Metric Value"
):
    """
    Plots a given metric (e.g., 'val_acc' or 'val_loss') from the nested dictionary:
      data[algorithm][hyperparam_str] = {"val_acc": Series, "val_loss": Series, ...}
    """
    plt.figure(figsize=(8, 6))
    
    for algorithm, hyperparams_dict in data.items():
        for hyperparam_str, metrics_dict in hyperparams_dict.items():
            # Ensure this run actually has the metric
            series = metrics_dict.get(metric_key)
            if isinstance(series, pd.Series):
                plt.plot(series.index, series.values, label=f"{algorithm} {hyperparam_str}")
            else:
                print(f"Metric '{metric_key}' for '{algorithm} {hyperparam_str}' is not a pandas Series. Skipping.")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def check_missing_epochs(directory, expected_epochs):
    """
    Checks for CSV files missing certain epoch values, printing their names.

    Args:
        directory (str): Path to the directory containing CSV files.
        expected_epochs (list or range): The set of epoch values you expect.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and 'divebatch' in filename:
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                if 'epoch' not in df.columns:
                    print(f"[Warning] '{filename}' missing 'epoch' column.")
                    continue

                file_epochs = set(df['epoch'].unique())
                missing_epochs = set(expected_epochs) - file_epochs
                if missing_epochs:
                    print(f"[Info] File '{filename}' missing epochs: {sorted(missing_epochs)}")
            except Exception as e:
                print(f"[Error] reading '{filename}': {e}")


def reindex_epochs_starting_from_zero(directory, start_epoch=0):
    """
    For CSV files whose first epoch equals `start_epoch`, reindex them to start from 1.

    Args:
        directory (str): Path to the directory containing CSV files.
        start_epoch (int): The epoch value to detect (default=0).
    """
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                if 'epoch' not in df.columns:
                    print(f"[Warning] '{filename}' has no 'epoch' column.")
                    continue

                first_epoch = df['epoch'].iloc[0]
                if first_epoch == start_epoch:
                    print(f"[Info] Reindexing epochs for file: '{filename}' (from {start_epoch} to 1..)")
                    df['epoch'] = df['epoch'] + 1
                    df.to_csv(file_path, index=False)
            except Exception as e:
                print(f"[Error] processing '{filename}': {e}")


# -------------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------------
def main():
    """
    Main function to:
      1. Process CIFAR-10 logs to collect both val_acc and val_loss.
      2. Split into rescale vs. non-rescale data.
      3. Filter specific runs (e.g., lr0.1).
      4. Plot both val_acc and val_loss.
      5. Optionally split by 'rf10'/'rf20', check epochs, etc.
    """
    # 1) Process CIFAR-10 logs, now collecting both val_acc and val_loss
    average_metrics_cifar10 = process_directory(CIFAR100_DIR)
    print(average_metrics_cifar10['sgd']['lr0.1']['val_acc'])
    # # Result shape:
    # # average_metrics_cifar10[algorithm][hyperparam_str] = {
    # #     "val_acc": pd.Series or None,
    # #     "val_loss": pd.Series or None
    # # }

    # # 2) Separate into rescale vs. non-rescale
    # rescale_data = defaultdict(dict)
    # non_rescale_data = defaultdict(dict)

    # for algorithm, hp_map in average_metrics_cifar10.items():
    #     for hyperparam_str, metrics_dict in hp_map.items():
    #         # If 'sgd' in algorithm => put in both rescale and non-rescale
    #         if 'sgd' in algorithm:
    #             rescale_data[algorithm][hyperparam_str] = metrics_dict
    #             non_rescale_data[algorithm][hyperparam_str] = metrics_dict
    #         # Else, check if hyperparam_str has 'rescale'
    #         elif "rescale" in hyperparam_str:
    #             rescale_data[algorithm][hyperparam_str] = metrics_dict
    #         else:
    #             non_rescale_data[algorithm][hyperparam_str] = metrics_dict

    # # 3) Filter to runs with "lr0.1" (excluding "lr0.16") in non-rescale_data
    # filtered_lr01 = {}
    # for algo, hp_map in non_rescale_data.items():
    #     for hyperparam_str, metrics_dict in hp_map.items():
    #         # Only keep if "lr0.1" in hyperparam_str and "lr0.16" not in hyperparam_str
    #         if "lr0.1" in hyperparam_str and "lr0.16" not in hyperparam_str:
    #             # Keep entire metrics_dict => has both "val_acc" & "val_loss"
    #             filtered_lr01[(algo, hyperparam_str)] = metrics_dict

    # # 4) Plot BOTH val_acc and val_loss from the entire non_rescale_data or rescale_data
    # #    For example, let's show the entire non_rescale_data for demonstration.
    
    # # Plot ACC
    # plot_metric(
    #     data=non_rescale_data,
    #     metric_key="val_acc",
    #     title="Non-Rescale: Validation Accuracy (CIFAR-10)",
    #     ylabel="Validation Accuracy"
    # )
    # # Plot LOSS
    # plot_metric(
    #     data=non_rescale_data,
    #     metric_key="val_loss",
    #     title="Non-Rescale: Validation Loss (CIFAR-10)",
    #     ylabel="Validation Loss"
    # )
    
    # # 5) If desired, also plot from the 'rescale_data'
    # plot_metric(
    #     data=rescale_data,
    #     metric_key="val_acc",
    #     title="Rescaled: Validation Accuracy (CIFAR-10)",
    #     ylabel="Validation Accuracy"
    # )
    # plot_metric(
    #     data=rescale_data,
    #     metric_key="val_loss",
    #     title="Rescaled: Validation Loss (CIFAR-10)",
    #     ylabel="Validation Loss"
    # )

    # # 6) Optionally, split the 'filtered_lr01' subset into 'rf10' / 'rf20' plots
    # rf10_data = {}
    # rf20_data = {}
    # for (algo, hparam_str), metrics_dict in filtered_lr01.items():
    #     if "rf10" in hparam_str:
    #         rf10_data[(algo, hparam_str)] = metrics_dict
    #     elif "rf20" in hparam_str:
    #         rf20_data[(algo, hparam_str)] = metrics_dict
    #     else:
    #         # No "rf"? => goes to both
    #         rf10_data[(algo, hparam_str)] = metrics_dict
    #         rf20_data[(algo, hparam_str)] = metrics_dict
    
    # # Example: plot ACC for rf10
    # plot_metric(
    #     data={"rf10": rf10_data},  # pass a dict of dict(s)
    #     metric_key="val_acc",
    #     title="RF=10 (lr0.1) - Validation Accuracy",
    #     ylabel="Validation Accuracy"
    # )
    # # Example: plot LOSS for rf10
    # plot_metric(
    #     data={"rf10": rf10_data},
    #     metric_key="val_loss",
    #     title="RF=10 (lr0.1) - Validation Loss",
    #     ylabel="Validation Loss"
    # )

    # # Similarly for rf20
    # plot_metric(
    #     data={"rf20": rf20_data},
    #     metric_key="val_acc",
    #     title="RF=20 (lr0.1) - Validation Accuracy",
    #     ylabel="Validation Accuracy"
    # )
    # plot_metric(
    #     data={"rf20": rf20_data},
    #     metric_key="val_loss",
    #     title="RF=20 (lr0.1) - Validation Loss",
    #     ylabel="Validation Loss"
    # )

    # # 7) Example: Check for missing epochs, reindex, etc.
    # expected_epochs = range(1, 201)
    # check_missing_epochs(CIFAR10_DIR, expected_epochs)
    # # reindex_epochs_starting_from_zero(CIFAR10_DIR, start_epoch=0)


# -------------------------------------------------------------------------
# 4. Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
