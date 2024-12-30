import subprocess
import itertools
import time
import os
import sys
from typing import List, Dict
import pynvml  # For GPU detection
import psutil  # For CPU detection
from itertools import product




# Initialize NVML
def initialize_nvml():
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"Failed to initialize NVML: {e}")
        sys.exit(1)

# Shutdown NVML
def shutdown_nvml():
    try:
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        print(f"Failed to shutdown NVML: {e}")

# Get available GPUs based on utilization
def get_available_gpus() -> List[int]:
    initialize_nvml()
    device_count = pynvml.nvmlDeviceGetCount()
    available_gpus = []
    print("Checking GPU utilization...")
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        print(f"GPU {i}: Utilization {utilization}%")
        if utilization < 5:  # GPU is free if utilization is below 5%
            available_gpus.append(i)
    shutdown_nvml()
    return available_gpus

# Generate commands for hyperparameter combinations
def generate_commands(grid: Dict[str, List], log_dir: str) -> List[str]:
    combinations = list(itertools.product(
        grid['lr'], grid['batch_size'], grid['algorithm'], grid['seed'],
        grid['data_dir'], grid['dataset'], grid['model'], grid['epochs']
    ))

    lr_batch_pairs = [
        [(4096/128) * 0.1, (4096/128) * 0.01],  # Pair for batch_size 4096
        [(2048/128) * 0.1, (2048/128) * 0.01],  # Pair for batch_size 2048
        [0.1, 0.01]                             # Pair for batch_size 128
    ]
    batch_sizes = [[4096], [2048], [128]]

    valid_combinations = []
    for i, batch_size_group in enumerate(batch_sizes):
        for bs in batch_size_group:
            for lr in lr_batch_pairs[i]:
                valid_combinations.append((lr, bs))


    commands = []
    for combo in combinations:
        lr, batch_size, algorithm, seed, data_dir, dataset, model, epochs = combo
        if (lr, batch_size) not in valid_combinations:
            continue
        log_subdir = os.path.join(log_dir, f"{dataset}_lr{lr}_bs{batch_size}_alg{algorithm}_seed{seed}")
        os.makedirs(log_subdir, exist_ok=True)
        log_file = os.path.join(log_subdir, "train.log")
        
        cmd = (
            f"python main.py "
            f"--data_dir {data_dir} "
            f"--dataset {dataset} "
            f"--model {model} "
            f"--epochs {epochs} "
            f"--algorithm {algorithm} "
            f"--lr {lr} "
            f"--batch_size {batch_size} "
            f"--seed {seed} "
            f"--resume"
            f"> {log_file} 2>&1"
        )
        commands.append(cmd)
    print(len(commands))
    return commands


    

def multi_gpu_launcher(commands: List[str], gpus: List[int]):
    print('Starting multi_gpu_launcher...')
    n_gpus = len(gpus)
    if n_gpus == 0:
        print("No free GPUs available. Exiting.")
        sys.exit(1)

    n_gpus = len(gpus)

    procs_by_gpu = {gpu: None for gpu in gpus}
    job_queue = list(commands)  # Commands to execute
    n_cpus = psutil.cpu_count(logical=False) 
    cpus_per_gpu = n_cpus // len(gpus)

    try:
        while job_queue or any(proc is not None for proc in procs_by_gpu.values()):
            for gpu in gpus:
                proc = procs_by_gpu[gpu]
                if proc is None or proc.poll() is not None:
                    # if proc is not None:  # Cleanup finished process
                    #     # print(f"Job on GPU {gpu} has completed.")
                    #     procs_by_gpu[gpu] = None
                    if job_queue:
                        cmd = job_queue.pop(0)
                        full_cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"
                        print(f"Launching on GPU {gpu}: {cmd}")
                        try:
                            procs_by_gpu[gpu] = subprocess.Popen(full_cmd, shell=True)
                            # breakpoint()
                            # procs_by_gpu[gpu] = proc
 
                        except Exception as e:
                            print(f"GPU {gpu}: Failed to start command. Error: {e}")
            time.sleep(1)  # Check GPU availability every 2 seconds

    except KeyboardInterrupt:
        print("Hyperparameter search interrupted by user. Terminating running jobs...")
        for proc in procs_by_gpu.values():
            if proc is not None:
                proc.terminate()
        shutdown_nvml()
        sys.exit(0)

    # Wait for all processes to complete
    for gpu, proc in procs_by_gpu.items():
        if proc is not None:
            proc.wait()
            print(f"Job on GPU {gpu} has finished.")

    print("All hyperparameter search jobs have been completed.")



# Main hyperparameter search function
def hyperparameter_search(grid: Dict[str, List], log_dir: str = './hyperparam_search_logs'):
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate all commands
    commands = generate_commands(grid, log_dir)
    print(commands)
    print(f"Generated {len(commands)} commands.")

    # Detect available GPUs
    available_gpus = get_available_gpus()
    print(f"Available GPUs for launching jobs: {available_gpus}")

    if not available_gpus:
        print("No free GPUs available at the moment. Exiting.")
        sys.exit(1)

    # Launch the commands across available GPUs
    multi_gpu_launcher(commands, available_gpus)
# Main execution block
if __name__ == "__main__":
    # Define hyperparameter grid
    
    lr_batch_pairs = [
        [(4096/128) * 0.1, (4096/128) * 0.01],  # Pair for batch_size 4096
        [(2048/128) * 0.1, (2048/128) * 0.01],  # Pair for batch_size 2048
        [0.1, 0.01]                             # Pair for batch_size 128
    ]
    batch_sizes = [[4096], [2048], [128]]

    valid_combinations = []
    for i, batch_size_group in enumerate(batch_sizes):
        for bs in batch_size_group:
            for lr in lr_batch_pairs[i]:
                valid_combinations.append((lr, bs))


    hyperparameter_grid = {
        'lr': [lr for lr_pair in lr_batch_pairs for lr in lr_pair],
        'batch_size': [bs for bs_pair in batch_sizes for bs in bs_pair],
        'algorithm': ['sgd'],
        'seed': list(range(1, 2)),  # Seeds 1 through 10
        'data_dir': ['../data'],
        'dataset': ['cifar100','cifar10'],
        'model': ['resnet18'],
        'epochs': [200]
    }

    # Delete the combination if lr and batch_size are not paired
    # for example, if lr is 0.1 and batch_size is 128, then the combination is valid
    # if lr is 0.1 and batch_size is 2048, then the combination is invalid
    # your code here

    # Generate all combinations
    all_combinations = list(product(
        hyperparameter_grid['lr'],
        hyperparameter_grid['batch_size'],
        hyperparameter_grid['algorithm'],
        hyperparameter_grid['seed'],
        hyperparameter_grid['data_dir'],
        hyperparameter_grid['dataset'],
        hyperparameter_grid['model'],
        hyperparameter_grid['epochs']
    ))

    hyperparameter_grid_list = []
    for lr, bs in valid_combinations:
        grid = hyperparameter_grid.copy()
        grid['lr'] = [lr]
        grid['batch_size'] = [bs]
        hyperparameter_grid_list.append(grid)

    filtered_combinations = [
        combo for combo in all_combinations
        if (combo[0], combo[1]) in valid_combinations
    ]


    # Directory to save logs
    log_directory = './hyperparam_search_logs'

    # Run hyperparameter search
    hyperparameter_search(hyperparameter_grid, log_dir=log_directory)
