import pickle
import os
import pandas as pd
import numpy as np
from MA.Experiments.Utils import makeDir
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import math

# Script for creating plots for the function approximation experiment

# Configure Matplotlib Styling
# For all Params see : https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
params = {
    'legend.fontsize': 8,
    'axes.labelsize': 14,
    'axes.titlesize': 20, # figure title
    'xtick.labelsize': 14,
    'ytick.labelsize':14,
    'font.family': 'serif',
    # 'legend.shadow': True,
    # 'xtick.labelbottom': True,
    'figure.subplot.bottom': .125,
    'figure.subplot.top':.8,
    'axes.titlepad':15,
}
pylab.rcParams.update(params)

# rename methods for report
report_name_list = ['Unconstrained MLP', 'Min-Max', 'Monotonic Dense', 'Lipschitz Dense', 'Certified Monotonic', 'Monotonic Envelope']

# rename tasks for report
report_task_list = ["sigmoid10", "square", "square root", "non-smooth", "non-monotonic 1", "non-monotonic 2", "non-monotonic 3"]

# 'run_id' is the name of the result folder that gets created by experiment script 'univariate_experiments.py'
run_id = "N50_5T/"
fileroot = f"../../univariate/results/{run_id}pickles/"

# read result pickle files
fns = os.listdir(fileroot)
with_liu = False
data_dict = {}
for fn in fns:
    print(fn)
    with open(fileroot + fn, 'rb') as f:
        data_dict[fn.split('.')[0]] = pickle.load(f)
    if "_nan.pkl" in fn:
        with_liu = True

# Hard-Coding Run Mapping where monotonicity is not reached (Liu)
non_mon_liu_dict_train = {i: np.ones_like(data_dict["Y_Hat_train"], dtype=bool) for i in ["N_5T", "Noise_5T", "N50_5T"]}
non_mon_liu_dict_test = {i: np.ones_like(data_dict["Y_Hat_test"], dtype=bool) for i in ["N_5T", "Noise_5T", "N50_5T"]}
for i in [non_mon_liu_dict_train, non_mon_liu_dict_test]:
    i["N_5T"][6,5,3] = False
    i["Noise_5T"][6,5,3] = False
    i["Noise_5T"][0,5,3] = False
    i["N50_5T"][2,5,0] = False
    i["N50_5T"][4,5,2] = False

for fn in ["Y_Hat_test", "Y_Hat_train"]:
    data_dict[fn+"_nan"] = data_dict[fn]
    if "train" in fn:
        data_dict[fn+"_nan"][~non_mon_liu_dict_train[run_id[:-1]]] = np.nan
    elif "test" in fn:
        data_dict[fn + "_nan"][~non_mon_liu_dict_test[run_id[:-1]]] = np.nan

name_mapping = {}
for old, new in zip(data_dict["methods"], report_name_list):
    name_mapping[old] = new
task_mapping = {}
for old, new in zip(data_dict["tasks"], report_task_list):
    task_mapping[old] = new

N_tasks = len(data_dict["Y_Hat_test"])
N_methods = len(data_dict["Y_Hat_test"][0])
N_trials = len(data_dict["Y_Hat_test"][0][0])

v_y_hat_test = "Y_Hat_test" if not with_liu else "Y_Hat_test_nan"
v_y_hat_train = "Y_Hat_train" if not with_liu else "Y_Hat_train_nan"

# calculate best, worst and median run-index for each method and epoch for plotting those
best_idx_map = np.ones((N_tasks, N_methods)) * 10
worst_idx_map = np.ones((N_tasks, N_methods)) * 10
median_idx_map = np.ones((N_tasks, N_methods)) * 10

for task_id in range(N_tasks):
    for method_id in range(N_methods):
        best_idx_map[task_id][method_id] = np.argsort(data_dict["MSE_test_nan"][task_id, method_id])[0]
        worst_idx_map[task_id][method_id] = np.argsort(data_dict["MSE_test_nan"][task_id, method_id])[-1]
        median_idx_map[task_id][method_id] = np.argsort(data_dict["MSE_test_nan"][task_id, method_id])[N_trials//2]

# Plots displaying the best out of 5 trials per task and per method
for task_id, task in enumerate(data_dict["tasks"]):
    makeDir(run_id + task + '/pngs/', True)

    plt.figure()
    plt.plot(data_dict["X_test"][task_id, 0], data_dict["Y_test"][task_id, 0], '--', label="Orig Function")
    for method_id, method in enumerate(data_dict["methods"]):
        trial_idx = int(best_idx_map[task_id,method_id])
        if math.isnan(data_dict[v_y_hat_test][task_id, method_id, trial_idx][0]):
            continue
        plt.plot(data_dict["X_test"][task_id, 0], data_dict[v_y_hat_test][task_id, method_id, trial_idx], label=name_mapping[method])
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{task_mapping[task]} Best Runs (Test)')
    plt.savefig(run_id + task + f"/pngs/best_runs_test.png")
    plt.close('all')

# Plots displaying the worst out of 5 trials per task and per method
for task_id, task in enumerate(data_dict["tasks"]):
    makeDir(run_id + task + '/pngs/', True)

    plt.figure()
    plt.plot(data_dict["X_test"][task_id, 0], data_dict["Y_test"][task_id, 0], '--', label="Orig Function")
    for method_id, method in enumerate(data_dict["methods"]):
        trial_idx = int(worst_idx_map[task_id, method_id])
        if math.isnan(data_dict[v_y_hat_test][task_id, method_id, trial_idx][0]):
            continue
        plt.plot(data_dict["X_test"][task_id, 0], data_dict[v_y_hat_test][task_id, method_id, trial_idx], label=name_mapping[method])
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{task_mapping[task]} Worst Runs (Test)')
    plt.savefig(run_id + task + f"/pngs/worst_runs_test.png")
    plt.close('all')

# Plots displaying the median out of 5 trials per task and per method
for task_id, task in enumerate(data_dict["tasks"]):
    makeDir(run_id + task + '/pngs/', True)

    plt.figure()
    plt.plot(data_dict["X_test"][task_id, 0], data_dict["Y_test"][task_id, 0], '--', label="Orig Function")
    for method_id, method in enumerate(data_dict["methods"]):
        trial_idx = int(median_idx_map[task_id, method_id])
        if math.isnan(data_dict[v_y_hat_test][task_id, method_id, trial_idx][0]):
            continue
        plt.plot(data_dict["X_test"][task_id, 0], data_dict[v_y_hat_test][task_id, method_id, trial_idx], label=name_mapping[method])
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{task_mapping[task]} Median Runs (Test)')
    plt.savefig(run_id + task + f"/pngs/median_runs_test.png")
    plt.close('all')

# create plots for every test task and save them in '<trail>/pngs/' (keeps non monotonic fits from liu)

for t in range(N_trials):
    makeDir(run_id + str(t) +'/pngs/', True)
    for task_id, task in enumerate(data_dict["tasks"]):
        plt.figure()
        plt.plot(data_dict["X_test"][task_id, t], data_dict["Y_test"][task_id, t], '--', label="Orig Function")
        for method_id, method in enumerate(data_dict["methods"]):
            plt.plot(data_dict["X_test"][task_id, t], data_dict["Y_Hat_test"][task_id, method_id, t], label=name_mapping[method])
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{task_mapping[task]} Function Approximation (Test)')
        plt.savefig(run_id + str(t) + f"/pngs/{task}_test.png")
        plt.show()

    plt.close('all')

# create plots for every train task and save them in '<trial>/pngs/' (keeps non monotonic fits from liu)
for t in range(N_trials):
    for task_id, task in enumerate(data_dict["tasks"]):
        plt.figure()
        plt.scatter(data_dict["X_train"][task_id, t], data_dict["Y_train"][task_id, t], s=5, label="Train data")
        for method_id, method in enumerate(data_dict["methods"]):
            plt.plot(data_dict["X_train"][task_id, t], data_dict["Y_Hat_train"][task_id, method_id, t], label=name_mapping[method])
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{task_mapping[task]} Function Approximation (Train)')
        plt.savefig(run_id + str(t) + f"/pngs/{task}_train.png")

    plt.close('all')


# Plots displaying all trials for a given method and a given task and save them in '<method>/pngs' (keeps non monotonic fits from liu)
for method_id, method in enumerate(data_dict["methods"]):
    makeDir(run_id + method + '/pngs/', True)
    for task_id, task in enumerate(data_dict["tasks"]):
        plt.figure()
        plt.plot(data_dict["X_test"][task_id, 0], data_dict["Y_test"][task_id, 0], '--', label="Orig Function")
        for t in range(N_trials):
            plt.plot(data_dict["X_test"][task_id, t], data_dict["Y_Hat_test"][task_id, method_id, t], label=t)
        # plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{task_mapping[task]} Function Approximation (Test)')
        plt.savefig(run_id + method + f"/pngs/{task}_test.png")

    plt.close('all')

# # Plots displaying mse optimization over epochs
# for method_id, method in enumerate(data_dict["methods"]):
#     makeDir(run_id + method + '/pngs/', True)
#     for task_id, task in enumerate(data_dict["tasks"]):
#         for t in range(N_trials):
#             plt.figure()
#             history = data_dict["train_histories"][task_id, method_id, t]
#             if not isinstance(history, np.ndarray):
#                 continue
#             plt.plot(history)
#             plt.title(f'Mean Squared Error over Training Epochs \n{method}_{task}_{t}')
#             plt.xlabel('Epoch')
#             plt.ylabel('Mean Squared Error')
#             plt.savefig(run_id + method + f"/pngs/train_history_{method}_{task}_{t}.png")
#
#     plt.close('all')
