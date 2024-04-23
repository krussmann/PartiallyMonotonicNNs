import pickle
import os
import pandas as pd
import numpy as np
from MA.Experiments.Utils import makeDir

# Script for aggregating the univariate metrics and saving them as an excel workbook
# 'run_id' is the name of the result folder that gets created by experiment script 'univariate_experiments.py'

run_id = "N50_5T/"
fileroot = f"../../univariate/results/{run_id}pickles/"

# read pickle files (results)
fns = os.listdir(fileroot)
data_dict = {}
with_liu = False
for fn in fns:
    with open(fileroot + fn, 'rb') as f:
        data_dict[fn.split('.')[0]] = pickle.load(f)
    if "_nan.pkl" in fn:
        with_liu = True
N_tasks = len(data_dict["Y_Hat_test"])
N_methods = len(data_dict["Y_Hat_test"][0])
N_trials = len(data_dict["Y_Hat_test"][0][0])
METHODS = data_dict["methods"]
TASKS = data_dict["tasks"]

# if '..._nan' metrics are available (liu), they are evaluated
v_mse_train = "MSE_train" if not with_liu else "MSE_train_nan"
v_mse_test = "MSE_test" if not with_liu else "MSE_test_nan"
v_times = "times" if not with_liu else "times_nan"
v_train_histories = "train_histories" if not with_liu else "train_histories_nan"

makeDir(run_id + "metrics/big_dif/", True)

## Metric calculation
# Biggest difference between label and prediction averaged over trials
biggest_dif_avgs = np.zeros((N_tasks, N_methods))
biggest_dif_trial = np.zeros((N_tasks, N_methods))
biggest_dif_x = np.zeros((N_tasks, N_methods))
biggest_dif_dif = np.zeros((N_tasks, N_methods))
for task_id, task in enumerate(data_dict["Y_Hat_test"]):
    for method_id, method in enumerate(task):
        abs_sum = 0.0
        max_value = 0
        max_trial = 0
        max_x = 0
        for trial, preds in enumerate(method):
            max_dif = max(abs(data_dict["Y_test"][task_id][trial] - preds))
            argmax_x = np.argmax(abs(data_dict["Y_test"][task_id][trial] - preds))
            abs_sum += max_dif
            if max_dif > max_value:
                max_value = max_dif
                max_trial = trial
                max_x = argmax_x
        biggest_dif_avgs[task_id][method_id] = abs_sum/N_trials
        biggest_dif_trial[task_id][method_id] = max_trial
        biggest_dif_x[task_id][method_id] = data_dict["X_test"][task_id][max_trial][max_x]
        biggest_dif_dif[task_id][method_id] = max_value

# ndarray to df transform for simple write to excel later
df_big_dif = pd.DataFrame(biggest_dif_avgs, index=TASKS, columns=METHODS)
df_biggest_dif_trial = pd.DataFrame(biggest_dif_trial, index=TASKS, columns=METHODS)
df_biggest_dif_x = pd.DataFrame(biggest_dif_x, index=TASKS, columns=METHODS)
df_biggest_dif_dif = pd.DataFrame(biggest_dif_dif, index=TASKS, columns=METHODS)
df_big_dif.to_csv(run_id+'metrics/big_dif/biggest_dif_avg.csv', index=True, sep=";", decimal=",")
df_biggest_dif_trial.to_csv(run_id+'metrics/big_dif/biggest_dif_trial.csv', index=True, sep=";", decimal=",")
df_biggest_dif_x.to_csv(run_id+'metrics/big_dif/biggest_dif_x.csv', index=True, sep=";", decimal=",")
df_biggest_dif_dif.to_csv(run_id+'metrics/big_dif/biggest_dif_dif.csv', index=True, sep=";", decimal=",")


# Quantitative Metric
makeDir(run_id + "metrics/mse/", True)
# MSE Average
mse_avg_train = [[np.nanmean(mses) for mses in methods] for methods in data_dict[v_mse_train]]
mse_avg_test = [[np.nanmean(mses) for mses in methods] for methods in data_dict[v_mse_test]]
# MSE Median
mse_med_train = [[np.nanmedian(mses) for mses in methods] for methods in data_dict[v_mse_train]]
mse_med_test = [[np.nanmedian(mses) for mses in methods] for methods in data_dict[v_mse_test]]
# MSE Standard Deviation
mse_std_train = [[np.nanstd(mses) for mses in methods] for methods in data_dict[v_mse_train]]
mse_std_test = [[np.nanstd(mses) for mses in methods] for methods in data_dict[v_mse_test]]

df_mse_avg_train = pd.DataFrame(mse_avg_train, index=TASKS, columns=METHODS)
df_mse_avg_test = pd.DataFrame(mse_avg_test, index=TASKS, columns=METHODS)
df_mse_med_train = pd.DataFrame(mse_med_train, index=TASKS, columns=METHODS)
df_mse_med_test = pd.DataFrame(mse_med_test, index=TASKS, columns=METHODS)
df_mse_std_train = pd.DataFrame(mse_std_train, index=TASKS, columns=METHODS)
df_mse_std_test = pd.DataFrame(mse_std_test, index=TASKS, columns=METHODS)
df_mse_avg_train.to_csv(run_id+'metrics/mse/mse_avg_train.csv', index=True, sep=";", decimal=",")
df_mse_avg_test.to_csv(run_id+'metrics/mse/mse_avg_test.csv', index=True, sep=";", decimal=",")
df_mse_med_train.to_csv(run_id+'metrics/mse/mse_med_train.csv', index=True, sep=";", decimal=",")
df_mse_med_test.to_csv(run_id+'metrics/mse/mse_med_test.csv', index=True, sep=";", decimal=",")
df_mse_std_train.to_csv(run_id+'metrics/mse/mse_std_train.csv', index=True, sep=";", decimal=",")
df_mse_std_test.to_csv(run_id+'metrics/mse/mse_std_test.csv', index=True, sep=";", decimal=",")

# Model Params
params = [p for p in data_dict["model_params"].values()]
keys = [k for k in data_dict["model_params"].keys()]
df_params = pd.DataFrame([params], index= ["Model Parameter"], columns= keys)

# Epoch_count
epoch_counts = [[[np.nan if not isinstance(t, np.ndarray) else len(t) for t in method] for method in task] for task in data_dict[v_train_histories]]
epoch_median = np.nanmedian(epoch_counts, axis=(2))
epoch_mean = np.nanmean(epoch_counts, axis=(2))
epoch_std = np.nanstd(epoch_counts, axis=(2))

df_epoch_median = pd.DataFrame(epoch_median, index=TASKS, columns=METHODS)
df_epoch_mean = pd.DataFrame(epoch_mean, index=TASKS, columns=METHODS)
df_epoch_std = pd.DataFrame(epoch_std, index=TASKS, columns=METHODS)

# Times
makeDir(run_id + "metrics/times/", True)
init_times = [[[trial["model_init"] if isinstance(trial, dict) and trial["model_init"] is not None else np.nan for trial in method] for method in task] for task in data_dict[v_times]]
train_times = [[[trial["training"] if isinstance(trial, dict) else np.nan for trial in method] for method in task] for task in data_dict[v_times]]
train_times = [[[0 if trial is None else trial for trial in method] for method in task] for task in train_times] # correct sivaraman None values
infer_times = [[[trial["inference"] if isinstance(trial, dict) else np.nan for trial in method] for method in task] for task in data_dict[v_times]]

init_median = np.nanmedian(init_times, axis= (0,2))
train_median = np.nanmedian(train_times, axis= (0,2))
infer_median = np.nanmedian(infer_times, axis= (0,2))

init_mean = np.nanmean(init_times, axis= (0,2))
train_mean = np.nanmean(train_times, axis= (0,2))
infer_mean = np.nanmean(infer_times, axis= (0,2))

init_std = np.nanstd(init_times, axis= (0,2))
train_std = np.nanstd(train_times, axis= (0,2))
infer_std = np.nanstd(infer_times, axis= (0,2))

times_stack = np.vstack((init_median,
                        train_median,
                        infer_median,
                        init_mean,
                        train_mean,
                        infer_mean,
                        init_std,
                        train_std,
                        infer_std
                        ))

df_times = pd.DataFrame(times_stack, index=["init_median",
                        "train_median",
                        "infer_median",
                        "init_mean",
                        "train_mean",
                        "infer_mean",
                        "init_std",
                        "train_std",
                        "infer_std" ], columns=METHODS)
df_times.to_csv(run_id+'metrics/times/times.csv', index=True, sep=";", decimal=",")

# Export metrics as Excel Workbook
excel_writer = pd.ExcelWriter(run_id+f'metrics/metrics_{run_id[:-1]}.xlsx', engine='xlsxwriter')
df_big_dif.to_excel(excel_writer,sheet_name='biggest_dif_avg')
df_biggest_dif_trial.to_excel(excel_writer,sheet_name='biggest_dif_trial')
df_biggest_dif_x.to_excel(excel_writer,sheet_name='biggest_dif_x')
df_biggest_dif_dif.to_excel(excel_writer,sheet_name='biggest_dif_difference')

df_mse_avg_train.to_excel(excel_writer,sheet_name='mse_avg_train')
df_mse_avg_test.to_excel(excel_writer,sheet_name='mse_avg_test')
df_mse_med_train.to_excel(excel_writer,sheet_name='mse_med_train')
df_mse_med_test.to_excel(excel_writer,sheet_name='mse_med_test')
df_mse_std_train.to_excel(excel_writer,sheet_name='mse_std_train')
df_mse_std_test.to_excel(excel_writer,sheet_name='mse_std_test')
df_times.to_excel(excel_writer, sheet_name='times')
df_params.to_excel(excel_writer, sheet_name='m_params')
df_epoch_median.to_excel(excel_writer, sheet_name='epoch_median')
df_epoch_mean.to_excel(excel_writer, sheet_name='epoch_mean')
df_epoch_std.to_excel(excel_writer, sheet_name='epoch_std')
excel_writer.close()