import os
import numpy as np
from MA.Experiments.Utils import makeDir
import pandas as pd

# Script for aggregating the multivariate metrics and saving them as an excel workbook
# 'run_id' is the name of the result folder that gets created by experiment script 'multivariate_experiments.py'

run_id = '20240423949'
result_dir = '../../multivariate/results/'+run_id+'/'
npzfile = np.load(result_dir+os.listdir(result_dir)[0], allow_pickle=True)
print("found files:", npzfile.files)

makeDir(run_id + "/metrics/", True)
# reading methods comprised in results
methods = [k for k in npzfile["model_params"].item().keys()]

# Quantitative MSE mean, median, std
# MSE Average
mse_avg_train = np.nanmean(npzfile["mse_train"], axis=1).reshape(1, len(methods))
mse_avg_test = np.nanmean(npzfile["mse_test"], axis=1).reshape(1, len(methods))
# MSE Median
mse_med_train = np.nanmedian(npzfile["mse_train"], axis=1).reshape(1, len(methods))
mse_med_test = np.nanmedian(npzfile["mse_test"], axis=1).reshape(1, len(methods))
# MSE Standard Deviation
mse_std_train = np.nanstd(npzfile["mse_train"], axis=1).reshape(1, len(methods))
mse_std_test = np.nanstd(npzfile["mse_test"], axis=1).reshape(1, len(methods))

# ndarray to df transform for simple write to excel
df_mse_avg_train = pd.DataFrame(mse_avg_train, index=["mse_mean_train"], columns=methods)
df_mse_avg_test = pd.DataFrame(mse_avg_test, index=["mse_mean_test"], columns=methods)
df_mse_med_train = pd.DataFrame(mse_med_train, index=["mse_med_train"], columns=methods)
df_mse_med_test = pd.DataFrame(mse_med_test, index=["mse_med_test"], columns=methods)
df_mse_std_train = pd.DataFrame(mse_std_train, index=["mse_std_train"], columns=methods)
df_mse_std_test = pd.DataFrame(mse_std_test, index=["mse_std_test"], columns=methods)

# Model Params
params = [p for p in npzfile["model_params"].item().values()]
df_params = pd.DataFrame([params], index= ["Model Parameter"], columns= methods)

# Epoch_count
epoch_counts = [[np.nan if not isinstance(split, np.ndarray) else len(split) for split in method] for method in npzfile["train_histories"]]
# epoch_counts = [[[len(t) for t in method] for method in task] for task in npzfile[v_train_histories]]
epoch_mean = np.nanmean(epoch_counts, axis=(1)).reshape(1, len(methods))
epoch_median = np.nanmedian(epoch_counts, axis=(1)).reshape(1, len(methods))
epoch_std = np.nanstd(epoch_counts, axis=(1)).reshape(1, len(methods))

df_epoch_mean = pd.DataFrame(epoch_mean, index=["Epoch Mean"], columns=methods)
df_epoch_median = pd.DataFrame(epoch_median, index=["Epoch Median"], columns=methods)
df_epoch_std = pd.DataFrame(epoch_std, index=["Epoch Std"], columns=methods)

# Train-/Inference-Times
init_times = [[split["model_init"] if isinstance(split, dict) and split["model_init"] is not None else np.nan for split in method] for method in npzfile["times"]]
train_times = [[split["training"] if isinstance(split, dict) else np.nan for split in method] for method in npzfile["times"]]
train_times = [[0 if split is None else split for split in method] for method in train_times] # correct sivaraman None values
infer_times = [[split["inference"] if isinstance(split, dict) else np.nan for split in method] for method in npzfile["times"]]

init_median = np.nanmedian(init_times, axis= (1))
train_median = np.nanmedian(train_times, axis= (1))
infer_median = np.nanmedian(infer_times, axis= (1))

init_mean = np.nanmean(init_times, axis= (1))
train_mean = np.nanmean(train_times, axis= (1))
infer_mean = np.nanmean(infer_times, axis= (1))

init_std = np.nanstd(init_times, axis= (1))
train_std = np.nanstd(train_times, axis= (1))
infer_std = np.nanstd(infer_times, axis= (1))

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
                        "infer_std" ], columns=methods)

# Export metrics as Excel Workbook
excel_writer = pd.ExcelWriter(run_id+f'/metrics/metrics_{run_id[:-1]}.xlsx', engine='xlsxwriter')

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