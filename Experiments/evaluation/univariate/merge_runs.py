import os
import pickle
import numpy as np

# Script for merging the result pickle files for two independent runs

def merge_results(runid1, runid2, liu=None, siva=None):
    fileroot1 = f"../../univariate/results/{runid1}/pickles/"
    fileroot2 = f"../../univariate/results/{runid2}/pickles/"
    fileroot_merge = f"../../univariate/results/merge_{runid1}_{runid2}/pickles/"
    os.makedirs(fileroot_merge)

    fns1 = os.listdir(fileroot1)
    fns2 = os.listdir(fileroot2)
    data_dict1, data_dict2 = {}, {}

    for fn in fns1:
        # read in fp1
        with open(fileroot1 + fn, 'rb') as f:
            data_dict1[fn.split('.')[0]] = pickle.load(f)
    for fn in fns2:
        # read in fp2
        with open(fileroot2 + fn, 'rb') as f:
            data_dict2[fn.split('.')[0]] = pickle.load(f)

    if siva is not None:
        siva_dict = [data_dict1, data_dict2][siva]
        for p in ["MSE_train", "MSE_test", "times", "train_histories",
                 "monotonic_liu", "Y_Hat_train", "Y_Hat_test"]:
            siva_dict[p] = np.delete(siva_dict[p],0,1) # delete metrics from unconstrained mlp
            siva_dict[p] = np.repeat(siva_dict[p], 5, axis=2) # copy metrics from 1 trail to other 4 trials

        siva_dict["methods"].remove("Unconstrained_tf")
        siva_dict["model_params"].pop("Unconstrained_tf")


    for p in data_dict1.keys():
        if p in ["MSE_train", "MSE_train_nan", "MSE_test", "MSE_test_nan", "times", "times_nan", "train_histories", "train_histories_nan", "monotonic_liu", "Y_Hat_train", "Y_Hat_test"]:
            p2 = p if p in data_dict2.keys() else "_".join(p.split("_")[:-1]) # handles special case when '..._nan'-file does not exist
            concat = np.concatenate([data_dict1[p], data_dict2[p2]], axis=1)
            with open(fileroot_merge + f"{p}.pkl", 'wb') as f:
                pickle.dump(concat, f)

        if p in ["X_train", "Y_train", "X_test", "Y_test", "tasks"]:
            with open(fileroot_merge + f"{p}.pkl", 'wb') as f:
                pickle.dump(data_dict1[p], f)

    if liu is not None:
        liu_dict = [data_dict1, data_dict2][liu]
        for p in ["MSE_train", "MSE_test", "times", "train_histories"]:
            # Set elements to np.nan based on the mask
            temp_nan = liu_dict[p]
            # 'monotonic_liu' is a binary mapping where monotonicity was certified
            temp_nan[~liu_dict["monotonic_liu"]] = np.nan # metric set to nan where monotonicity could not be guaranteed
            concat = np.concatenate([data_dict1[p], temp_nan], axis=1)
            with open(fileroot_merge + f"{p}_nan.pkl", 'wb') as f:
                pickle.dump(concat, f)

    new_methods = data_dict1["methods"] + data_dict2["methods"]
    with open(fileroot_merge + "methods.pkl", 'wb') as f:
        pickle.dump(new_methods, f)

    new_model_params = data_dict1["model_params"] | data_dict2["model_params"]
    with open(fileroot_merge + "model_params.pkl", 'wb') as f:
        pickle.dump(new_model_params, f)

if __name__=='__main__':
    # define main and second as the run_ids that got created by experiment execution
    main = "merge_usrin_liu"
    second = "siva"
    merge_results(main, second, siva=1)



