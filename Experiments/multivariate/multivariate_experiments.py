import copy
import random
import sys
import importlib
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from sklearn import preprocessing

# Importing custom models and utilities
from MA.Models.Unconstrained_NN.model import Network_tf
from MA.Experiments.Utils import fit_torch_val, PrintReportCallback, EarlyStoppingCallback, makeDir
from MA.Models.MaxMinNetworks_Sill_1997.model import MonotonicNNMultiv
from MA.Models.Expressive_MNN_Nolte_2023.model import ExpressiveNetwork
from MA.Models.CMNN_Runje_2023.model import CMNN
from MA.Models.CertifiedMonotonicNetwork_Liu_et_al_2020.utils.networks import MLP_relu
from MA.Models.CertifiedMonotonicNetwork_Liu_et_al_2020.model_utils import fit_certified_MN_val
from MA.Models.CertifiedMonotonicNetwork_Liu_et_al_2020.utils.certify import certify_neural_network
from MA.Models.Counter_Examples_Sivaraman_2020.src.Utils import makeDir,write_to_csv,copyfiles,copyFile, readConfigurations
from MA.Models.Counter_Examples_Sivaraman_2020.src.VerifiedPredictions import getEnvelopePredictions

def fit_model(model, method, x_train, y_train, x_train_torch, y_train_torch, x_test, y_test, val_split, epochs, patience, lr):
    """
    Fit the specified model using the given method.

    Parameters:
    model (object): The model object to be trained.
    method (str): The method used for training the model.
    x_train (np.ndarray): Training input data.
    y_train (np.ndarray): Training target data.
    x_train_torch (torch.Tensor): Training input data in PyTorch tensor format.
    y_train_torch (torch.Tensor): Training target data in PyTorch tensor format.
    val_split (float): Ratio of validation split.
    epochs (int): Number of epochs for training.
    patience (int): Number of unimproved epochs before early stopping is triggered.
    lr (float): Learning rate for optimization.

    Returns:
    tuple: Training history and trained model object.
    """
    is_monotonic = True
    if method in ['Unconstrained_tf', 'Runje_2023']:
        # TensorFlow/Keras-based training
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        criterion = tf.keras.losses.MeanSquaredError()
        model.compile(loss=criterion, optimizer=optimizer)
        history = model.fit(x_train.astype(np.float32), y_train.astype(np.float32), validation_split=val_split, epochs=epochs,
                                        batch_size=x_train.shape[0], verbose=0,
                                        callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
                                                   PrintReportCallback()], shuffle=True)
        history = history.history['loss']

    elif method in ['Sill_1997', 'Nolte_2023']:
        # PyTorch-based training
        model_best = copy.deepcopy(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        if method == 'Nolte_2023':
            y_train_torch = y_train_torch.reshape(-1,1)
        history = fit_torch_val(model, model_best, criterion, optimizer, x_train_torch, y_train_torch, val_split=val_split,
                                patience=patience, num_epochs=epochs, batch_size=x_train_torch.shape[0], verbose=True)
        model = model_best
    elif method == 'Liu_et_al_2020':
        # Certified Monotonic Network training
        bool_mask = np.array([True if i else False for i in mask])
        feature_num = x_train_torch.shape[1]
        mono_feature = sum(mask)
        x_train_torch = torch.cat([x_train_torch[:, bool_mask], x_train_torch[:, ~bool_mask]], dim=1)
        # x_test_torch = torch.cat([x_test_torch[:, bool_mask], x_test_torch[:, ~bool_mask]], dim=1)
        criterion = nn.MSELoss()
        mono_flag = False
        reg_strength = 1e6
        history = None
        while not mono_flag and (reg_strength < 100_000_000_000):
            print(f"Current regularization strength: {reg_strength}")
            model = MLP_relu(**m_params[method])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            history = fit_certified_MN_val(model, criterion, optimizer, reg_strength, x_train_torch, y_train_torch,
                                           val_split, patience, mono_feature, epochs)
            mono_flag = certify_neural_network(model, mono_feature_num=mono_feature)
            if not mono_flag:
                print("Not monotonic. Increasing Regularization. New Trainingepisode.")
            else:
                print("Monotonic. End Training.")
            reg_strength *= 10
        is_monotonic = mono_flag

    elif method == 'Sivaraman_et_al_2020':
        # Special method using Keras/TensorFlow for envelope training
        # Start with training an unconstrained MLP
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        criterion = tf.keras.losses.MeanSquaredError()
        model.compile(loss=criterion, optimizer=optimizer)
        history = model.fit(x_train.astype(np.float32), y_train.astype(np.float32), validation_split=val_split, epochs=epochs,
                            batch_size=x_train.shape[0], verbose=0,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
                                       PrintReportCallback()], shuffle=True)
        history = history.history['loss']

        # save model for envelope technique (Sivaraman)
        data_dir = result_dir + 'ressources/sivaraman/' + str(SPLIT) + '/'
        makeDir(data_dir, True)
        for i in range(len(model.layers)):
            weight = model.layers[i].get_weights()[0]
            bias = model.layers[i].get_weights()[1]
            np.savetxt(data_dir + "weights_layer%d.csv" % (i), weight, delimiter=",")
            np.savetxt(data_dir + "bias_layer%d.csv" % (i), bias, delimiter=",")
        model.save(data_dir + "model.h5")

        # save train and test data as csvs
        data_dir = result_dir + 'ressources/sivaraman/' + str(SPLIT) + '/'
        df_train = pd.DataFrame(x_train.astype(np.float32),
                                columns=[f'x{i + 1}' for i in range(x_train.shape[1])])
        df_train['y'] = y_train.astype(np.float32)
        df_train.to_csv(data_dir + 'train_data.csv')
        df_test = pd.DataFrame(x_test.astype(np.float32),
                               columns=[f'x{i + 1}' for i in range(x_test.shape[1])])
        df_test['y'] = y_test.astype(np.float32)
        df_test.to_csv(data_dir + 'test_data.csv')
        time.sleep(3)

    return history, model, is_monotonic

def predict(model, method, x_train, y_train, x_test, y_test, x_train_torch, x_test_torch, y_test_torch):
    """
    Make predictions using the trained model and calculate the MSE.

    Parameters:
    model (object): The trained model object.
    method (str): The method used for making predictions.
    x_train (np.ndarray): Training input data.
    y_train (np.ndarray): Training target data.
    x_test (np.ndarray): Test input data.
    y_test (np.ndarray): Test target data.
    x_train_torch (torch.Tensor): Training input data in PyTorch tensor format.
    x_test_torch (torch.Tensor): Test input data in PyTorch tensor format.
    y_test_torch (torch.Tensor): Test target data in PyTorch tensor format.

    Returns:
    tuple: MSE outputs for training and test data.
    """
    if method == 'Unconstrained_tf':
        # Using TensorFlow/Keras model for predictions
        y_pred_train = model.predict(x_train.astype(np.float32))
        y_pred_test = model.predict(x_test.astype(np.float32))
        criterion = tf.keras.losses.MeanSquaredError()
        train_loss = criterion(y_train.astype(np.float32), y_pred_train).numpy()
        if not clip:
            test_loss = criterion(y_test.astype(np.float32), y_pred_test).numpy()
        else:
            test_loss = criterion(y_test.astype(np.float32), np.clip(y_pred_test, 0., 1.)).numpy()

    elif method in ['Sill_1997', 'Nolte_2023']:
        # Using PyTorch model for predictions
        y_pred_train = model(x_train_torch).unsqueeze(-1).detach().numpy()
        y_pred_test = model(x_test_torch).unsqueeze(-1).detach().numpy()
        if method == 'Nolte_2023':
            y_pred_train = y_pred_train.squeeze()
            y_pred_test = y_pred_test.squeeze()
        train_loss = mse(y_train, y_pred_train)
        if not clip:
            test_loss = mse(y_test, y_pred_test)
        else:
            test_loss = mse(y_test, np.clip(y_pred_test, 0., 1.))

    elif method == 'Liu_et_al_2020':
        # Certified Monotonic Network predictions
        mono_feature = sum(mask)
        bool_mask = np.array([True if i else False for i in mask])
        x_train_torch = torch.cat([x_train_torch[:, bool_mask], x_train_torch[:, ~bool_mask]], dim=1)
        x_test_torch = torch.cat([x_test_torch[:, bool_mask], x_test_torch[:, ~bool_mask]], dim=1)
        y_pred_train = model(x_train_torch[:, :mono_feature], x_train_torch[:, mono_feature:]).detach().numpy()
        y_pred_test = model(x_test_torch[:, :mono_feature], x_test_torch[:, mono_feature:]).detach().numpy()
        train_loss = mse(y_train, y_pred_train)
        if not clip:
            test_loss = mse(y_test, y_pred_test)
        else:
            test_loss = mse(y_test, np.clip(y_pred_test, 0., 1.))
    elif method == 'Sivaraman_et_al_2020':
        # Special method involving envelope predictions
        data_dir = result_dir + 'ressources/sivaraman/' + str(SPLIT) + '/'
        df_train = pd.read_csv(data_dir + 'train_data.csv')
        data_train = df_train.values
        x_train = data_train[:, 1:-1]
        y_train = data_train[:, -1]

        df_test = pd.read_csv(data_dir + 'test_data.csv')
        data_test = df_test.values
        x_test = data_test[:, 1:-1]
        y_test = data_test[:, -1]

        criterion = tf.keras.losses.MeanSquaredError()
        envelope = ' lower envelope'
        config_file = 'ressources/sivaraman/configurations/mine.txt'
        configurations = readConfigurations(config_file)
        configurations['model_dir'] = data_dir
        model_file = configurations['model_dir']
        configurations['min_max_values'] = {f'x{i + 1}': [0, 1] for i in range(x_train.shape[1])}
        configurations['column_names'] = [f for f in configurations['min_max_values'].keys()]

        # --- import the nn model --------
        sys.path.append('./ressources/sivaraman/Models')
        evaluate = importlib.__import__(configurations['model']).evaluate
        output = importlib.__import__(configurations['model']).output

        # ----- import solver functions --------
        sys.path.append('../../Models/Counter_Examples_Sivaraman_2020/src')
        counter_example_generator_upper = importlib.__import__(
            configurations['solver']).counter_example_generator_upper_env
        counter_example_generator_lower = importlib.__import__(
            configurations['solver']).counter_example_generator_lower_env

        counter_pair_generator = importlib.__import__(configurations['solver']).counter_pair_generator

        parse_data = {'configurations': configurations,
                      'model_file': model_file,
                      'output': output,
                      'counter_example_generator_lower': counter_example_generator_lower,
                      'counter_example_generator_upper': counter_example_generator_upper}
        predictions = {}
        # for test_file in [data_dir + 'train_data.csv', data_dir + 'test_data.csv']:
        for test_file in [data_dir + 'test_data.csv', data_dir + 'train_data.csv']:
            parse_data['test_file'] = test_file
            getEnvelopePredictions(data=parse_data)
            envelope_preds = pd.read_csv(data_dir + 'monotonic_predictions.csv')
            envelope_preds.loc[len(envelope_preds)] = envelope_preds.loc[len(envelope_preds) - 1]
            predictions[test_file] = envelope_preds[envelope]

        y_pred_train = np.array(predictions[data_dir + 'train_data.csv'])
        y_pred_test = np.array(predictions[data_dir + 'test_data.csv'])

        train_loss = criterion(y_train.astype(np.float32), y_pred_train).numpy()
        test_loss = criterion(y_test.astype(np.float32), y_pred_test).numpy()

    elif method == 'Runje_2023':
        y_pred_train = np.array(model.predict({"input_1": x_train}, verbose=0))
        y_pred_test = np.array(model.predict({"input_1": x_test}, verbose=0))
        criterion = tf.keras.losses.MeanSquaredError()
        train_loss = criterion(y_train.astype(np.float32), y_pred_train).numpy()
        if not clip:
            test_loss = criterion(y_test.astype(np.float32), y_pred_test).numpy()
        else:
            test_loss = criterion(y_test.astype(np.float32), np.clip(y_pred_test, 0., 1.)).numpy()

    return train_loss, test_loss

def cross_validation(methods, x, y, n_folds=5, lr=1e-3, val_split=0.25, epochs=100000, patience=1000):
    kf = KFold(n_splits=n_folds)

    mse_train = np.zeros((len(methods), n_folds))
    mse_test = np.zeros((len(methods), n_folds))
    monotonic_liu = np.zeros((len(methods), n_folds))
    times = np.empty((len(methods), n_folds), dtype=object)
    train_histories = np.empty((len(methods), n_folds), dtype=object)
    model_params = {}

    for split, (train_idx, test_idx) in enumerate(kf.split(x)):
        print("split:", split)
        SPLIT = split
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Convert data to PyTorch tensors
        x_train_torch = torch.from_numpy(x_train.astype(np.float32)).clone()
        y_train_torch = torch.from_numpy(y_train.astype(np.float32)).clone()
        x_test_torch = torch.from_numpy(x_test.astype(np.float32)).clone()
        y_test_torch = torch.from_numpy(y_test.astype(np.float32)).clone()

        for m, (method, ModelClass) in enumerate(methods.items()):
            # for m, method in enumerate(methods):
            print(split, method)
            t0 = time.time()

            # Initialize model
            model = ModelClass(**m_params[method])
            if method not in model_params:
                model_params[method] = model.count_params()

            # Train model
            history, model, is_monotonic = fit_model(model, method, x_train, y_train, x_train_torch, y_train_torch, x_test, y_test,
                                                   val_split, epochs, patience, lr)
            t1 = time.time()
            monotonic_liu[m, split] = int(is_monotonic)

            # Get predictions
            mse_train[m, split], mse_test[m, split] = predict(model, method, x_train, y_train, x_test, y_test, x_train_torch, x_test_torch, y_test_torch)
            print(f"\t\tTest loss: {mse_test[m, split]:.8f}")
            t2 = time.time()

            times[m, split] = {"model_init": None, "training": t1 - t0, "inference": t2 - t1}
            train_histories[m, split] = history

    return mse_train, mse_test, times, train_histories, model_params, monotonic_liu


if __name__ == '__main__':
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

    # Create run ID and folder for the experiment results
    result_dir = 'results/' + datetime.now().strftime('%Y%m%d')+ str(datetime.now().hour) + str(datetime.now().minute)+'/'
    makeDir(result_dir)
    SPLIT = 0

    K = 20  # number of Min-Max groups and neurons per group
    clip = True

    # Data Preprocessing
    df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx')
    x, y = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values, df[["Y1"]].values
    dim = x.shape[1]  # number of input variables
    x = preprocessing.MinMaxScaler().fit_transform(x)
    y = preprocessing.MinMaxScaler().fit_transform(y)
    mask = np.array([0, 0, 1, 0, 1, 0, 1, 0])

    # Mapping of methods to corresponding models and parameters
    methods = {
        'Unconstrained_tf': Network_tf,
        'Sill_1997': MonotonicNNMultiv,
        'Liu_et_al_2020': MLP_relu,
        'Sivaraman_et_al_2020': Network_tf,
        'Runje_2023': CMNN,
        'Nolte_2023': ExpressiveNetwork
    }
    # Parameters for each method
    m_params = {
        'Unconstrained_tf': {'width': 128, 'input_shape': (dim,)},
        'Sill_1997': {'n': dim, 'K': K, 'h_K': K, 'mask': mask},
        'Liu_et_al_2020': {'mono_feature': sum(mask),
                           'non_mono_feature': dim - sum(mask),
                           'mono_sub_num': 1,
                           'non_mono_sub_num': 1,
                           'mono_hidden_num': 100,
                           'non_mono_hidden_num': 100},
        'Sivaraman_et_al_2020': {'width': 15, 'input_shape': (dim,)},
        'Runje_2023': {'input_shape': (dim,), 'monotonicity_indicator': mask},
        'Nolte_2023': {'input_dim': dim, 'width': 128, 'sigma': 1, 'monotone_constraints': mask}
    }

    # Perform cross-validation
    mse_train, mse_test, times, train_histories, model_params, monotonic_liu = cross_validation(methods, x, y)

    print(np.mean(mse_test, axis=1))
    print(np.median(mse_test, axis=1))

    # Save results
    np.savez(result_dir + "energy-y1-results-val.npz", mse_train=mse_train, mse_test=mse_test,
             times=times, train_histories=train_histories, model_params=model_params, monotonic_liu=monotonic_liu)
