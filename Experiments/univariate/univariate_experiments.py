import pandas as pd
import numpy as np
import torch
import random
import time
import sys
import importlib
from datetime import datetime
import pickle

# Importing custom models and utilities
from MA.Models.Unconstrained_NN.model import Network_tf
from MA.Models.MaxMinNetworks_Sill_1997.model import MonotonicNN
from MA.Models.CertifiedMonotonicNetwork_Liu_et_al_2020.utils.networks import MLP_relu
from MA.Models.CertifiedMonotonicNetwork_Liu_et_al_2020.utils.certify import certify_neural_network
from MA.Models.CertifiedMonotonicNetwork_Liu_et_al_2020.model_utils import fit_certified_MN
from MA.Models.Expressive_MNN_Nolte_2023.model import ExpressiveNetwork
from MA.Models.CMNN_Runje_2023.model import CMNN
from MA.Models.Counter_Examples_Sivaraman_2020.src.Utils import readConfigurations
from MA.Models.Counter_Examples_Sivaraman_2020.src.VerifiedPredictions import getEnvelopePredictions
from MA.Experiments.Utils import fit_torch, EarlyStoppingCallback, PrintReportCallback, makeDir
from MA.Datasets.univariate import generate1D

import keras
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error as mse

# Define functions

def fit_model(model, method, x_train, y_train, x_train_torch, y_train_torch, epochs, lr):
    """
    Fit the specified model using the given method.

    Parameters:
    model (object): The model object to be trained.
    method (str): The method used for training the model.
    x_train (np.ndarray): Training input data.
    y_train (np.ndarray): Training target data.
    x_train_torch (torch.Tensor): Training input data in PyTorch tensor format.
    y_train_torch (torch.Tensor): Training target data in PyTorch tensor format.
    epochs (int): Number of epochs for training.
    lr (float): Learning rate for optimization.

    Returns:
    tuple: Training history, training time, and trained model object.
    """
    if method in ['Unconstrained_tf', 'Runje_2023']:
        # TensorFlow/Keras-based training
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        criterion = tf.keras.losses.MeanSquaredError()
        model.compile(loss=criterion, optimizer=optimizer)
        history = model.fit(x_train.astype(np.float32), y_train.astype(np.float32),
                            epochs=epochs, batch_size=x_train.shape[0], verbose=0,
                            callbacks=[EarlyStoppingCallback(), PrintReportCallback()], shuffle=True)
        history = history.history['loss']
        train_time = time.time() - t0

    elif method in ['Sill_1997', 'Nolte_2023']:
        # PyTorch-based training
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        if method == 'Nolte_2023':
            y_train_torch = y_train_torch.reshape(-1,1)
        history = fit_torch(model, criterion, optimizer, x_train_torch, y_train_torch,
                            batch_size=x_train_torch.shape[0], num_epochs=epochs)
        train_time = time.time() - t0

    elif method == 'Liu_et_al_2020':
        # Certified Monotonic Network training
        mono_feature = 1
        mono_flag = False
        reg_strength = 1e2
        criterion = nn.MSELoss()
        while not mono_flag and (reg_strength < 10_000_000):
            print(f"Current regularization strength: {reg_strength}")
            model = ModelClass(**m_params[method])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            history = fit_certified_MN(model, criterion, optimizer, reg_strength,
                                       x_train_torch, y_train_torch.unsqueeze(-1),
                                       x_test_torch, y_test_torch.unsqueeze(-1),
                                       num_epochs=epochs)
            mono_flag = certify_neural_network(model, mono_feature_num=mono_feature)
            if not mono_flag:
                print("Not monotonic. Increasing Regularization. New Trainingepisode.")
            else:
                print("Monotonic. End Training.")
            reg_strength *= 10
        train_time = time.time() - t0
        output_vars['monotonic_liu'][task_id, list(methods.keys()).index(method), trial] = mono_flag

    elif method == 'Sivaraman_et_al_2020':
        # Special method using Keras/TensorFlow for envelope training
        # Start with training an unconstrained MLP
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        criterion = tf.keras.losses.MeanSquaredError()
        model.compile(loss=criterion, optimizer=optimizer)
        history = model.fit(x_train.astype(np.float32), y_train.astype(np.float32), epochs=epochs,
                            batch_size=x_train.shape[0], verbose=0,
                            callbacks=[EarlyStoppingCallback(), PrintReportCallback()], shuffle=True)
        history = history.history['loss']
        train_time = time.time() - t0

        # Save model for envelope method (Sivaraman)
        data_dir = result_dir + 'ressources/sivaraman/' + task + '/' + str(trial) + '/'
        makeDir(data_dir, True)
        for i in range(len(model.layers)):
            weight = model.layers[i].get_weights()[0]
            bias = model.layers[i].get_weights()[1]
            np.savetxt(data_dir + "weights_layer%d.csv" % (i), weight, delimiter=",")
            np.savetxt(data_dir + "bias_layer%d.csv" % (i), bias, delimiter=",")
        model.save(data_dir + "model.h5")

    return history, train_time, model

def predict(model, method, x_train, y_train, x_test, y_test, x_train_torch, x_test_torch, y_test_torch):
    """
    Make predictions using the trained model.

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
    tuple: Predicted outputs for training and test data, test loss, and inference time.
    """
    if method == 'Unconstrained_tf':
        # Using TensorFlow/Keras model for predictions
        y_pred_test = model.predict(x_test.astype(np.float32)).squeeze()
        inference_time = time.time() - train_time
        y_pred_train = model.predict(x_train.astype(np.float32)).squeeze()
        criterion = tf.keras.losses.MeanSquaredError()
        test_loss = criterion(y_test.astype(np.float32), y_pred_test).numpy()
    elif method in ['Sill_1997', 'Nolte_2023']:
        # Using PyTorch model for predictions
        y_pred_test = model(x_test_torch)
        inference_time = time.time() - train_time
        y_pred_train = model(x_train_torch)
        criterion = nn.MSELoss()
        if method == 'Nolte_2023':
            y_pred_test = y_pred_test.squeeze()
        test_loss = criterion(y_pred_test, y_test_torch).item()
    elif method == 'Liu_et_al_2020':
        # Certified Monotonic Network predictions
        mono_feature = 1
        y_pred_test = model(x_test_torch[:, :mono_feature], x_test_torch[:, mono_feature:])
        inference_time = time.time() - train_time
        y_pred_train = model(x_train_torch[:, :mono_feature], x_train_torch[:, mono_feature:])
        criterion = nn.MSELoss()
        test_loss = criterion(y_pred_test.squeeze(), y_test_torch).item()
    elif method == 'Sivaraman_et_al_2020':
        # Special method involving envelope predictions
        criterion = tf.keras.losses.MeanSquaredError()
        # This part involves additional processing for envelope predictions
        envelope = ' lower envelope'
        data_dir = result_dir + 'ressources/sivaraman/' + task + '/' + str(trial) + '/'
        config_file = 'ressources/sivaraman/configurations/mine.txt'
        configurations = readConfigurations(config_file)
        configurations['model_dir'] = data_dir
        model_file = configurations['model_dir']
        # solver_times = []
        configurations['min_max_values'] = {'x': list((0, 1))}

        # --- import the nn model --------
        sys.path.append('./ressources/sivaraman/Models')
        evaluate = importlib.__import__(configurations['model']).evaluate
        # t0 = time.time()
        output = importlib.__import__(configurations['model']).output
        # t1 = time.time()

        # ----- import solver functions --------
        sys.path.append('../../Models/Counter_Examples_Sivaraman_2020/src')
        counter_example_generator_upper = importlib.__import__(
            configurations['solver']).counter_example_generator_upper_env
        counter_example_generator_lower = importlib.__import__(
            configurations['solver']).counter_example_generator_lower_env

        counter_pair_generator = importlib.__import__(configurations['solver']).counter_pair_generator

        # -------- COMET ---------
        df_train = pd.DataFrame({"x": x_train.astype(np.float32).squeeze(), "y": y_train.astype(np.float32)}).to_csv(
            data_dir + 'train_data.csv')
        df_test = pd.DataFrame({"x": x_test.astype(np.float32).squeeze(), "y": y_test.astype(np.float32)}).to_csv(
            data_dir + 'test_data.csv')

        parse_data = {'configurations': configurations,
                      'model_file': model_file,
                      'output': output,
                      'counter_example_generator_lower': counter_example_generator_lower,
                      'counter_example_generator_upper': counter_example_generator_upper}
        predictions = {}
        t2_measured = False
        for test_file in [data_dir + 'test_data.csv', data_dir + 'train_data.csv']:
            parse_data['test_file'] = test_file
            getEnvelopePredictions(data=parse_data)
            if not t2_measured:
                inference_time = time.time() - train_time
                t2_measured = True
            envelope_preds = pd.read_csv(data_dir + 'monotonic_predictions.csv')
            envelope_preds.loc[len(envelope_preds)] = envelope_preds.loc[len(envelope_preds) - 1]
            predictions[test_file] = envelope_preds[envelope]

        y_pred_train = np.array(predictions[data_dir + 'train_data.csv'])
        y_pred_test = np.array(predictions[data_dir + 'test_data.csv'])
        test_loss = criterion(y_test.astype(np.float32), y_pred_test).numpy()
    elif method == 'Runje_2023':
        y_pred_test = np.array(model.predict({"input_1": x_test}, verbose=0)).squeeze()
        inference_time = time.time() - train_time
        y_pred_train = np.array(model.predict({"input_1": x_train}, verbose=0)).squeeze()
        criterion = tf.keras.losses.MeanSquaredError()
        test_loss = criterion(y_test.astype(np.float32), y_pred_test).numpy()

    return y_pred_train, y_pred_test, test_loss, inference_time

def save_results(method, task_id, trial, y_pred_train, y_pred_test, y_train, y_test, test_loss, history, train_time, inference_time):
    """
    Save results of model training and prediction.

    Parameters:
    method (str): The method used for training and prediction.
    task_id (int): Task ID for the experiment.
    trial (int): Trial number for the experiment.
    y_pred_train (np.ndarray or torch.Tensor): Predicted outputs for training data.
    y_pred_test (np.ndarray or torch.Tensor): Predicted outputs for test data.
    y_train (np.ndarray): True outputs for training data.
    y_test (np.ndarray): True outputs for test data.
    test_loss (float): Loss value on the test data.
    history (list): Training history of the model.
    train_time (float): Training time of the model.
    inference_time (float): Inference time of the model.
    """
    method_id = list(methods.keys()).index(method)
    if torch.is_tensor(y_pred_train):
        output_vars['Y_Hat_train'][task_id, method_id, trial] = y_pred_train.detach().numpy().squeeze()
        output_vars['Y_Hat_test'][task_id, method_id, trial] = y_pred_test.detach().numpy().squeeze()
        output_vars['MSE_train'][task_id, method_id, trial] = mse(y_train, y_pred_train.detach().numpy().squeeze())
        output_vars['MSE_test'][task_id, method_id, trial] = mse(y_test, y_pred_test.detach().numpy().squeeze())
    else:
        output_vars['Y_Hat_train'][task_id, method_id, trial] = y_pred_train
        output_vars['Y_Hat_test'][task_id, method_id, trial] = y_pred_test
        output_vars['MSE_train'][task_id, method_id, trial] = mse(y_train, y_pred_train)
        output_vars['MSE_test'][task_id, method_id, trial] = mse(y_test, y_pred_test)

    output_vars['times'][task_id, method_id, trial] = {"model_init": None, "training": train_time, "inference": inference_time}
    output_vars['train_histories'][task_id, method_id, trial] = history

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
tf.random.set_seed(seed)

# Define constants
T = 1
K = 20
N_train = 50
N_test = 1000
sigma = 0.01
lr = 0.001
epochs = 5000
tasks = ['sigmoid10', 'sq', 'sqrt', 'non_smooth','non_mon1', 'non_mon2', 'non_mon3']

# Experiment setup
result_dir = 'results/' + datetime.now().strftime('%Y%m%d%H%M') + '/'
makeDir(result_dir)

# Mapping of methods to corresponding models and parameters
methods = {
    'Unconstrained_tf': Network_tf,
    'Sill_1997': MonotonicNN,
    'Liu_et_al_2020': MLP_relu,
    # 'Sivaraman_et_al_2020': Network_tf,
    'Runje_2023': CMNN,
    'Nolte_2023': ExpressiveNetwork
}

# Parameters for each method
m_params = {
    'Unconstrained_tf': {'width':128, 'input_shape':(1,)},
    'Sill_1997': {'n':1, 'K':K, 'h_K':K},
    'Liu_et_al_2020': {'mono_feature':1,
                       'non_mono_feature':0,
                       'mono_sub_num':1,
                       'non_mono_sub_num':1,
                       'mono_hidden_num':100,
                       'non_mono_hidden_num':100},
    'Sivaraman_et_al_2020': {'width':15, 'input_shape':(1,)},
    'Runje_2023': {'input_shape':(1,), 'monotonicity_indicator':[1]},
    'Nolte_2023': {'width':128, 'sigma':1, 'monotone_constraints':[1]}
}

# Output variables for storing experiment results
num_tasks = len(tasks)
num_methods = len(methods)
output_vars = {
        "MSE_train": np.zeros((num_tasks, num_methods, T)),
        "MSE_test": np.zeros((num_tasks, num_methods, T)),
        "X_train": np.zeros((num_tasks, T, N_train)),
        "Y_train": np.zeros((num_tasks, T, N_train)),
        "X_test": np.zeros((num_tasks, T, N_test)),
        "Y_test": np.zeros((num_tasks, T, N_test)),
        "Y_Hat_train": np.zeros((num_tasks, num_methods, T, N_train)),
        "Y_Hat_test": np.zeros((num_tasks, num_methods, T, N_test)),
        "times": np.empty((num_tasks, num_methods, T), dtype=object),
        "train_histories": np.empty((num_tasks, num_methods, T), dtype=object),
        "model_params": {},
        "monotonic_liu": np.empty((num_tasks, num_methods, T), dtype=bool),
        "methods": list(methods.keys()),
        "tasks": tasks
    }

# Running the experiments
for trial in range(T):
    for task_id, task in enumerate(tasks):
        # Generate data for the task
        x_train, y_train = generate1D(task, sigma=sigma, random_x=True, N=N_train)
        x_test, y_test = generate1D(task, sigma=0., random_x=False, N=N_test)
        output_vars['X_test'][task_id, trial] = x_test.reshape(-1)
        output_vars['Y_test'][task_id, trial] = y_test
        output_vars['X_train'][task_id, trial] = x_train.reshape(-1)
        output_vars['Y_train'][task_id, trial] = y_train

        # Convert data to PyTorch tensors
        x_train_torch = torch.from_numpy(x_train.astype(np.float32)).clone()
        y_train_torch = torch.from_numpy(y_train.astype(np.float32)).clone()
        x_test_torch = torch.from_numpy(x_test.astype(np.float32)).clone()
        y_test_torch = torch.from_numpy(y_test.astype(np.float32)).clone()

        for method, ModelClass in methods.items():
            print(method, task, trial)
            t0 = time.time()

            # Initialize model
            model = ModelClass(**m_params[method])
            if method not in output_vars['model_params']:
                output_vars['model_params'][method] = model.count_params()

            # Train model
            history, train_time, model = fit_model(model, method, x_train, y_train, x_train_torch, y_train_torch, epochs, lr)

            # Get predictions
            y_pred_train, y_pred_test, test_loss, inference_time = predict(model, method, x_train, y_train, x_test, y_test, x_train_torch, x_test_torch, y_test_torch)
            print(f"\t\tTest loss: {test_loss:.8f}")

            # Save results
            save_results(method, task_id, trial, y_pred_train, y_pred_test, y_train, y_test, test_loss, history, train_time, inference_time)

# Save experiment data using pickle
makeDir(result_dir + 'pickles/')
filenames = [result_dir + 'pickles/' + i +'.pkl' for i in output_vars.keys()]
for data, filename in zip(output_vars.values(), filenames):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)