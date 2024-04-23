import numpy as np
import tensorflow as tf
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

class Progress():
    """
    Early Stopping monitoring via training progress as proposed by:
    Prechelt, Lutz. Early Stopping—But When? Neural Networks: Tricks of the Trade (2012): 53-67.

    Args:
        strip (int): sequence length of training observed
        threshold (float): lower threshold on progress
    """
    def __init__(self, strip=5, threshold=0.01):
        self.strip = strip
        self.E = np.ones(strip)
        self.t = 0
        self.valid = False
        self.threshold = threshold


    def progress(self):
        """
        Compute the progress based on recent training errors.

        Returns:
            float: Progress value indicating how much the mean training error deviates from min over the strip
        """
        return 1000 * ((self.E.mean() / self.E.min()) - 1.)

    def stop(self):
        if self.valid == False: # less than strip training errors available
            return False
        r = (self.progress() < self.threshold)
        return r

    def update(self, e):
        """
        Update the Progress object with a new training error.

        Args:
            e (float): Training error for the current iteration.

        Returns:
            bool: True if training should stop based on the updated progress, False otherwise.
        """
        self.E[np.mod(self.t, self.strip)] = e
        self.t += 1
        if self.t >= self.strip:
            self.valid = True
        return self.stop()

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EarlyStoppingCallback, self).__init__()
        self.P = Progress(5, threshold=1e-4)

    def on_epoch_end(self, epoch, logs=None):
        stop = self.P.update(logs.get("loss"))
        if stop and epoch > 20000:
            print("\nEarly Stopping.")
            self.model.stop_training = True

# class EarlyStoppingCallback(tf.keras.callbacks.Callback):
#     def __init__(self):
#         super(EarlyStoppingCallback, self).__init__()
#         self.P = Progress(5, threshold=1e-4)
#         self.epoch_c = 0
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.epoch_c += 1
#         stop = self.P.update(logs.get("loss"))
#         if stop and self.epoch_c > 20000:
#             print("\nEarly Stopping.")
#             self.model.stop_training = True


class PrintReportCallback(tf.keras.callbacks.Callback):
    def __init__(self, show_number=100):
        super(PrintReportCallback, self).__init__()
        self.show_number = show_number

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0 or epoch == 1:
            print('\t\tEpoch: [' + str(epoch) + '/' + str(self.params['epochs']) + '] Loss: ' + str(logs.get('loss')))


def fit_torch(model, crit, optim, x, y, num_epochs=100_000, batch_size=None, verbose=True):
    """
    Custom PyTorch training script for univariate datasets (without validation data)

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        crit: The loss function (criterion) to be optimized.
        optim: The optimizer used for training.
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target labels.
        num_epochs (int): The number of epochs for training. Default is 100000.
        batch_size (int): The size of mini-batches. If None, it's set to the size of the entire dataset. Default is None.

    Returns:
        numpy.array: Array containing the mean squared error (MSE) loss history over epochs.
    """
    # Create DataLoader for mini-batch iteration
    dataset = TensorDataset(x, y)
    if batch_size is None:
        batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize progress tracker
    threshold = 1e-4
    P = Progress(5, threshold=threshold)

    mse_history = [] # store mean squared error history

    # Iterate over epochs
    for epoch in range(num_epochs):
        train_losses = []

        # Iterate over mini-batches
        for inputs, targets in dataloader:
            optim.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass
            loss = crit(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optim.step()  # Update weights
            train_losses.append(loss.item())

        # Calculate mean epoch loss
        epoch_loss = np.mean(train_losses)

        # Print epoch loss if verbose is True
        if verbose and (epoch + 1) % 100 == 0:
            print(f"\t\tEpoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.10f}")

        # check for early stopping
        stop = P.update(epoch_loss)
        if (stop):
            print("Early Stopping.")
            return np.array(mse_history)

    return np.array(mse_history)



def makeDir(path, recursive = False):
    if not os.path.exists(path):
        if recursive:
          os.makedirs(path,exist_ok=True)
        else:
            os.mkdir(path)


def fit_torch_val(model, best_model, crit, optim, x, y, val_split=.2, patience=100, num_epochs=100_000, verbose=True, batch_size=None):
    """
    Custom PyTorch training script for univariate datasets (with validation data)

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        best_model (torch.nn.Module): The best model obtained during training (used for early stopping).
        crit: The loss function (criterion) to be optimized.
        optim: The optimizer used for training.
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target labels.
        val_split (float): The fraction of data to be used for validation. Default is 0.2.
        patience (int): Number of epochs to wait before early stopping if validation loss does not improve. Default is 100.
        num_epochs (int): The number of epochs for training. Default is 100000.
        batch_size (int): The size of mini-batches. If None, it's set to the size of the entire dataset. Default is None.

    Returns:
        numpy.array: Array containing the mean squared error (MSE) loss history over epochs.
    """
    # Split data into training and validation sets
    end = int(len(x) - len(x) * val_split)
    x_val, y_val = x[end:], y[end:]
    x, y = x[:end], y[:end]

    # Create DataLoader for training data
    train_dataset = TensorDataset(x, y)
    if batch_size is None:
        batch_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val = float('inf') # Initialize best validation loss
    best_epoch = 0 # Initialize best epoch
    mse_history = [] # Store mean squared error history

    # Iterate over epochs
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        # Iterate over mini-batches
        for inputs, targets in train_loader:
            optim.zero_grad() # Zero gradients
            outputs = model(inputs).unsqueeze(1) # Forward pass
            if len(outputs.size()) == 3:
                outputs = torch.reshape(outputs, outputs.size()[:-1])
            # outputs = model(x_rand[i:i + batch_size]).unsqueeze(1)
            loss = crit(outputs, targets) # Compute loss
            loss.backward() # Backward pass
            optim.step() # Update weights
            train_losses.append(loss.item())

        # Calculate mean epoch loss
        epoch_loss = np.mean(train_losses)
        mse_history.append(epoch_loss)

        # Compute validation loss
        model.eval()
        with torch.no_grad():
            outputs = model(x_val).unsqueeze(1)
            if len(outputs.size()) == 3:
                outputs = torch.reshape(outputs, outputs.size()[:-1])
            val_loss = crit(outputs, y_val)

        # Print epoch loss if verbose is True
        if verbose and (epoch + 1) % 100 == 0:
            print(f"\t\tEpoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.10f}")

        # Update best model if validation loss improves
        if epoch == 0 or val_loss < best_val:
            best_val = val_loss
            best_model.load_state_dict(model.state_dict())
            best_epoch = epoch

        # Early stopping if validation loss does not improve
        if(epoch-best_epoch > patience):
            print(f"Early Stopping at epoch {epoch}.")
            return np.array(mse_history)
    return np.array(mse_history)