import torch
import torch.nn as nn

"""
Monotonic Neural Network.

Implementation follows:
Joseph Sill. Monotonic Networks. Advances in Neural Information Processing Systems 10, 1997.

"""
class MonotonicNN(nn.Module):
    def __init__(self, n, K, h_K, b_z = 1., b_t = 1.):
        """
        Initialize a Monotonic Neural Network.

        Parameters:
        - n (int): Input dimension.
        - K (int): Number of groups/layers.
        - h_K (int): Number of neurons per group/layer.
        - b_z (float): Standard deviation for weight initialization.
        - b_t (float): Standard deviation for bias initialization.
        """
        super(MonotonicNN, self).__init__()
        self.K = K
        self.h_K = h_K
        self.b_z = b_z
        self.b_t = b_t
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.z = nn.ParameterList([nn.Parameter(torch.ones(h_K, n), requires_grad=True) for i in range(K)]) # weights
        self.t = nn.ParameterList([nn.Parameter(torch.ones(h_K), requires_grad=True) for i in range(K)]) # bias
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset model parameters (weights and biases) using truncated normal initialization.
        """
        for i in range(self.K):
            torch.nn.init.trunc_normal_(self.z[i], std=self.b_z)
            torch.nn.init.trunc_normal_(self.t[i], std=self.b_t)

    def forward(self, x):
        for i in range(self.K):  # loop over groups
            # hidden layer
            w = torch.exp(self.z[i])  # positive weights
            g = torch.matmul(x, w.t()) + self.t[i]
            g = torch.max(g, axis=1)
            # output layer
            if i==0:
                y = g.values
            else:
                y = torch.minimum(y, g.values)
        return y

    def count_params(self):
        """
        Count the total number of parameters (weights and biases) in the model.

        Returns:
        - int: Total number of parameters.
        """
        n_weights = sum(p.numel() for p in self.z)
        n_biases = sum(p.numel() for p in self.t)
        return n_weights + n_biases

class MonotonicNNMultiv(nn.Module):
    def __init__(self, n, K, h_K, mask, b_z = 1., b_t = 1.):
        """
        Initialize a Multivariate Monotonic Neural Network.

        Parameters:
        - n (int): Input dimension.
        - K (int): Number of groups/layers.
        - h_K (int): Number of neurons per group/layer.
        - mask (list or array-like): Mask for enforcing monotonic constraints.
        - b_z (float): Standard deviation for weight initialization.
        - b_t (float): Standard deviation for bias initialization.
        """
        super(MonotonicNNMultiv, self).__init__()
        self.K = K
        self.h_K = h_K
        self.b_z = b_z
        self.b_t = b_t
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.z = nn.ParameterList([nn.Parameter(torch.ones(h_K, n), requires_grad=True) for i in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.ones(h_K), requires_grad=True) for i in range(K)])
        self.softmax = nn.Softmax(dim=1)
        self.mask = torch.BoolTensor(mask)
        # assert mask.shape == (n, )
        self.mask_inv = ~self.mask
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset model parameters (weights and biases) using truncated normal initialization.
        """
        for i in range(self.K):
            torch.nn.init.trunc_normal_(self.z[i], std=self.b_z)
            torch.nn.init.trunc_normal_(self.t[i], std=self.b_t)

    def forward(self, x):
        for i in range(self.K):  # loop over groups
            # hidden layer
            w = torch.exp(self.z[i])  # positive weights
            w = self.mask * w + self.mask_inv * self.z[i]  # restore non-constrained
            g = torch.matmul(x, w.t()) + self.t[i]
            g = torch.max(g, axis=1)
            # output layer
            if i==0:
                y = g.values
            else:
                y = torch.minimum(y, g.values)
        return y

    def count_params(self):
        """
        Count the total number of parameters (weights and biases) in the model.

        Returns:
        - int: Total number of parameters.
        """
        n_weights = sum(p.numel() for p in self.z)
        n_biases = sum(p.numel() for p in self.t)
        return n_weights + n_biases
