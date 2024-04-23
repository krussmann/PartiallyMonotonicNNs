import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# Configure Matplotlib Styling
# For all Params see : https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
params = {
    'legend.fontsize': 16,
    'axes.labelsize': 14,
    'axes.titlesize': 24, # figure title
    'xtick.labelsize': 14,
    'ytick.labelsize':14,
    'font.family': 'serif',
    # 'legend.shadow': True,
    # 'xtick.labelbottom': True,
    'figure.subplot.bottom': .125,
    'figure.subplot.top':.8,
    'axes.titlepad':10,
}

def generate1D(function, sigma=0., random_x=False, N=1000):
    """
    Generate toy example univariate data.

    Args:
        function (str): The type of function to generate data from.
            Available options: 'sigmoid10', 'sq', 'sqrt', 'non_smooth', 'non_mon1', 'non_mon2', 'non_mon3'.
        sigma (float): Standard deviation of the Gaussian noise to be added to the generated y values.
        random_x (bool): If True, randomly generate x values within a certain range. If False, generate x values uniformly.
        N (int): Number of data points to generate.

    Returns:
        Tuple: A tuple containing x values and corresponding y values as numpy arrays.
    """
    if function not in ['sigmoid10', 'sq', 'sqrt', 'non_smooth', 'non_mon1', 'non_mon2', 'non_mon3']:
        raise ValueError("Unsupported function. Choose from 'sigmoid10', 'sq', 'sqrt', 'non_smooth', 'non_mon1', 'non_mon2', 'non_mon3'.")

    # Define the range of x values based on the selected function
    x_range = (-10, 10) if function == "non_mon1" else (0,20) if function == ("non_mon2") else (0,1)

    # Generate x values
    if random_x:
        x = np.array([random.uniform(x_range[0], x_range[1]) for _ in range(N)])
        x.sort()  # Sort x values if randomly generated
    else:
        x = np.arange(x_range[0], x_range[1], (x_range[1]-x_range[0]) / N)

    # Generate y values based on the selected function

    if function ==  'sigmoid10':
        y = 1. / (1. + np.exp(-(x-1 / 2.) * 10.))
    elif function == 'sq':
        y = x ** 2
    elif function == 'sqrt':
        y = np.sqrt(x)
    elif function == 'non_smooth':
        segments = [(0, 0), (0.25, 0.5), (0.75, 0.5), (1, 1)]
        y = np.interp(x, [point[0] for point in segments], [point[1] for point in segments], left=0, right=0)
    elif function == 'non_mon1':
        y = -0.1*x*np.cos(x)
        x = (x - x_range[0]) / (x_range[1] - x_range[0])
        y = (y - min(y)) / (max(y) - min(y))
    elif function == 'non_mon2':
        amplitude = .5
        frequency = 1.0
        linear_slope = 0.2
        sinusoidal = amplitude * np.sin(frequency * x)
        linear = linear_slope * x
        y = sinusoidal + linear
        x = (x - x_range[0]) / (x_range[1] - x_range[0])
        y = (y - min(y)) / (max(y) - min(y))
    elif function == 'non_mon3':
        segments = [(0, 0), (0.25, 0.66), (0.5, 0.33), (0.75, 1), (1, 0.66)]
        y = np.interp(x, [point[0] for point in segments], [point[1] for point in segments], left=0, right=0)

    # Add Gaussian noise to y values
    y += sigma * np.random.normal(0,1.,N)

    return x.reshape(N,1), y


if __name__ == '__main__':
    pylab.rcParams.update(params)
    x_line, y_line = generate1D('non_mon1', 0.1, True, 50) # training data
    x_line_t, y_line_t = generate1D('non_mon1', 0, False, 1000) # test data

    # create the plot including test and training data
    plt.plot(x_line_t, y_line_t, '--', color='red', label='Test Data')
    plt.scatter(x_line, y_line, color='blue', label='Training Data')

    # # export the dataset
    # df = pd.DataFrame({"x":x_line.squeeze(), "y": y_line})
    # df = df.sample(frac=1)
    # print(df.head())
    # df.to_csv("data.csv")

    plt.title('f(x) = non monotonic 3 \n (Noise=.1, N=50)')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    # save plot
    # save_path = '..\\docs\\assets\\'
    # plt.savefig(save_path + 'toy_example2.png')
    plt.show()