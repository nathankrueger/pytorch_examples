import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return torch.pow(x-2.0, 2)

def fP(x):
    return 2*x - 4

# function f(x) = (x-2)^2
x_vals = np.linspace(-7,9,300)
y_vals = f(torch.tensor(x_vals)).numpy()
sns.lineplot(x=x_vals, y=y_vals, label='$f(x) = (x - 2)^2$')

# derivative of function f(x) = f'(x) = 2x - 4
y_vals = fP(torch.tensor(x_vals)).numpy()
sns.lineplot(x=x_vals, y=y_vals, label='$\'f(x) = 2x - 4$')

# line x = 0
y_vals = np.zeros(y_vals.shape[0])
sns.lineplot(x=x_vals, y=y_vals, label='f(x) = 0')

plt.show()