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

# example use of autograd
x = torch.tensor([-5.0], requires_grad=True)
y = f(x)
y.backward()
print(f"Gradient of f'(-5) = {x.grad.numpy()[0]}")

# create a tensor that is tracked by autograd
x = torch.tensor([-5.0], requires_grad=True)
x_cur = x.clone()
x_prev = x_cur * 100
epsilon = 1e-5
learn_rate = 1e-1

# keep moving the position in the opposite direction of the derivative
# (gradient) until we are hovering around a local / global minimum
while torch.linalg.norm(x_cur - x_prev) > epsilon:
    x_prev = x_cur.clone()

    # do some math operations on 'x' which requires_grad so it is tracked by autograd
    y = f(x)

    # tell autograd to compute the gradient
    y.backward()
    x.data -= learn_rate * x.grad

    # zero out the gradient for the subsequent iteration, this isn't done automatically by torch
    x.grad.zero_()
    x_cur = x.data

# confirm the computed minimum is close to f'(x) = 0 ---> x = 2
print(x_cur.detach().numpy()[0])