# f(x,y) = e^(sin(x)^2) / (x-y)^2 + (x-y)^2  solve for x=0.2, y=10
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return torch.exp(torch.pow(torch.sin(x), 2)) / torch.pow(x-y, 2) + torch.pow(x-y, 2)

def f_np(x, y):
    return (np.e ** (np.sin(x) ** 2)) / (x - y) ** 2 + (x - y) ** 2

x = torch.nn.Parameter(torch.tensor([0.2, 10.0]), requires_grad=True)
epsilon = 1e-5
lr = 1e-1

optimizer = torch.optim.SGD(params=[x], lr=lr)
for epoch in range(100):
    optimizer.zero_grad()
    loss = f(x[0], x[1])

    # compute gradient
    loss.backward()

    # x = x - lr * x.grad
    optimizer.step()

print(f'x: {x.detach().numpy()[0]} y:{x.detach().numpy()[1]}')

sns.set_style('whitegrid')
x_vals = np.linspace(-10,10,500)
y_vals = np.linspace(-10,10,500)
xplot, yplot = np.meshgrid(x_vals, y_vals)
zplot = f_np(xplot, yplot)
axes = plt.axes(projection='3d')
axes.plot_surface(xplot, yplot, zplot)
axes.scatter3D([3.14],[4.14],f_np(np.asarray([3.14]), np.asarray([4.14])))

plt.show()