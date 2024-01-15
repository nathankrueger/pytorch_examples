import torch
from torch.utils.data import *
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.datasets import make_moons

def moveTo(obj, device):
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    elif hasattr(obj, 'to'):
        return obj.to(device)
    else:
        return obj

def train_simple_network(model, loss_func, training_loader, epochs, device='cpu'):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train() # puts the model in training mode
        running_loss = 0.0

        for inputs, labels in tqdm(training_loader, desc="Batch", leave=False):
            inputs = moveTo(inputs, device)
            labels = moveTo(labels, device)

            optimizer.zero_grad()

            y_hat = model(inputs)
            loss = loss_func(y_hat, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

class Simple1DRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

    def __getitem__(self, index):
        return (
            torch.tensor(self.X[index,:], dtype=torch.float32),
            torch.tensor(self.y[index], dtype=torch.float32)
        )
    
    def __len__(self):
        return self.X.shape[0]

def simple_linear_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1)
    )
    #model = torch.nn.Linear(1, 1)

    X = np.linspace(0, 20, num=200)
    y = X + (np.sin(X) * 2) + np.random.normal(size=X.shape)

    training_loader = DataLoader(Simple1DRegressionDataset(X, y), shuffle=True)
    loss_func = torch.nn.MSELoss()
    device = torch.device('cpu')

    train_simple_network(model, loss_func, training_loader, 200, device)

    with torch.no_grad():
        Y_pred = model(torch.tensor(X.reshape(-1, 1), device=device, dtype=torch.float32)).cpu().numpy()

    sns.scatterplot(x=X, y=y, color='blue', label='Data')
    sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Linear Model')

    plt.show()

def visualize2DSoftmax(X, y, model, title=None):
    x_min = np.min(X[:,0]) - 0.5
    x_max = np.max(X[:,0]) + 0.5
    y_min = np.min(X[:,1]) - 0.5
    y_max = np.max(X[:,1]) + 0.5

    xv, yv = np.meshgrid(
            np.linspace(x_min, x_max, num=20),
            np.linspace(y_min, y_max, num=20),
            indexing='ij'
        )
    xy_v = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))

    with torch.no_grad():
        logits = model(torch.tensor(xy_v, dtype=torch.float32))
        y_hat = F.softmax(logits, dim=1).numpy()

    cs = plt.contourf(xv, yv, y_hat[:,0].reshape(20, 20),
                       levels=np.linspace(0, 1, num=20), cmap=plt.cm.RdYlBu)
    ax = plt.gca()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)

    if title is not None:
        ax.set_title(title)

    plt.show()

def simple_logreg_model():
    X, y = make_moons(n_samples=200, noise=0.05)

    classification_dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    training_loader = DataLoader(classification_dataset)

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 2)
    )

    train_simple_network(model, torch.nn.CrossEntropyLoss(), training_loader, epochs=250)
    visualize2DSoftmax(X, y, model)

if __name__ == '__main__':
    simple_logreg_model()
    #simple_linear_model()