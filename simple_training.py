import torch
from torch.utils.data import *
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def train_simple_model(model, loss_func, training_loader, epochs, device):
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
        torch.nn.Linear(10, 1)
    )
    model = torch.nn.Linear(1, 1)

    X = np.linspace(0, 20, num=200)
    y = X + (np.sin(X) * 2) + np.random.normal(size=X.shape)

    training_loader = DataLoader(Simple1DRegressionDataset(X, y), shuffle=True)
    loss_func = torch.nn.MSELoss()
    device = torch.device('cpu')

    train_simple_model(model, loss_func, training_loader, 30, device)

    with torch.no_grad():
        Y_pred = model(torch.tensor(X.reshape(-1, 1), device=device, dtype=torch.float32)).cpu().numpy()

    sns.scatterplot(x=X, y=y, color='blue', label='Data')
    sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Linear Model')

    plt.show()

if __name__ == '__main__':
    simple_linear_model()