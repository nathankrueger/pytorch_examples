import torch
from torch.utils.data import *
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import time
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

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

def train_simple_network(
        model: torch.nn.Module,
        loss_func,
        training_loader,
        validation_loader=None,
        score_funcs: dict=None, # dictionary of metric name to metric func
        epochs: int=10,
        device: str='cpu',
        checkpoint_path: str=None
    ) -> pd.DataFrame:
    
    to_track = ['epoch', 'total time', 'train loss']
    if validation_loader is not None:
        to_track.append('validation loss')
    for eval_score in score_funcs:
        to_track.append(f'train {eval_score}')
        if validation_loader is not None:
            to_track.append(f'validation {eval_score}')

    total_train_time = 0
    results = {}

    for item in to_track:
        results[item] = []
        
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train() # puts the model in training mode

        total_train_time += run_epoch(
            model,
            optimizer,
            training_loader,
            loss_func,
            device,
            results,
            score_funcs,
            prefix='train',
            desc='Training'
        )

        results['total time'].append(total_train_time)
        results['epoch'].append(epoch)

        if validation_loader is not None:
            model = model.eval()
            
            with torch.no_grad():
                total_train_time += run_epoch(
                    model,
                    optimizer,
                    validation_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix='validation',
                    desc='Validation'
                )

        if checkpoint_path is not None:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'results': results
                },
                checkpoint_path
            )

    return pd.DataFrame.from_dict(results)

def run_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader,
        loss_func,
        device,
        results,
        score_funcs,
        prefix: str='',
        desc: str=None
    ):

    running_loss = []
    y_true = []
    y_pred = []

    # start time for running through one epoch of the data
    start = time.time()

    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        # move the batch to the compute device
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs)

        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()

            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

    # end time for the epoch
    end = time.time()

    y_pred = np.asarray(y_pred)
    is_classification = y_pred.shape[1] > 1

    if len(y_pred.shape) == 2 and is_classification:
        y_pred = np.argmax(y_pred, axis=1)

    # calculate the average loss over all batches the current epoch is comprised of
    results[f'{prefix} loss'].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        try:
            results[f'{prefix} {name}'].append(score_func(y_true, y_pred))
        except:
            results[f'{prefix} {name}'].append(float('NaN'))

    # return the time spent on the epoch
    return end - start

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

def simple_logreg_model(hidden_dim=30, batch_size=32):
    # prepare the synthetic dataset
    X, y = make_moons(n_samples=1000, noise=0.05)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

    training_classification_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    training_loader = DataLoader(training_classification_dataset, batch_size=batch_size, shuffle=True)

    validation_classification_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    validation_loader = DataLoader(validation_classification_dataset, batch_size=batch_size, shuffle=True)

    # create the torch model
    model = torch.nn.Sequential(
        torch.nn.Linear(2, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, 2)
    )

    # train it!
    results = train_simple_network(
        model=model,
        loss_func=torch.nn.CrossEntropyLoss(),
        training_loader=training_loader,
        validation_loader=validation_loader,
        score_funcs={'F1': f1_score, 'Accuracy': accuracy_score},
        epochs=250,
        device='cpu',
        checkpoint_path='serialized_model.pt'
    )

    sns.lineplot(x='epoch', y='train Accuracy', data=results, label='Train')
    sns.lineplot(x='epoch', y='validation Accuracy', data=results, label='Validation')
    plt.show()

    visualize2DSoftmax(X, y, model)

if __name__ == '__main__':
    simple_logreg_model(hidden_dim=30, batch_size=1)
    #simple_linear_model()