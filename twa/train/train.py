import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .layers import *
from .losses import *
from .loaders import *
from .models import *
from .utils import *
from torchdiffeq import odeint
from twa.utils import ensure_dir, write_yaml
from twa.data import topo_point_vs_cycle, pt_attr_idx
from sklearn import metrics
import torch.optim as optim

def train_model(train_dataset, to_angle=True, num_epochs=20, device='cuda', verbose=False, lr=1e-4, 
          batch_size=64, num_lattice=64, report_interval=2, num_classes=2, **kwargs_model):
    """
    Train a AttentionwFC_classify on the given dataset.
    """

    if verbose:
        print('Training data size: {}'.format(len(train_dataset)))
        print('Loading...')
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # train
    in_channels = 1 if to_angle else 2
    in_shape = (in_channels, num_lattice, num_lattice)
    model = AttentionwFC_classify(in_shape, num_classes, **kwargs_model).to(device)


    if verbose:
        print('Training...')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = nn.BCEWithLogitsLoss() if num_classes > 1 else nn.MSELoss()
    
    losses = []
    w = 1
    for epoch in range(num_epochs):
        model.train()
        for idx, (X, label) in enumerate(train_data):
            X, label = X.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(X)
            
            loss = loss_fn(output, label.float())
            loss.backward()
            optimizer.step()
            
            if verbose and (idx % report_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * len(X), len(train_data.dataset),
                    report_interval * idx / len(train_data), loss.item()))
                losses.append(loss.item())

    return model, losses
            



def predict_model(model, test_dataset, verbose=False, save=False, save_dir=None, tt='test', device='cuda', repeats=10, is_prob=True, batch_size=64):
    """
    Evaluate a model on the given dataset.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    
    model.eval()
    
    test_loss = 0
    correct = 0
    thr = 0
    num_samples = len(test_dataset)
    
    preds = []
    latents = []
    outputs = []

    if verbose:
        print('Computing predictions...')
    start = time.time()
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for data, label in test_data:
            data, label = data.to(device), label.to(device)
            
            if is_prob:
                output = torch.zeros_like(label, dtype=torch.float32)
                for i in range(repeats):
                    output += model(data)
                output /= repeats
            else:
                output = model(data)
            loss = loss_fn(output, label.float()).item()  # sum up batch loss
            test_loss += loss
            pred = output > thr  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).all(axis=1).sum().item()
            
            latent = model.encode(data)
            
            # if save:
            outputs.append(torch.Tensor.cpu(output).detach().numpy())
            preds.append(torch.Tensor.cpu(pred).detach().numpy())
            latents.append(torch.Tensor.cpu(latent).detach().numpy())
            
    end = time.time()
    if verbose:
        print('Time: {}'.format(end - start))

    test_loss /= num_samples
    correct /= num_samples

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%)\n'.format(
            test_loss, correct))
        
    res = {f'{tt}_loss': test_loss, f'{tt}_accuracy': correct}
    preds = np.concatenate(preds)
    latents = np.concatenate(latents)
    outputs = np.concatenate(outputs)
    
    y_pt = test_dataset.label[:,0]
    pred_pt = outputs[:,0]
    print(y_pt.shape, pred_pt.shape)
    fpr, tpr, thresholds = metrics.roc_curve(y_pt, pred_pt)
    auc = metrics.auc(fpr, tpr)


    if save and save_dir is not None:
        np.save(os.path.join(save_dir, f'yhat_{tt}.npy'), preds)
        np.save(os.path.join(save_dir, f'embeddings_{tt}.npy'), latents)
        np.save(os.path.join(save_dir, f'outputs_{tt}.npy'), outputs)
        write_yaml(os.path.join(save_dir, f'training_results.yaml'), res)
        print('Saved predictions to: {}'.format(save_dir))

    return correct, auc, outputs 
