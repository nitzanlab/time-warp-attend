import os
import torch
import time
import numpy as np
import torch.nn.functional as F
from torch import nn
from twa.train.loaders import VecTopoDataset
from twa.train.models import load_model
from torch.utils.data import DataLoader
from twa.utils import ensure_dir, write_yaml
from twa.data import topo_point_vs_cycle, pt_attr_idx
from sklearn import metrics
import torch.optim as optim
from phase2vec.utils._utils import get_command_defaults
from phase2vec.cli import generate_net_config


def train_model(train_dataset, model_type=None, num_epochs=20, device='cuda', verbose=False, lr=1e-4, 
          batch_size=64, report_interval=2, num_classes=2, **kwargs_model):
    """
    Train model on a given dataset.
    """

    if verbose:
        print('Training data size: {}'.format(len(train_dataset)))
        print('Loading...')
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    data,label = next(iter(train_data))
    in_shape = list(data.shape[1:])
    model = load_model(model_type, in_shape=in_shape, out_shape=num_classes, **kwargs_model).to(device)

    if verbose:
        print('Training...')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = nn.BCEWithLogitsLoss() if num_classes > 1 else nn.MSELoss()
    loss_fn = nn.MSELoss() if model_type == 'AE' else loss_fn

    losses = []
    w = 1
    for epoch in range(num_epochs):
        model.train()
        for idx, (X, label) in enumerate(train_data):
            X, label = X.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(X)
            if model_type == 'AE':
                loss = loss_fn(output, X)
            else:
                loss = loss_fn(output, label.float())
            loss.backward()
            optimizer.step()
            
            if verbose and (idx % report_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * len(X), len(train_data.dataset),
                    report_interval * idx / len(train_data), loss.item()))
                losses.append(loss.item())

    return model, losses


def train_model_alt(train_dataset, model_type=None, pretrained_path=None, num_classes=2, pretrain_data_path='../phase2vec/output/data/polynomial/', **kwargs_train):
    """
    Handles pretrained alternative models. Options include:
    - vf_AEFC - vector field AE + FC
    - p2v_AEFC - phase2vec + FC
    Our model variations and FC of parameter representation do not require pretraining.
    """
    kwargs_train = {} if kwargs_train is None else kwargs_train

    if model_type == 'vf_AEFC':
        print('Training vector field AE...')
        datatype = 'vector'
        datasize = 10000
        if not os.path.isdir(pretrain_data_path):
            print('Pretrain data not found in {}'.format(pretrain_data_path))

        print('Loading pretrain data from {}'.format(pretrain_data_path))
        train_dataset2 = VecTopoDataset(pretrain_data_path, datatype=datatype, datasize=datasize)
        model_ae, _ = train_model(train_dataset=train_dataset2, verbose=False, model_type='AE')
        
        model_cl = load_model(model_type='FC', in_shape=model_ae.latent_dim, out_shape=num_classes)
        kwargs_train['model_ae'] = model_ae
        kwargs_train['model_cl'] = model_cl
        model_type = 'AEFC'
        
    elif model_type == 'p2v_AEFC':
        print('Loading phase2vec pretrained model...')
        net_info = get_command_defaults(generate_net_config)
        model_type = net_info['net_class']

        # These parameters are not considered architectural parameters for the net, so we delete them before they're passed to the net builder. 
        del net_info['net_class']
        del net_info['output_file']
        del net_info['pretrained_path']
        del net_info['ae']

        model_p2v = load_model(model_type='p2v', pretrained_path=pretrained_path, **net_info)
        model_cl = load_model(model_type='FC', in_shape=model_p2v.latent_dim, out_shape=num_classes)
        kwargs_train['model_ae'] = model_p2v
        kwargs_train['model_cl'] = model_cl
        pretrained_path = None
        model_type = 'AEFC'
        
    print('Training classifier...')
    model, losses = train_model(train_dataset=train_dataset, model_type=model_type, pretrained_path=pretrained_path, **kwargs_train)

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
            
            outputs.append(torch.Tensor.cpu(output).detach().numpy())
            preds.append(torch.Tensor.cpu(pred).detach().numpy())
            
            
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
    outputs = np.concatenate(outputs)
    
    y_pt = test_dataset.label[:,0]
    pred_pt = outputs[:,0]

    auc = None
    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_pt, pred_pt)
        auc = metrics.auc(fpr, tpr)
    except:
        pass

    if save and save_dir is not None:
        np.save(os.path.join(save_dir, f'yhat_{tt}.npy'), preds)
        # np.save(os.path.join(save_dir, f'embeddings_{tt}.npy'), latents)
        np.save(os.path.join(save_dir, f'outputs_{tt}.npy'), outputs)
        write_yaml(os.path.join(save_dir, f'training_results.yaml'), res)
        print('Saved predictions to: {}'.format(save_dir))

    return correct, auc, outputs
