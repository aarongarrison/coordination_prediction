import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import argparse

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
from torch_geometric.nn import GAT, GCN
import torch_scatter

from train import run_train
from utils.data_utils import LigandDataset
from utils.initialization import initialize_files

def save_file(path, hidden_channels, num_layers, dropout, activation, val_loss, epoch):
    #open the saved files
    # path = 'data/gat_sweep.npz'
    npzfile = np.load(path, allow_pickle=True)
    hc_list = npzfile['hc_list']
    nl_list = npzfile['nl_list']
    do_list = npzfile['do_list']
    act_list = npzfile['act_list']
    loss_list = npzfile['loss_list']
    epoch_list = npzfile['epoch_list']

    #append the parameters
    hc_list = np.append(hc_list, hidden_channels)
    nl_list = np.append(nl_list, num_layers)
    do_list = np.append(do_list, dropout)
    act_list = np.append(act_list, activation)
    loss_list = np.append(loss_list, val_loss)
    epoch_list = np.append(epoch_list, epoch+1)

    #save parameters
    np.savez(path, hc_list=hc_list, nl_list=nl_list, do_list=do_list, act_list=act_list, loss_list=loss_list, epoch_list=epoch_list)
    print(f'Saved to {path} | HC:{hc_list[-1]} | NL:{nl_list[-1]} | DO:{do_list[-1]} | ACT:{act_list[-1]}')


def sweep_gat(path,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device):
    """ Hyperparameter sweep for GAT model. """
    print('Running GAT')
    # for hidden_channels in [5, 10, 15, 20, 25]:
    for hidden_channels in [10, 15, 20, 25]:
        # for num_layers in [2, 4, 6, 8, 10]:
        for num_layers in [2]:
            # for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for dropout in [0.1, 0.3, 0.5]:
                # for activation in ['tanh', 'relu']:
                for activation in ['relu']:
                    #Implement training / evaluation loop here - need to calculate val loss
                    model = GAT(-1, hidden_channels=hidden_channels, num_layers=num_layers,
                                out_channels=1, dropout=dropout, act=activation)

                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                    
                    epoch, _, _, best_loss = run_train(model=model,
                                                       train_loader=train_loader,
                                                       val_loader=val_loader,
                                                       num_epochs=50,
                                                       patience=5,
                                                       optimizer=optimizer,
                                                       scheduler=scheduler,
                                                       device=device, verbose=False)
                    save_file(path,
                              hidden_channels=hidden_channels,
                              num_layers=num_layers,
                              dropout=dropout,
                              activation=activation,
                              val_loss=best_loss,
                              epoch=epoch)


def sweep_gcn(path,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device):
    """ Hyperparameter sweep for GCN model. """
    print('Running GCN')
    # for hidden_channels in [5, 10, 15, 20, 25]:
    for hidden_channels in [10, 15, 20, 25]:
        # for num_layers in [2, 4, 6, 8, 10]:
        for num_layers in [2]:
            # for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for dropout in [0.1, 0.3, 0.5]:
                # for activation in ['tanh', 'relu']:
                for activation in ['relu']:
                    #Implement training / evaluation loop here - need to calculate val loss
                    model = GCN(-1, hidden_channels=hidden_channels, num_layers=num_layers,
                                out_channels=1, dropout=dropout, act=activation)

                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

                    
                    epoch, _, _, best_loss = run_train(model=model,
                                                       train_loader=train_loader,
                                                       val_loader=val_loader,
                                                       num_epochs=100,
                                                       patience=10,
                                                       optimizer=optimizer,
                                                       scheduler=scheduler,
                                                       device=device, verbose=False)

                    save_file(path,
                              hidden_channels=hidden_channels,
                              num_layers=num_layers,
                              dropout=dropout,
                              activation=activation,
                              val_loss=best_loss,
                              epoch=epoch)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path-dir', type=str, help='Path to directory to save')
    parser.add_argument('--data-dir', type=str, help="Path to directory holding train and val torch Datasets.")
    parser.add_argument('--gnn-type', type=str, help='Either "GAT" or "GCN".')
    args = parser.parse_args()

    save_path_dir = args.save_path_dir
    data_path = args.data_dir
    gnn_type = str(args.gnn_type).lower() # gat or gcn

    save_path = save_path_dir+f'/sweep_{gnn_type}.npz'

    train_load_path = data_path+'/train_dataset.pt'
    val_load_path = data_path+'/val_dataset.pt'

    train_data = torch.load(train_load_path)
    val_data = torch.load(val_load_path)
    print(f'Train data loaded from {train_load_path}\nVal data loaded from {val_load_path}')

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=100, shuffle=True)

    initialize_files(save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using GPU')

    if gnn_type == 'gat':
        print(f'Running GAT and saving to {save_path}')
        sweep_gat(save_path, train_loader=train_loader, val_loader=val_loader, device=device)
    elif gnn_type == 'gcn':
        print(f'Running GCN and saving to {save_path}')
        sweep_gcn(save_path, train_loader=train_loader, val_loader=val_loader, device=device)

    print('Finished')

