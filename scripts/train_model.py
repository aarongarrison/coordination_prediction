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

from pathlib import Path


#Load model
# gat_path = ''

# gat = torch_geometric.nn.GAT(-1, 20, num_layers=2, out_channels=1, dropout=0.5)
# gat.load_state_dict(torch.load(gat_path))
# gat.eval()

# gcn_path = ''

# gcn = torch_geometric.nn.GCN(-1, 10, num_layers=2, out_channels=1, dropout=0.5)
# gcn.load_state_dict(torch.load(gcn_path))
# gcn.eval()

def train_model_save_params(model_save_path, model, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using GPU')

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    epoch, train_epoch_losses, val_epoch_losses, best_loss = run_train(model=model,
                                                                       train_loader=train_loader,
                                                                       val_loader=val_loader,
                                                                       num_epochs=50,
                                                                       patience=5,
                                                                       optimizer=optimizer,
                                                                       scheduler=scheduler,
                                                                       device=device,
                                                                       verbose=True)
    print(f'Finished training. Best loss: {best_loss} at epoch {epoch+1}.')
    torch.save(model.state_dict(), model_save_path)
    print(f'Saved params to {model_save_path}')

    model_save_path = Path(model_save_path)
    train_file_name = model_save_path.stem + '_losses.npz'
    losses_path = model_save_path.parent / train_file_name

    np.savez(losses_path, train_losses=train_epoch_losses, val_losses=val_epoch_losses)
    print(f'Saved losses from training to {losses_path}')

    
def load_model(model_type: str, hidden_dim, num_layers, dropout):
    """ Load the provided model with specified hyperparamters. """
    if model_type.lower() == 'gat':
        model = GAT(-1, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=1, dropout=dropout, act='relu')
        print('GAT Loaded')

    elif model_type.lower() == 'gcn':
        model = GCN(-1, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=1, dropout=dropout, act='relu')
        print('GCN Loaded')
    return model


if __name__=='__main__':
    # argparse processing
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path-dir', type=str, help='Path to directory to save')
    parser.add_argument('--data-dir', type=str, help="Path to directory holding train and val torch Datasets.")
    parser.add_argument('--gnn-type', type=str, help='Either "GAT" or "GCN".')
    args = parser.parse_args()

    save_path_dir = args.save_path_dir
    data_path = args.data_dir
    gnn_type = str(args.gnn_type).lower() # gat or gcn

    # Where to save model parameters
    save_path = save_path_dir+f'/params_{gnn_type}.pt'

    # Get data
    train_load_path = data_path+'/train_dataset_smiles.pt'
    val_load_path = data_path+'/val_dataset_smiles.pt'

    train_data = torch.load(train_load_path)
    val_data = torch.load(val_load_path)
    print(f'Train data loaded from {train_load_path}\nVal data loaded from {val_load_path}')

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=100, shuffle=True)

    # Both GAT and GCN were optimized at these hyperparameters
    model = load_model(gnn_type, hidden_dim=20, num_layers=2, dropout=0.1)

    print(f'Running {gnn_type.upper()} and saving to {save_path}')

    train_model_save_params(save_path, model, train_loader=train_loader, val_loader=val_loader)

    print('Finished')

