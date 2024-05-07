import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
import torch_scatter


def compute_batch_loss(preds: torch.Tensor, labels: torch.Tensor, inds: torch.Tensor):
    """
    Computes the cross-entropy loss with separate weighting for positive and negative samples.

    Parameters
    ----------
    preds : torch.Tensor (N,1)
        Atom-wise predicted logits for not-being or being a coordinating atom
    labels : torch.Tensor (N,1)
        Atom-wise labels for whether it isn't or is a coordinating atom
    inds : torch.Tensor (batch_size+1)
        The indices defining the ligands within each batch. Uses the batch.ptr generated by the torch_geometric dataloader.
    Return
    ------
    torch.Tensor (1,)
        Mean batch loss
    """
    # Compute cross-entropy per atom
    loss_per_node = torch.nn.functional.binary_cross_entropy(preds, labels, reduction='none',
                                                             weight=torch.Tensor([1]).to(preds.device))
    
    # Get how many ones/zeros are in each individual graph
    num_ones_per_graph = torch.Tensor([len(labels[inds[i-1]:inds[i]].nonzero()) for i in range(1,len(inds))],
                                     ).to(torch.long)
    num_zeros_per_graph = torch.Tensor([len(torch.where(labels[inds[i-1]:inds[i]]==0)[0]) for i in range(1,len(inds))],
                                     ).to(torch.long)
    # Get the ids for use of scatter mean
    ones_seg_ids = torch.repeat_interleave(torch.arange(len(num_ones_per_graph)), num_ones_per_graph).to(preds.device)
    zeros_seg_ids = torch.repeat_interleave(torch.arange(len(num_zeros_per_graph)), num_zeros_per_graph).to(preds.device)
    # Compute mean loss for each pos/neg for each graph
    pos_loss = torch_scatter.scatter_mean(loss_per_node[labels.flatten().nonzero().flatten()], ones_seg_ids, dim=0)
    neg_loss = torch_scatter.scatter_mean(loss_per_node[torch.where(labels==0)[0]], zeros_seg_ids, dim=0)
    # Combine the loss
    combined_loss_per_graph = pos_loss + neg_loss # element-wise for each graph
    
    return (combined_loss_per_graph.mean()) # mean across the batch


def predict(model: torch.nn.Module, batch, device):
    """ Predice node-wise probabilities of being a coordinating atom given a batch. """
    model.to(device)
    
    batch = batch.to(device)
    
    out_logits = model(x=batch.x, edge_index=batch.edge_index.to(torch.int64), edge_attr=batch.edge_attr)
    out_probs = torch.nn.functional.sigmoid(out_logits)
    return out_probs


def run_train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, patience: int, optimizer, scheduler, device, verbose: bool=False):
    """
    Training loop with early-stopping implemented

    Parameters
    ----------
    model : torch_geometric.nn.model
        Pytorch model to train.
    train_loader : torch_geometric.loader.DataLoader
        DataLoader with training set.
    val_loader : torch_geometric.loader.DataLoader
        DataLoader with validation set.
    num_epochs : int
        Max number of epochs to train.
    patience : int
        Number of epochs to wait for early stopping
    optimizer : torch.optim optimizer
        Optimizer object
    scheduler : torch.optim lr_schedular
        Schedular object
    device : torch.device
        Device type
    verbose : bool (default=False)
        Whether to print progress

    Returns
    -------
    tuple (length = 4)
        int : Final epoch (started at 0)
        list[float] : Training epoch losses (length epoch+1)
        list[float] : Validation epoch losses (length epoch+1)
        float : Best validation loss
    """
    model.to(device)

    train_epoch_losses = []
    val_epoch_losses = []

    best_loss = 10000 # large number
    # Training Loop
    for epoch in range(num_epochs):
        epoch_train_loss = 0
        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)

            out_logits = model(x=batch.x, edge_index=batch.edge_index.to(torch.int64), edge_attr=batch.edge_attr)
            # Switch to probabilities for the loss function
            out_probs = torch.nn.functional.sigmoid(out_logits)
            loss = compute_batch_loss(out_probs, batch.y.to(torch.float32), batch.ptr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        scheduler.step()

        epoch_train_loss = epoch_train_loss / (i+1)
        train_epoch_losses.append(epoch_train_loss)

        # Validation
        epoch_val_loss = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = batch.to(device)
                out_logits = model(x=batch.x, edge_index=batch.edge_index.to(torch.int64), edge_attr=batch.edge_attr)
                out_probs = torch.nn.functional.sigmoid(out_logits)
                loss = compute_batch_loss(out_probs, batch.y.to(torch.float32), batch.ptr)
                epoch_val_loss += loss.item()
        epoch_val_loss = epoch_val_loss / (i+1)
        val_epoch_losses.append(epoch_val_loss)

        if verbose:
            print(f'{model.__class__.__name__} | Epoch: {epoch+1} | Avg Train Loss: {epoch_train_loss:.3} | Avg Val Loss: {epoch_val_loss:.3}')

        # Early stopping
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            patience = 5  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                break

        # Load the best model weights
        model.load_state_dict(best_model_weights)

    return epoch, train_epoch_losses, val_epoch_losses, best_loss