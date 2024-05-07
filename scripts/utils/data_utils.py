import numpy as np

import torch
from torch_geometric.data import Data


class LigandDataset():
    def __init__(self, atom_list, natom_list, edge_index_list, edge_feats_list, y_list, denticities_list):
        self.atom_list = atom_list
        self.natom_list = natom_list
        self.edge_index_list = edge_index_list
        self.edge_feats_list = edge_feats_list
        self.y_list = y_list
        self.denticities_list = denticities_list
        
    def __len__(self):
        return len(self.atom_list)
        
    def __getitem__(self, idx):
        return Data(x=torch.Tensor(self.atom_list[idx]),
                    natoms=torch.Tensor([self.natom_list[idx]]),
                    edge_index=torch.Tensor(np.array(self.edge_index_list[idx])),
                    edge_attr=torch.Tensor(self.edge_feats_list[idx]),
                    y=torch.Tensor(self.y_list[idx]).unsqueeze(1).to(torch.long),
                    denticity=torch.Tensor([self.denticities_list[idx]]),
                    # y=torch.nn.functional.one_hot(torch.Tensor(self.y_list[idx]).to(torch.long), num_classes=2) # one-hot
                   )