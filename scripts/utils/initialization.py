import numpy as np

def initialize_files(path):
    hc_list = []
    nl_list = []
    do_list = []
    act_list = []
    loss_list = []
    epoch_list = []

    # for path in ['data/gat_sweep.npz', 'data/gcn_sweep.npz']:
    np.savez(path, hc_list=hc_list, nl_list=nl_list, do_list=do_list, act_list=act_list, loss_list=loss_list, epoch_list=epoch_list)