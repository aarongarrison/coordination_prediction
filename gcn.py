#Hyperparameter sweep
import numpy as np
val_loss = 0
for hidden_channels in [5, 10, 15, 20, 25]:
    for num_layers in [2, 4, 6, 8, 10]:
        for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for activation in ['tanh', 'relu']:
                #Implement training / evaluation loop here - need to calculate val loss

                #open the saved files
                path = 'data/gcn_sweep.npz'
                npzfile = np.load(path, allow_pickle=True)
                hc_list = npzfile['hc_list']
                nl_list = npzfile['nl_list']
                do_list = npzfile['do_list']
                act_list = npzfile['act_list']
                loss_list = npzfile['loss_list']

                #append the parameters
                hc_list = np.append(hc_list, hidden_channels)
                print(hc_list, hidden_channels)
                nl_list = np.append(nl_list, num_layers)
                do_list = np.append(do_list, dropout)
                act_list = np.append(act_list, activation)
                loss_list = np.append(loss_list, val_loss)

                #save parameters
                np.savez(path, hc_list=hc_list, nl_list=nl_list, do_list=do_list, act_list=act_list, loss_list=loss_list)
                