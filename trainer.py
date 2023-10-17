import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ah_tactile_player import AHTactilePlayer

class Trainer:
    """
    Helper class to train convolutional neural network with datalodaer

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        batch_size (int): 
        stdev (float): 
        device (str): 
    """
    def __init__(self,
                model,
                optimizer,
                device='cpu'):

        self.device = device
        self.optimizer = optimizer        
        self.model = model.to(self.device)

    def save(self, epoch, loss, savename):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    #'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': loss[0],
                    'test_loss': loss[1],
                    }, savename)

    def process_epoch(self, data, batch_size, training=True):
        
        if not training:
            self.model.eval()

        total_loss = 0.0
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        for n_batch, (xi, yi, data_length, file_name) in enumerate(data_loader):
            xi = xi["tactile"]
            yi = yi["tactile"]
            # import ipdb; ipdb.set_trace()
            tmp_xi_list = []
            tmp_yi_list = []
            for i in range(len(xi)):
                tmp_xi = xi[i,:data_length[i],:]
                tmp_yi = yi[i,:data_length[i],:]
                tmp_xi_list.append(tmp_xi)
                tmp_yi_list.append(tmp_yi)
            xi = torch.cat(tmp_xi_list).to(self.device)
            yi = torch.cat(tmp_yi_list).to(self.device)
            # import ipdb; ipdb.set_trace()
            yi_hat, hid = self.model(xi)
            loss = nn.MSELoss()(yi_hat, yi)
            total_loss += loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
    
    def plot_prediction(self, dataset, scaling_df, batch_size, save_dir, seq_num=1, prefix=""):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        for n_batch, (x_data, y_data, data_length, file_name) in enumerate(data_loader):
            xt_batch = x_data["tactile"]
            yt_batch = y_data["tactile"]
            # import ipdb; ipdb.set_trace()
            for i in range(len(xt_batch)):
                # import ipdb; ipdb.set_trace()
                xt = xt_batch[i,:data_length[i],:].to(self.device)
                yt = yt_batch[i,:data_length[i],:].to("cpu").detach().numpy()
                yt_hat, hid = self.model(xt)
                # import ipdb; ipdb.set_trace()
                yt_hat = yt_hat.to("cpu").detach().numpy()
                yt = self.rescaling_data(yt, scaling_df, data_type="tactile")
                yt_hat = self.rescaling_data(yt_hat, scaling_df, data_type="tactile")

                original_csv_column = scaling_df.columns.values[1:]
                y_tac_df = self.convert_array2pandas(yt, original_csv_column)
                yt_hat_df = self.convert_array2pandas(yt_hat, original_csv_column)
                # import ipdb; ipdb.set_trace()
                save_title = file_name[i].split("/")[-1].replace(".csv", "")
                save_file = save_dir + save_title
                AHTactilePlayer([y_tac_df[seq_num:int(data_length[i])], yt_hat_df[:int(data_length[i])-seq_num]],
                                5, 0.6, save_file)

    def convert_array2pandas(self, array, column, data_type="tactile"):
        # import ipdb; ipdb.set_trace()
        data = np.zeros([len(array), len(column)])
        if data_type=="tactile":
            tac_index_st = np.where(column=="IndexTip_TactileB00C00X")[0][0]
            tac_index_en = np.where(column=='Palm_TactileB02C23Z')[0][0] + 1
            data[:, tac_index_st:tac_index_en] = array
        df = pd.DataFrame(data=data, columns=column)
        # import ipdb; ipdb.set_trace()
        return df    

    def rescaling_data(self, data, scaling_df, data_type="joint"):
        if "max" in scaling_df.index:
            mode = "normalization"
        elif "mean" in scaling_df.index:
            mode = "standardization"
        else:
            assert False, "scaling_df is invalid."
        if data_type=="joint":
            scaling_param = scaling_df.filter(regex="^Joint").values # Jointから始まる列
            if mode=="normalization":
                rescaled_data = data * (scaling_param[0] - scaling_param[1]) + scaling_param[1]
            if mode=="standardization":
                rescaled_data = data * scaling_param[1] + scaling_param[0]
        if data_type=="tactile":
            scaling_param = scaling_df.filter(regex="Tactile").values # Jointから始まる列
            if mode=="normalization":
                rescaled_data = data * (scaling_param[0] - scaling_param[1]) + scaling_param[1]
            if mode=="standardization":
                rescaled_data = data * scaling_param[1] + scaling_param[0]            
        return rescaled_data
            