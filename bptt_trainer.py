# fork from eipl
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
    
class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self,
                model,
                optimizer,
                loss_weights=[1.0, 1.0],
                device='cpu'):

        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.model = model.to(self.device)

    def save(self, epoch, loss, savename):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    #'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': loss[0],
                    'test_loss': loss[1],
                    }, savename)

    def split_dataset(self, data, batch_size=100):
        # import ipdb; ipdb.set_trace()
        batch_num = math.ceil(data.shape[0] / batch_size)
        # batch_num = math.ceil(data.shape[1] / batch_size)
        data_list = []
        for i in range(batch_num):
            batch_from = i * batch_size
            batch_to = (i + 1) * batch_size
            split_data = data[batch_from:batch_to, :]
            if split_data.shape[0] < batch_size:
                copy_dim = batch_size-split_data.shape[0]
                copied_data = split_data[-1,:].repeat((copy_dim,1))
                split_data = torch.cat([split_data, copied_data])
            data_list.append(split_data)
        # import ipdb; ipdb.set_trace()
        return data_list
    
    def process_epoch(self, data, batch_size, training=True):
        # import ipdb; ipdb.set_trace()
        if not training:
            self.model.eval()

        total_loss = 0.0
        for n_batch, (x_data, y_data, _) in enumerate(data):
            sequence_num = x_data["tactile"].shape[0]
            x_tac = self.split_dataset(x_data["tactile"].to(self.device), batch_size=batch_size)
            x_joint = self.split_dataset(x_data["joint"].to(self.device), batch_size=batch_size)
            y_tac = y_data["tactile"].to(self.device)
            y_joint = y_data["joint"].to(self.device)
            state = None
            yt_list, yj_list = [], []
            T = len(x_tac)
            for t in range(len(x_tac)):
                _yt_hat, _yj_hat, state = self.model(x_tac[t], x_joint[t], state)
                yt_list.append(_yt_hat)
                yj_list.append(_yj_hat)

            # state = None
            # yt_list, yj_list = [], []
            # T = x_tac.shape[1]
            # for t in range(T-1):
            #     # import ipdb; ipdb.set_trace()
            #     _yt_hat, _yj_hat, state = self.model(x_tac[:,t], x_joint[:,t], state)
            #     yt_list.append(_yt_hat)
            #     yj_list.append(_yj_hat)

            # import ipdb; ipdb.set_trace()
            yt_hat = torch.cat(yt_list)[:sequence_num]
            yj_hat = torch.cat(yj_list)[:sequence_num]
            loss = self.loss_weights[0]*nn.MSELoss()(yt_hat[:-1,:], y_tac[1:,:]) + self.loss_weights[1]*nn.MSELoss()(yj_hat[:-1,:], y_joint[1:,:])
            # yt_hat = torch.stack(yt_list).permute(2,0,1)[:,:,0]
            # yt_hat = torch.permute(torch.stack(yt_list), (1,0,2,3,4) )
            # yj_hat = torch.stack(yj_list).permute(2,0,1)[:,:,0]
            # yj_hat = torch.permute(torch.stack(yj_list), (1,0,2) )
            # loss = self.loss_weights[0]*nn.MSELoss()(yt_hat, y_tac[:,1:] ) + self.loss_weights[1]*nn.MSELoss()(yj_hat, y_joint[:,1:] )
            total_loss += loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch+1)
    
    def plot_prediction(self, data, scaling_df, batch_size, save_dir):
        # import ipdb; ipdb.set_trace()
        self.model.eval()

        for n_batch, (x_data, y_data, file_name) in enumerate(data):
            sequence_num = x_data["tactile"].shape[0]
            x_tac = self.split_dataset(x_data["tactile"].to(self.device), batch_size=batch_size)
            x_joint = self.split_dataset(x_data["joint"].to(self.device), batch_size=batch_size)
            y_tac = y_data["tactile"].to("cpu").detach().numpy()
            y_joint = y_data["joint"].to("cpu").detach().numpy()
            state = None
            yt_list, yj_list = [], []
            T = len(x_tac)
            for t in range(len(x_tac)):
                _yt_hat, _yj_hat, state = self.model(x_tac[t], x_joint[t], state)
                yt_list.append(_yt_hat)
                yj_list.append(_yj_hat)
            yt_hat = torch.cat(yt_list)[:sequence_num].to("cpu").detach().numpy()
            yj_hat = torch.cat(yj_list)[:sequence_num].to("cpu").detach().numpy()
            plt.figure(figsize=(15,5))
            # import ipdb; ipdb.set_trace()
            y_joint = self.rescaling_data(y_joint, scaling_df, data_type="joint")
            yj_hat = self.rescaling_data(yj_hat, scaling_df, data_type="joint")
            plt.plot(range(y_joint.shape[0]-1), y_joint[1:,:], ":")
            plt.plot(range(yj_hat.shape[0]-1), yj_hat[:-1,:], "-")
            # plt.show()
            save_file_name = file_name.split("/")[-1].replace(".csv", "")
            plt.title(save_file_name)
            plt.savefig(save_dir + save_file_name + ".png")

        return 
    
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
            