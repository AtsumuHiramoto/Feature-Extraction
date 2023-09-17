# fork from eipl
import torch
import torch.nn as nn
import math
    
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
        for n_batch, (x_data, y_data) in enumerate(data):
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