# fork from eipl
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import random
    
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
    
    def process_epoch(self, dataset, batch_size, seq_num=1, training=True):
        # import ipdb; ipdb.set_trace()
        if not training:
            self.model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        for n_batch, (x_data, y_data, data_length, file_name) in enumerate(data_loader):
        # for data in data_loader:
            # import ipdb; ipdb.set_trace()
            sequence_num = x_data["tactile"].shape[1]
            # x_tac = self.split_dataset(x_data["tactile"].to(self.device), batch_size=batch_size)
            # x_joint = self.split_dataset(x_data["joint"].to(self.device), batch_size=batch_size)
            x_tac = x_data["tactile"].to(self.device)
            y_tac = y_data["tactile"].to(self.device)
            x_joint = x_data["joint"].to(self.device)
            y_joint = y_data["joint"].to(self.device)
            state = None
            yt_list, yj_list = [], []
            # T = seq_num
            T = x_tac.shape[1]
            # import ipdb; ipdb.set_trace()
            for t in range(T-seq_num):
                _yt_hat, _yj_hat, state = self.model(x_tac[:,t], x_joint[:,t], state)
                yt_list.append(_yt_hat)
                yj_list.append(_yj_hat)
            yt_hat = torch.stack(yt_list).permute(1,0,2)
            yj_hat = torch.stack(yj_list).permute(1,0,2)

            # import ipdb; ipdb.set_trace()
            # yt_hat = torch.cat(yt_list)[:sequence_num]
            # yj_hat = torch.cat(yj_list)[:sequence_num]

            # calculate loss using actual data length
            for i in range(len(yt_hat)):
                if i==0:
                    loss = self.loss_weights[0]*nn.MSELoss()(yt_hat[i,:data_length[i]], y_tac[i,seq_num:data_length[i]+seq_num])\
                          + self.loss_weights[1]*nn.MSELoss()(yj_hat[i,:data_length[i]], y_joint[i,seq_num:data_length[i]+seq_num])
                else:
                    loss += self.loss_weights[0]*nn.MSELoss()(yt_hat[i,:data_length[i]], y_tac[i,seq_num:data_length[i]+seq_num])\
                          + self.loss_weights[1]*nn.MSELoss()(yj_hat[i,:data_length[i]], y_joint[i,seq_num:data_length[i]+seq_num])
            # loss = self.loss_weights[0]*nn.MSELoss()(yt_hat, y_tac[:,1:]) + self.loss_weights[1]*nn.MSELoss()(yj_hat, y_joint[:,1:])
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

    def plot_prediction(self, dataset, scaling_df, batch_size, save_dir, seq_num=1, prefix=""):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        for n_batch, (x_data, y_data, data_length, file_name) in enumerate(data_loader):
            sequence_num = x_data["tactile"].shape[1]
            x_tac = x_data["tactile"].to(self.device)
            y_tac = y_data["tactile"].to("cpu").detach().numpy()
            x_joint = x_data["joint"].to(self.device)
            y_joint = y_data["joint"].to("cpu").detach().numpy()
            state = None
            states = []
            yt_list, yj_list = [], []
            T = x_tac.shape[1]
            for t in range(T-seq_num):
                _yt_hat, _yj_hat, state = self.model(x_tac[:,t], x_joint[:,t], state)
                yt_list.append(_yt_hat)
                yj_list.append(_yj_hat)
                states.append(state[0])
            yt_hat = torch.stack(yt_list).permute(1,0,2).to("cpu").detach().numpy()
            yj_hat = torch.stack(yj_list).permute(1,0,2).to("cpu").detach().numpy()

            # import ipdb; ipdb.set_trace()
            y_joint = self.rescaling_data(y_joint, scaling_df, data_type="joint")
            yj_hat = self.rescaling_data(yj_hat, scaling_df, data_type="joint")
            for i in range(len(y_joint)):
                plt.figure(figsize=(15,5))
                # plt.plot(range(y_joint.shape[1]), y_joint[i, seq_num:], ":")
                # plt.plot(range(yj_hat.shape[1]), yj_hat[i], "-")
                plt.plot(range(data_length[i]-seq_num), y_joint[i, seq_num:data_length[i]], ":")
                plt.plot(range(data_length[i]-seq_num), yj_hat[i, :data_length[i]-seq_num], "-")
                # plt.show()
                save_title = file_name[i].split("/")[-1].replace(".csv", "")
                plt.title(save_title)
                save_file_name = save_dir + prefix + "_" + save_title
                plt.savefig(save_file_name + ".png")
            self.plot_pca(states, save_file_name + ".gif")
        return

    def plot_pca(self, states, save_file_name):
        import matplotlib.animation as anim
        from sklearn.decomposition import PCA
        import numpy as np
        states = torch.stack(states).permute(1,0,2)
        states = states.to("cpu").detach().numpy()
        N,T,D  = states.shape
        states = states.reshape(-1,D)
        # loop_ct = float(360)/T
        loop_ct = float(360)/100
        pca_dim = 3
        pca     = PCA(n_components=pca_dim).fit(states)
        pca_val = pca.transform(states)
        # Reshape the states from [-1, pca_dim] to [N,T,pca_dim] to
        # visualize each state as a 3D scatter.
        pca_val = pca_val.reshape( N, T, pca_dim )

        fig = plt.figure(dpi=60)
        ax = fig.add_subplot(projection='3d')

        def anim_update(i):
            ax.cla()
            angle = int(loop_ct * i)
            ax.view_init(30, angle)

            # c_list = ['C0','C1','C2','C3','C4']
            c_list = ["C{}".format(s) for s in range(N)]
            for n, color in enumerate(c_list):
                ax.scatter( pca_val[n,1:,0], pca_val[n,1:,1], pca_val[n,1:,2], color=color, s=3.0 )

            ax.scatter( pca_val[n,0,0], pca_val[n,0,1], pca_val[n,0,2], color='k', s=30.0 )
            pca_ratio = pca.explained_variance_ratio_ * 100
            ax.set_xlabel('PC1 ({:.1f}%)'.format(pca_ratio[0]) )
            ax.set_ylabel('PC2 ({:.1f}%)'.format(pca_ratio[1]) )
            ax.set_zlabel('PC3 ({:.1f}%)'.format(pca_ratio[2]) )

        # ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T/10)), frames=T)
        ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(100/10)), frames=100)
        # ani.save( './output/PCA_{}.gif'.format(save_file_name) )
        ani.save(save_file_name)
    def plot_prediction_(self, data, scaling_df, batch_size, save_dir):
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
            