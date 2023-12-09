import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
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
                input_data,
                output_data,
                model,
                optimizer,
                device='cpu',
                tactile_scale=None):
        
        self.input_data = input_data
        self.output_data = output_data

        self.device = device
        self.optimizer = optimizer        
        self.model = model.to(self.device)
        self.finger_range = [[i for i in range(0, 78*3)],
                             [i for i in range(78*3, 156*3)],
                             [i for i in range(156*3, 234*3)],
                             [i for i in range(234*3, 296*3)],
                             [i for i in range(296*3, 368*3)]]
        self.tactile_scale = tactile_scale

    def save(self, epoch, loss, savename):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    #'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': loss[0],
                    'test_loss': loss[1],
                    }, savename)

    def process_epoch(self, data, batch_size, 
                      finger_loss=[1.0, 1.0, 1.0, 1.0, 1.0], training=True):
        
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
            # loss = nn.MSELoss()(yi_hat, yi)
            # import ipdb; ipdb.set_trace()
            if "thumb" in self.input_data:
                loss = finger_loss[3]*nn.MSELoss()(yi_hat, yi) # only thumb
            else:
                loss = finger_loss[0]*nn.MSELoss()(yi_hat[:,self.finger_range[0]], yi[:,self.finger_range[0]])\
                + finger_loss[1]*nn.MSELoss()(yi_hat[:,self.finger_range[1]], yi[:,self.finger_range[1]])\
                + finger_loss[2]*nn.MSELoss()(yi_hat[:,self.finger_range[2]], yi[:,self.finger_range[2]])\
                + finger_loss[3]*nn.MSELoss()(yi_hat[:,self.finger_range[3]], yi[:,self.finger_range[3]])\
                + finger_loss[4]*nn.MSELoss()(yi_hat[:,self.finger_range[4]], yi[:,self.finger_range[4]])
            
            total_loss += loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
    
    def plot_prediction(self, dataset, scaling_df, batch_size, save_dir, seq_num=1, prefix=""):
        # batch_size = 100
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        for n_batch, (x_data, y_data, data_length, file_name) in enumerate(data_loader):
            pose_list, switch_list, hid_list = [], [], []
            xt_batch = x_data["tactile"]
            yt_batch = y_data["tactile"]
            # import ipdb; ipdb.set_trace()
            for i in range(len(xt_batch)):
                # import ipdb; ipdb.set_trace()
                xt = xt_batch[i,:data_length[i],:].to(self.device)
                yt = yt_batch[i,:data_length[i],:].to("cpu").detach().numpy()
                yt_hat, hid = self.model(xt)
                hid = hid.to("cpu").detach().numpy()
                # import ipdb; ipdb.set_trace()
                yt_hat = yt_hat.to("cpu").detach().numpy()
                # if "thumb" not in self.input_data:
                if False:
                    yt = self.rescaling_data(yt, scaling_df, data_type="tactile")
                    yt_hat = self.rescaling_data(yt_hat, scaling_df, data_type="tactile")

                    plt.figure(figsize=(15,5))
                    plt.plot(range(len(hid)), hid)
                    save_title = file_name[i].split("/")[-1].replace(".csv", "")
                    plt.title(save_title)
                    save_file_name = save_dir + prefix + "_tacF_" + save_title
                    plt.savefig(save_file_name + ".png")

                    original_csv_column = scaling_df.columns.values[1:]
                    y_tac_df = self.convert_array2pandas(yt, original_csv_column)
                    yt_hat_df = self.convert_array2pandas(yt_hat, original_csv_column)
                    # import ipdb; ipdb.set_trace()
                    save_title = file_name[i].split("/")[-1].replace(".csv", "")
                    save_file = save_dir + save_title
                    AHTactilePlayer([y_tac_df[seq_num:int(data_length[i])], yt_hat_df[:int(data_length[i])-seq_num]],
                                    5, 0.6, save_file)
                if True:
                    pose_command = x_data["pose"][i,:data_length[i],0].to("cpu").detach().numpy()
                    switching_point = x_data["switching"][i,:data_length[i],0].to("cpu").detach().numpy()
                    switching_point[-1] = -1
                    save_title = file_name[i].split("/")[-1].replace(".csv", "")
                    save_file_name = save_dir + prefix + "_pca_" + save_title
                    # self.plot_pca2D(hid, pose_command, switching_point, save_file_name)
                    pose_list.append(pose_command)
                    # import ipdb; ipdb.set_trace()
                    switch_list.append(switching_point)
                    hid_list.append(hid)
            # import ipdb; ipdb.set_trace()
        pose_list = np.concatenate(pose_list)
        switch_list = np.concatenate(switch_list)
        hid_list = np.concatenate(hid_list)
        save_file_name = save_dir + prefix + "_pca_all_" + save_title
        self.plot_pca2D(hid_list, pose_list, switch_list, save_file_name, all=False)
                

    def plot_pca2D(self, hid, pose_command, switching_point, save_file_name, all=True):
        # import ipdb; ipdb.set_trace()
        from sklearn.decomposition import PCA
        pca_dim = 2
        pca     = PCA(n_components=pca_dim).fit(hid)
        pca_val = pca.transform(hid)
        plt.figure(figsize=(10,10))
        color_list=[]
        for t in range(pca_val.shape[0]):
            if all==False:
                if switching_point[t] not in [-1,3,14]:
                # if switching_point[t] not in [-1,14]:
                    # continue
                    if t!=0 and t!=pca_val.shape[1]-1:
                        continue
            print(t, "/", pca_val.shape[0]-1)
            # import ipdb; ipdb.set_trace()
            tmp_pose = pose_command[t]
            size = 4
            alpha = 0.5
            if tmp_pose in [0,1,2]:
                color="grey"
                label="first closing"
            elif tmp_pose in [3]:
                color="pink"
                label="closing"
            elif tmp_pose in [4]:
                color="yellowgreen"
                label="return thumb"
            elif tmp_pose in [10,11,12,13,14]:
                color="green"
                label="open"
            elif tmp_pose in [20,21,22,23,24]:
                color="skyblue"
                label="slide to left"
            elif tmp_pose in [30,31,32,33,34]:
                color="purple"
                label="slide to right"
            # if t > 0:
            #     plt.plot([pca_val[t-1,0],pca_val[t,0]], [pca_val[t-1,1],pca_val[t,1]],color=color,alpha=0.3)
            if t==0:
                alpha=1.0
                color="black"
                label="start"
                size = 40
            # if t==pca_val.shape[1]-1:
            if switching_point[t]==-1:
                alpha=1.0
                color="red"
                label="end"
                size = 40
            if switching_point[t]==3:
                alpha=1.0
                color="blue"
                label="switching point(closing)"
                size = 40
            if switching_point[t]==14:
                alpha=1.0
                color="orange"
                # color="blue"
                label="switching point(opening)"
                size = 40
            if color not in color_list:
                plt.scatter(pca_val[t,0], pca_val[t,1],color=color, s=size, label=label, alpha=alpha)
                color_list.append(color)
            else:
                plt.scatter(pca_val[t,0], pca_val[t,1],color=color, s=size, alpha=alpha)
        plt.legend()
        print(save_file_name)
        plt.savefig(save_file_name)
        plt.close()

    
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
            if self.tactile_scale is not None:
                if self.tactile_scale=="sqrt":
                    rescaled_data = rescaled_data * np.abs(rescaled_data)
                if self.tactile_scale=="log":
                    rescaled_data = math.e ** (np.abs(rescaled_data.astype(float))) * rescaled_data / np.abs(rescaled_data) - 1.0
        return rescaled_data
            